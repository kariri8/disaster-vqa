"""
scripts/03_train_student.py
────────────────────────────────────────────────────────────────
Phase 2: Train Moondream2 (Student) with three-loss knowledge
distillation from the LLaVA Teacher.
 
Loss breakdown
──────────────
  L_task    (α)  — Cross-entropy on answer tokens (student VQA loss)
  L_feature (β)  — MSE between projected vision-encoder outputs
                   (teacher 4096-d → proj_dim, student 2048-d → proj_dim)
  L_KD      (γ)  — KL divergence on softened answer logits (T=4)
 
  L_total = α·L_task + β·L_feature + γ·L_KD
 
Feature alignment layer
───────────────────────
We align the mean-pooled output of each model's vision encoder
(before the LM) because:
  • It is purely visual — no positional shift between seq lengths.
  • The teacher's CLIP features contain strong bi-temporal change signal
    that we want to transfer to the student.
  • LM hidden states are vocabulary-dependent; aligning them would
    require matching the full context length, which is unstable.
 
QLoRA is applied to Moondream2's attention projections so that only
~1 % of parameters are updated during distillation.
 
Usage:
    python scripts/03_train_student.py [--config PATH] [--debug]
"""
 
import argparse
import json
import logging
import os
import random
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional
 
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm.auto import tqdm
 
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs.config import Config, cfg
from scripts.datasets import StudentVQADataset, load_pairs_and_splits
from scripts.models import (
    DEVICE, DTYPE,
    FeatureProjector,
    build_optimizer_scheduler,
    build_projectors,
    load_student,
    load_teacher,
)
from scripts.utils import (
    EarlyStopping, MetricLogger, parse_answer, save_ckpt,
)
 
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
 
 
# ════════════════════════════════════════════════════════════════
# Teacher feature extraction
# ════════════════════════════════════════════════════════════════
 
@torch.no_grad()
def get_teacher_features(teacher, processor, teacher_image: Image.Image,
                         question: str, cfg: Config):
    """
    Forward one sample through Teacher.
    Returns:
        vision_pool : mean-pooled vision encoder output  [1, teacher_hidden]
        logits      : LM output logits (for KL)
    """
    teacher.eval()
    prompt = (
        f"USER: <image>\n"
        f"This image shows the same area before (left) and after (right) a disaster.\n"
        f"{question}\nASSISTANT:"
    )
    inputs = processor(
        text=[prompt], images=[teacher_image],
        return_tensors="pt", 
        padding=True, 
        truncation=True, 
        max_length=1024,
        do_resize=True,
        size={"height": 336, "width": 336},  # Increased to fit 576 image tokens + text
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
 
    outputs = teacher(**inputs, output_hidden_states=True)
 
    # Vision pool: mean over vision token positions in the last hidden state
    # LLaVA interleaves image tokens at the start — take the first 576 positions
    # (CLIP ViT-L/14-336 → 576 patches)
    last_hs = outputs.hidden_states[-1]          # [1, seq_len, 4096]
    n_vision = min(576, last_hs.shape[1])
    vision_pool = last_hs[:, :n_vision, :].mean(dim=1)  # [1, 4096]
 
    return vision_pool, outputs.logits
 
 
def build_teacher_image(item: dict, cfg: Config) -> Optional[Image.Image]:
    """Construct the side-by-side pre+post image for the teacher."""
    if not item.get("pre_ok") or not os.path.isfile(item.get("pre_path", "")):
        return None
    sz = cfg.image_size
    pre = Image.open(item["pre_path"]).convert("RGB").resize((sz, sz))
    post = Image.open(item["post_path"]).convert("RGB").resize((sz, sz))
    canvas = Image.new("RGB", (sz * 2, sz))
    canvas.paste(pre, (0, 0))
    canvas.paste(post, (sz, 0))
    return canvas
 
 
# ════════════════════════════════════════════════════════════════
# Student forward helpers
# ════════════════════════════════════════════════════════════════
 
def student_forward(model, processor, image, question, answer):
    prompt = f"USER: <image>\n{question}\nASSISTANT: {answer}"

    inputs = processor(
        text=[prompt],
        images=[image],
        return_tensors="pt",
    ).to(model.device)

    actual_model = model.base_model.model if hasattr(model, "base_model") else model

    # Navigating the Hugging Face LLaVA hierarchy safely
    if hasattr(actual_model, "model") and hasattr(actual_model.model, "vision_tower"):
        vision_tower = actual_model.model.vision_tower
    else:
        vision_tower = actual_model.vision_tower

    # Extract vision pool — drop CLS token at position 0
    with torch.no_grad():
        vision_out = vision_tower(
            inputs["pixel_values"].to(dtype=DTYPE),
            output_hidden_states=True
        )
        patch_tokens = vision_out.last_hidden_state[:, 1:, :]  # drop CLS
        vision_pool = patch_tokens.mean(dim=1)                 # [1, hidden_dim]

    outputs = model(**inputs)

    shift_logits = outputs.logits[..., :-1, :].contiguous()
    ans_tokens   = inputs["input_ids"][..., 1:].contiguous()

    return shift_logits, vision_pool, ans_tokens
 
 
# ════════════════════════════════════════════════════════════════
# Combined distillation loss
# ════════════════════════════════════════════════════════════════
 
def distillation_loss(shift_logits, vision_pool, ans_tokens,
                      t_vision_pool, t_logits,
                      s_proj, t_proj,
                      cfg, has_teacher, alpha=None, beta=None, gamma=None):

    # 1. Task loss
    alpha = alpha if alpha is not None else cfg.alpha
    beta  = beta  if beta  is not None else cfg.beta
    gamma = gamma if gamma is not None else cfg.gamma

    l_task = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        ans_tokens.reshape(-1),
    )

    l_feature = torch.tensor(0.0, device=DEVICE)
    l_kd = torch.tensor(0.0, device=DEVICE)

    if has_teacher:
        # 2. Feature alignment
        s_feat = s_proj(vision_pool)
        t_feat = t_proj(t_vision_pool.detach()).detach()  # detach t_proj too
        l_feature = 1 - F.cosine_similarity(
            F.normalize(s_feat, dim=-1),
            F.normalize(t_feat, dim=-1),
            dim=-1
        ).mean()

        l_kd = torch.tensor(0.0, device=DEVICE) 

    total = alpha * l_task + beta * l_feature + gamma * l_kd
    return total, l_task, l_feature, l_kd
 
 
# ════════════════════════════════════════════════════════════════
# Training loop
# ════════════════════════════════════════════════════════════════

def get_loss_weights(epoch, total_epochs, cfg, current_task_loss=None):
    warmup_epochs = max(1, int(total_epochs * 0.3))
    scale = min(1.0, epoch / warmup_epochs)

    alpha = cfg.alpha
    gamma = 0.0

    if current_task_loss is not None:
        # beta scales so feature loss ≈ 20% of task loss
        beta = (current_task_loss * 0.2) * scale
    else:
        beta = cfg.beta * scale

    return alpha, beta, gamma
 
def train_student(student, tokenizer, teacher, teacher_proc,
                  train_ds, val_ds, s_proj, t_proj,
                  cfg: Config, dirs: dict):
    logger.info("=" * 65)
    logger.info("  PHASE 2 — STUDENT DISTILLATION  (Moondream2 ← LLaVA)")
    logger.info("=" * 65)
 
    # Freeze teacher entirely
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
 
    # Optimise student (QLoRA params) + both projectors
    opt_params = (
        [p for p in student.parameters() if p.requires_grad]
        + list(s_proj.parameters())
        + list(t_proj.parameters())
    )
    optimizer, scheduler = build_optimizer_scheduler(
        opt_params, len(train_ds), cfg.student_batch_size,
        cfg.student_grad_accum, cfg.student_epochs, cfg,
    )
 
    scaler = torch.amp.GradScaler("cuda", enabled=(DTYPE == torch.float16))
    early = EarlyStopping(cfg.patience, mode="max")
    mlog = MetricLogger(os.path.join(dirs["logs"], "student_distill_log.json"))
    best_acc = 0.0
    best_path = os.path.join(dirs["student_ckpt"], "best.pt")
    avg_task_prev = 9.0
 
    for epoch in range(cfg.student_epochs):
        alpha, beta, gamma = get_loss_weights(
            epoch, cfg.student_epochs, cfg,
            current_task_loss=avg_task_prev
        )
        logger.info(f"  Ep {epoch+1} weights: α={alpha:.2f} β={beta:.3f} γ={gamma:.3f}")
        student.train(); s_proj.train(); t_proj.train()
        indices = list(range(len(train_ds)))
        random.shuffle(indices)
        epoch_losses = defaultdict(float)
        n_steps = 0
        optimizer.zero_grad()
 
        pbar = tqdm(range(0, len(indices), cfg.student_batch_size),
                    desc=f"Student Ep {epoch+1}/{cfg.student_epochs}")
 
        for start in pbar:
            batch_idx = indices[start:start + cfg.student_batch_size]
            batch = [train_ds[i] for i in batch_idx]
            batch_loss = torch.tensor(0.0, device=DEVICE, requires_grad=True)
 
            for item in batch:
                teacher_img = build_teacher_image(item, cfg)
                has_teacher = teacher_img is not None
 
                if has_teacher:
                    t_vision_pool, t_logits = get_teacher_features(
                        teacher, teacher_proc, teacher_img, item["question"], cfg)
                else:
                    t_vision_pool = t_logits = None
 
                with torch.amp.autocast("cuda", dtype=DTYPE):
                    shift_logits, vision_pool, ans_tokens = student_forward(
                        student, tokenizer, item["image"],
                        item["question"], item["answer"],
                    )
                    loss, l_task, l_feat, l_kd = distillation_loss(
                        shift_logits, vision_pool, ans_tokens,
                        t_vision_pool, t_logits,
                        s_proj, t_proj, cfg, has_teacher,
                        alpha=alpha, beta=beta, gamma=gamma,   # <-- add these params
                    )

                    # if n_steps == 1:
                    #     logger.info(f"  Loss breakdown — task={l_task.item():.4f}  "
                    #                 f"feat={l_feat.item():.4f}  kd={l_kd.item():.4f}")
 
                batch_loss = batch_loss + loss / (len(batch) * cfg.student_grad_accum)
                epoch_losses["task"] += l_task.item()
                epoch_losses["feature"] += l_feat.item()
                epoch_losses["kd"] += l_kd.item()
                epoch_losses["total"] += loss.item()
 
            scaler.scale(batch_loss).backward()
            n_steps += 1
 
            if n_steps % cfg.student_grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(opt_params, cfg.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
 
            pbar.set_postfix(loss=f"{batch_loss.item():.4f}")
 
        avg = {k: v / max(n_steps * cfg.student_batch_size, 1)
               for k, v in epoch_losses.items()}
        avg_task_prev = avg.get("task", avg_task_prev)
 
        # Validation
        student.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for vi in range(min(len(val_ds), 200)): # Reduced to 200 for speed
                item = val_ds[vi]
                # Prepare inference prompt
                inf_prompt = f"USER: <image>\n{item['question']}\nASSISTANT:"
                inf_inputs = tokenizer(text=inf_prompt, images=item["image"], return_tensors="pt").to(DEVICE)
                
                try:
                    output_ids = student.generate(
                        **inf_inputs, 
                        max_new_tokens=cfg.max_answer_tokens,
                        do_sample=False
                    )
                    # Decode only the new tokens
                    pred_text = tokenizer.decode(output_ids[0][inf_inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                    
                    if parse_answer(pred_text) == item["answer"].lower():
                        correct += 1
                except Exception as e:
                    logger.debug(f"Val error: {e}")
                total += 1

                
        val_acc = correct / max(total, 1)
        metrics = {"epoch": epoch + 1, **avg, "val_acc": val_acc}
        mlog.log(metrics)
        logger.info(
            f"  Ep {epoch+1}  total={avg.get('total',0):.4f}  "
            f"task={avg.get('task',0):.4f}  feat={avg.get('feature',0):.4f}  "
            f"kd={avg.get('kd',0):.4f}  val_acc={val_acc:.4f}"
        )
 
        if val_acc > best_acc:
            best_acc = val_acc
            save_ckpt(student, optimizer, epoch + 1, metrics, best_path, s_proj)
            logger.info(f"    ★ Best student (acc={best_acc:.4f})")
 
        if early(val_acc):
            logger.info(f"  Early stop at epoch {epoch+1}")
            break
 
    logger.info("Student distillation complete.")
    return student
 
 
# ════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════
 
def main(args):
    cfg_local = Config.load(args.config) if args.config else cfg
    if args.debug:
        cfg_local.debug_mode = True
 
    meta_dir = os.path.join(cfg_local.project_root, "metadata")
    dirs = {
        "student_ckpt": os.path.join(cfg_local.project_root, "checkpoints/student"),
        "logs": os.path.join(cfg_local.project_root, "logs"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
 
    all_pairs, splits = load_pairs_and_splits(meta_dir)
 
    teacher, teacher_proc = load_teacher(cfg_local)
    # Load best teacher LoRA weights if available
    teacher_lora_path = os.path.join(
        cfg_local.project_root, "checkpoints/teacher/best_lora")
    if os.path.isdir(teacher_lora_path):
        teacher, teacher_proc = load_teacher(cfg_local, apply_lora=False)  # no LoRA yet
        from peft import PeftModel
        teacher = PeftModel.from_pretrained(teacher, teacher_lora_path)    # single wrap
    else:
        teacher, teacher_proc = load_teacher(cfg_local, apply_lora=True)
 
    student, tokenizer = load_student(cfg_local, apply_qlora=True)
    from scripts.models import get_student_vision_hidden
    student_hidden = get_student_vision_hidden(student)
    print(f"Detected student vision hidden dim: {student_hidden}")
    t_proj, s_proj = build_projectors(cfg_local, student_hidden)
 
    train_ds = StudentVQADataset(all_pairs, splits["train_indices"], cfg_local.image_size)
    val_ds = StudentVQADataset(all_pairs, splits["val_indices"], cfg_local.image_size)

    actual = student.base_model.model if hasattr(student, "base_model") else student
    print("Model children:", [n for n, _ in actual.named_children()])
 
    train_student(
        student, tokenizer, teacher, teacher_proc,
        train_ds, val_ds, s_proj, t_proj, cfg_local, dirs,
    )
 
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--debug", action="store_true")
    main(parser.parse_args())
 