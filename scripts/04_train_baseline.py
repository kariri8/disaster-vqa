"""
scripts/04_train_baseline.py
────────────────────────────────────────────────────────────────
Baseline: Fine-tune student model with only the task CE loss — no
teacher signal. Used to measure the gap closed by distillation.

Gap closed (%) = (Student_acc - Baseline_acc) / (Teacher_acc - Baseline_acc)
"""

import argparse
import logging
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs.config import Config, cfg
from scripts.datasets import StudentVQADataset, load_pairs_and_splits
from scripts.models import DEVICE, DTYPE, build_optimizer_scheduler, load_student
from scripts.utils import EarlyStopping, MetricLogger, parse_answer, save_ckpt
# Reuse student_forward directly — same prompt format, same pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

def student_forward(model, processor, image, question, answer):
    prompt = f"USER: <image>\n{question}\nASSISTANT: {answer}"
    inputs = processor(
        text=[prompt],
        images=[image],
        return_tensors="pt",
    ).to(model.device)

    actual_model = model.base_model.model if hasattr(model, "base_model") else model

    if hasattr(actual_model, "model") and hasattr(actual_model.model, "vision_tower"):
        vision_tower = actual_model.model.vision_tower
    else:
        vision_tower = actual_model.vision_tower

    with torch.no_grad():
        vision_out = vision_tower(
            inputs["pixel_values"].to(dtype=DTYPE),
            output_hidden_states=True
        )
        patch_tokens = vision_out.last_hidden_state[:, 1:, :]
        vision_pool = patch_tokens.mean(dim=1)

    outputs = model(**inputs)
    shift_logits = outputs.logits[..., :-1, :].contiguous()
    ans_tokens = inputs["input_ids"][..., 1:].contiguous()

    return shift_logits, vision_pool, ans_tokens


def train_baseline(model, tokenizer, train_ds, val_ds,
                   cfg: Config, dirs: dict):
    logger.info("=" * 65)
    logger.info("  BASELINE TRAINING  (no distillation)")
    logger.info("=" * 65)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer, scheduler = build_optimizer_scheduler(
        params, len(train_ds), cfg.student_batch_size,
        cfg.student_grad_accum, cfg.baseline_epochs, cfg,
    )
    for pg in optimizer.param_groups:
        pg["lr"] = cfg.baseline_lr

    scaler = torch.amp.GradScaler("cuda", enabled=(DTYPE == torch.float16))
    early = EarlyStopping(cfg.patience, mode="max")
    mlog = MetricLogger(os.path.join(dirs["logs"], "baseline_log.json"))
    best_acc = 0.0
    best_path = os.path.join(dirs["baseline_ckpt"], "best.pt")

    for epoch in range(cfg.baseline_epochs):
        model.train()
        indices = list(range(len(train_ds)))
        random.shuffle(indices)
        epoch_loss, n_steps = 0.0, 0
        optimizer.zero_grad()

        pbar = tqdm(range(0, len(indices), cfg.student_batch_size),
                    desc=f"Baseline Ep {epoch+1}/{cfg.baseline_epochs}")

        for start in pbar:
            batch_idx = indices[start:start + cfg.student_batch_size]
            batch = [train_ds[i] for i in batch_idx]
            batch_loss = torch.tensor(0.0, device=DEVICE, requires_grad=True)

            for item in batch:
                with torch.amp.autocast("cuda", dtype=DTYPE):
                    shift_logits, _, ans_tokens = student_forward(
                        model, tokenizer,
                        item["image"], item["question"], item["answer"],
                    )
                    loss = F.cross_entropy(
                        shift_logits.reshape(-1, shift_logits.size(-1)),
                        ans_tokens.reshape(-1),
                    )
                batch_loss = batch_loss + loss / (len(batch) * cfg.student_grad_accum)
                epoch_loss += loss.item()

            scaler.scale(batch_loss).backward()
            n_steps += 1

            if n_steps % cfg.student_grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            pbar.set_postfix(loss=f"{batch_loss.item():.4f}")

        avg_loss = epoch_loss / max(n_steps * cfg.student_batch_size, 1)

        # Validation — identical to student validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for vi in range(min(len(val_ds), 200)):
                item = val_ds[vi]
                inf_prompt = f"USER: <image>\n{item['question']}\nASSISTANT:"
                inf_inputs = tokenizer(
                    text=inf_prompt, images=item["image"], return_tensors="pt"
                ).to(DEVICE)
                try:
                    output_ids = model.generate(
                        **inf_inputs,
                        max_new_tokens=cfg.max_answer_tokens,
                        do_sample=False,
                    )
                    pred_text = tokenizer.decode(
                        output_ids[0][inf_inputs["input_ids"].shape[1]:],
                        skip_special_tokens=True,
                    )
                    if parse_answer(pred_text) == item["answer"].lower():
                        correct += 1
                except Exception as e:
                    logger.debug(f"Val error: {e}")
                total += 1

        val_acc = correct / max(total, 1)
        metrics = {"epoch": epoch + 1, "train_loss": avg_loss, "val_acc": val_acc}
        mlog.log(metrics)
        logger.info(f"  Ep {epoch+1}  loss={avg_loss:.4f}  val_acc={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            save_ckpt(model, optimizer, epoch + 1, metrics, best_path)
            logger.info(f"    ★ Best baseline (acc={best_acc:.4f})")

        if early(val_acc):
            logger.info(f"  Early stop at epoch {epoch+1}")
            break

    logger.info("Baseline training complete.")
    return model, tokenizer


def main(args):
    cfg_local = Config.load(args.config) if args.config else cfg
    if args.debug:
        cfg_local.debug_mode = True

    meta_dir = os.path.join(cfg_local.project_root, "metadata")
    dirs = {
        "baseline_ckpt": os.path.join(cfg_local.project_root, "checkpoints/baseline"),
        "logs": os.path.join(cfg_local.project_root, "logs"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    all_pairs, splits = load_pairs_and_splits(meta_dir)
    model, tokenizer = load_student(cfg_local, apply_qlora=not args.full_ft)

    train_ds = StudentVQADataset(all_pairs, splits["train_indices"], cfg_local.image_size)
    val_ds = StudentVQADataset(all_pairs, splits["val_indices"], cfg_local.image_size)

    train_baseline(model, tokenizer, train_ds, val_ds, cfg_local, dirs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--full_ft", action="store_true",
                        help="Full fine-tuning instead of QLoRA")
    main(parser.parse_args())