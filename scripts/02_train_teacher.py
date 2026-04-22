"""
scripts/02_train_teacher.py
────────────────────────────────────────────────────────────────
Phase 1: Fine-tune LLaVA-1.5-7B (Oracle Teacher) on bi-temporal
disaster VQA using LoRA.
 
Usage:
    python scripts/02_train_teacher.py [--config PATH] [--debug]
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
 
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
 
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs.config import Config, cfg
from scripts.datasets import TeacherVQADataset, load_pairs_and_splits
from scripts.models import (
    DEVICE, DTYPE,
    build_optimizer_scheduler,
    load_teacher,
)
from scripts.utils import (
    EarlyStopping, MetricLogger, parse_answer, save_lora_checkpoint,
)
 
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
 
 
# ════════════════════════════════════════════════════════════════
# Training loop
# ════════════════════════════════════════════════════════════════
 
def mask_prompt_labels(input_ids, labels, prompts, processor):
    """Set label = -100 for all tokens before ASSISTANT: in each sample."""
    assistant_toks = processor.tokenizer.encode("ASSISTANT:", add_special_tokens=False)
    for i, prompt in enumerate(prompts):
        ids = input_ids[i].tolist()
        ans_start = None
        for pos in range(len(ids) - len(assistant_toks)):
            if ids[pos:pos + len(assistant_toks)] == assistant_toks:
                ans_start = pos + len(assistant_toks)
                break
        if ans_start is not None:
            labels[i, :ans_start] = -100
    return labels
 
 
def train_teacher(model, processor, train_ds, val_ds, cfg: Config, dirs: dict):
    logger.info("=" * 65)
    logger.info("  PHASE 1 — TEACHER TRAINING  (LLaVA-1.5-7B + LoRA)")
    logger.info("=" * 65)
 
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer, scheduler = build_optimizer_scheduler(
        params, len(train_ds), cfg.teacher_batch_size,
        cfg.teacher_grad_accum, cfg.teacher_epochs, cfg,
    )
    # Override lr with teacher-specific lr
    for pg in optimizer.param_groups:
        pg["lr"] = cfg.teacher_lr
 
    scaler = torch.amp.GradScaler("cuda", enabled=(DTYPE == torch.float16))
    early = EarlyStopping(cfg.patience, mode="max") # We want maximum accuracy
    mlog = MetricLogger(os.path.join(dirs["logs"], "teacher_log.json"))
    best_val_acc = 0.0
 
    for epoch in range(cfg.teacher_epochs):
        model.train()
        indices = list(range(len(train_ds)))
        random.shuffle(indices)
        epoch_loss, n_steps = 0.0, 0
        optimizer.zero_grad()
 
        pbar = tqdm(range(0, len(indices), cfg.teacher_batch_size),
                    desc=f"Teacher Ep {epoch+1}/{cfg.teacher_epochs}")
 
        for start in pbar:
            batch_idx = indices[start:start + cfg.teacher_batch_size]
            batch = [train_ds[i] for i in batch_idx]
 
            images = [b["image"] for b in batch]
            prompts = [b["prompt"] for b in batch]
 
            inputs = processor(
                text=prompts, images=images,
                return_tensors="pt", padding=True,
                truncation=True, max_length=2048,
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
 
            labels = inputs["input_ids"].clone()
            labels = mask_prompt_labels(inputs["input_ids"], labels, prompts, processor)
            inputs["labels"] = labels
 
            with torch.amp.autocast("cuda", dtype=DTYPE):
                outputs = model(**inputs)
                loss = outputs.loss / cfg.teacher_grad_accum
 
            scaler.scale(loss).backward()
            n_steps += 1
 
            if n_steps % cfg.teacher_grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
 
            epoch_loss += loss.item() * cfg.teacher_grad_accum
            pbar.set_postfix(loss=f"{loss.item() * cfg.teacher_grad_accum:.4f}")
 
        avg_train = epoch_loss / max(n_steps, 1)
 
        # Validation (generation accuracy on a capped subset)
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            val_limit = min(len(val_ds), 500)
            for vi in tqdm(range(0, val_limit, cfg.teacher_batch_size),
                           desc="Teacher Val", leave=False):
                batch = [val_ds[j]
                         for j in range(vi, min(vi + cfg.teacher_batch_size, val_limit))]
                images = [b["image"] for b in batch]
                questions = [b["question"] for b in batch]
                gts = [b["answer"] for b in batch]
 
                gen_prompts = [
                    f"USER: <image>\nThis image shows the same area before (left) "
                    f"and after (right) a disaster.\n{q}\nASSISTANT:"
                    for q in questions
                ]
                inputs = processor(
                    text=gen_prompts, images=images,
                    return_tensors="pt", padding=True, truncation=True, max_length=2048,
                )
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                gen_ids = model.generate(**inputs,
                                         max_new_tokens=cfg.max_answer_tokens,
                                         do_sample=False)
                inp_len = inputs["input_ids"].shape[1]
                for gen, gt in zip(gen_ids, gts):
                    decoded = processor.tokenizer.decode(
                        gen[inp_len:], skip_special_tokens=True)
                    if parse_answer(decoded) == gt.lower():
                        correct += 1
                    total += 1
 
        val_acc = correct / max(total, 1)
        metrics = {"epoch": epoch + 1, "train_loss": avg_train, "val_acc": val_acc}
        mlog.log(metrics)
        logger.info(f"  Ep {epoch+1}  train_loss={avg_train:.4f}  val_acc={val_acc:.4f}")
 
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_lora_checkpoint(model, dirs["teacher_ckpt"], tag="best")
            logger.info(f"    ★ Best teacher saved (val_acc={best_val_acc:.4f})")
            
        if early(val_acc):
            logger.info(f"  Early stop at epoch {epoch+1} because val_acc hasn't improved.")
            break
 
    logger.info("Teacher training complete.")
    return model
 
 
# ════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════
 
def main(args):
    cfg_local = Config.load(args.config) if args.config else cfg
    if args.debug:
        cfg_local.debug_mode = True
 
    meta_dir = os.path.join(cfg_local.project_root, "metadata")
    dirs = {
        "teacher_ckpt": os.path.join(cfg_local.project_root, "checkpoints/teacher"),
        "logs": os.path.join(cfg_local.project_root, "logs"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
 
    all_pairs, splits = load_pairs_and_splits(meta_dir)
    logger.info(f"Loaded {len(all_pairs)} pairs from {meta_dir}")
 
    model, processor = load_teacher(cfg_local)
 
    train_ds = TeacherVQADataset(
        all_pairs, splits["train_indices"], processor, cfg_local.image_size)
    val_ds = TeacherVQADataset(
        all_pairs, splits["val_indices"], processor, cfg_local.image_size)
 
    train_teacher(model, processor, train_ds, val_ds, cfg_local, dirs)
 
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--debug", action="store_true")
    main(parser.parse_args())