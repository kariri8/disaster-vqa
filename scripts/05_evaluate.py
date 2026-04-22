"""
scripts/05_evaluate.py
────────────────────────────────────────────────────────────────
Evaluate all three models (Teacher, Student, Baseline) on the
held-out test set. Computes:
  • Exact-match accuracy per question type (classification, count,
    binary, open-ended)
  • Macro-F1 and Weighted-F1 for damage classification
  • Per-class recall and confusion matrix
  • Binary precision/recall/F1
  • Count MAE and RMSE
  • Distillation efficiency: Gap Closed (%)
 
Results are saved as JSON and printed as a formatted table.
 
Usage:
    python scripts/05_evaluate.py [--config PATH]
        [--teacher_ckpt PATH]  [--student_ckpt PATH]  [--baseline_ckpt PATH]
"""
 
import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional
 
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score,
    mean_absolute_error, mean_squared_error,
    precision_score, recall_score,
)
from tqdm.auto import tqdm
 
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs.config import Config, cfg
from scripts.datasets import StudentVQADataset, load_pairs_and_splits
from scripts.models import DEVICE, DTYPE, load_student, load_teacher
from scripts.utils import parse_answer
 
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
 
 
# ════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════
 
def load_teacher_for_eval(cfg_local, ckpt_path: Optional[str]):
    model, processor = load_teacher(cfg_local)
    lora_dir = ckpt_path or os.path.join(
        cfg_local.project_root, "checkpoints/teacher/best_lora")
    if os.path.isdir(lora_dir):
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, lora_dir)
        logger.info(f"Teacher LoRA loaded from {lora_dir}")
    return model, processor
 
 
def load_student_for_eval(cfg_local, ckpt_path: Optional[str], apply_qlora: bool = True):
    model, tok = load_student(cfg_local, apply_qlora=apply_qlora)
    pt = ckpt_path or os.path.join(
        cfg_local.project_root,
        "checkpoints/student/best.pt" if apply_qlora else "checkpoints/baseline/best.pt",
    )
    if os.path.isfile(pt):
        state = torch.load(pt, map_location=DEVICE)
        # Only load model weights (skip optimizer, projector keys)
        model.load_state_dict(state["model"], strict=False)
        logger.info(f"Loaded weights from {pt}")
    return model, tok
 
 
def build_teacher_image(item: dict, cfg_local: Config) -> Optional[Image.Image]:
    if not item.get("pre_ok") or not os.path.isfile(item.get("pre_path", "")):
        return None
    sz = cfg_local.image_size
    pre = Image.open(item["pre_path"]).convert("RGB").resize((sz, sz))
    post = Image.open(item["post_path"]).convert("RGB").resize((sz, sz))
    canvas = Image.new("RGB", (sz * 2, sz))
    canvas.paste(pre, (0, 0))
    canvas.paste(post, (sz, 0))
    return canvas
 
 
# ════════════════════════════════════════════════════════════════
# Per-model evaluation
# ════════════════════════════════════════════════════════════════
 
def evaluate_model(model, tokenizer_or_proc, test_ds: StudentVQADataset,
                   model_name: str, is_teacher: bool, cfg_local: Config) -> tuple:
    logger.info(f"\n{'='*60}")
    logger.info(f"  Evaluating: {model_name}")
    logger.info(f"{'='*60}")
 
    model.eval()
    predictions = []
 
    batch_size = 64
    for start in tqdm(range(0, len(test_ds), batch_size), desc=f"Eval {model_name}"):
        batch = [test_ds[i] for i in range(start, min(start + batch_size, len(test_ds)))]
        
        prompts = [f"USER: <image>\n{item['question']}\nASSISTANT:" for item in batch]
        images = [item["image"] for item in batch]
        
        inf_inputs = tokenizer_or_proc(
            text=prompts,
            images=images,
            return_tensors="pt",
            padding=True,
        ).to(DEVICE)

        tokenizer_or_proc.tokenizer.padding_side = "left"
        
        with torch.no_grad():
            output_ids = model.generate(
                **inf_inputs,
                max_new_tokens=cfg_local.max_answer_tokens,
                do_sample=False,
            )
        
        inp_len = inf_inputs["input_ids"].shape[1]
        for j, item in enumerate(batch):
            raw = tokenizer_or_proc.decode(
                output_ids[j][inp_len:], skip_special_tokens=True)
            pred = parse_answer(raw)
            predictions.append({
                "scene_id": item["scene_id"],
                "question": item["question"],
                "question_type": item["question_type"],
                "answer_type": item["answer_type"],
                "true_answer": item["answer"],
                "pred_answer": pred,
                "pred_raw": raw,
            })
 
    return _compute_metrics(predictions, model_name, cfg_local), predictions
 
 
def _compute_metrics(predictions: list, model_name: str, cfg_local: Config) -> dict:
    df = pd.DataFrame(predictions)
    res = {"model": model_name}
 
    # Overall exact match (skip open-ended for exact match)
    structured = df[df["answer_type"] != "open"]
    df["correct"] = df["true_answer"].str.lower() == df["pred_answer"]
    res["overall_accuracy"] = structured.assign(
        correct=structured["true_answer"].str.lower() == structured["pred_answer"]
    )["correct"].mean()
 
    # Per-question-type accuracy
    res["per_qtype"] = (
        df.groupby("question_type")
        .apply(lambda g: (g["true_answer"].str.lower() == g["pred_answer"]).mean())
        .to_dict()
    )
 
    # Classification
    cls_df = df[df["answer_type"] == "classification"]
    if len(cls_df):
        labels = [l.lower() for l in cfg_local.damage_labels]
        def to_idx(v):
            try: return labels.index(v)
            except ValueError: return -1
        yt = cls_df["true_answer"].str.lower().apply(to_idx)
        yp = cls_df["pred_answer"].apply(to_idx)
        mask = (yt >= 0) & (yp >= 0)
        if mask.sum():
            yt_v, yp_v = yt[mask].values, yp[mask].values
            res["cls_accuracy"] = float(accuracy_score(yt_v, yp_v))
            res["cls_macro_f1"] = float(f1_score(yt_v, yp_v, average="macro", zero_division=0))
            res["cls_weighted_f1"] = float(f1_score(yt_v, yp_v, average="weighted", zero_division=0))
            res["cls_recall_per_class"] = recall_score(
                yt_v, yp_v, average=None, zero_division=0, labels=[0,1,2,3]).tolist()
            res["cls_confusion"] = confusion_matrix(yt_v, yp_v, labels=[0,1,2,3]).tolist()
 
    # Binary
    bin_df = df[df["answer_type"] == "binary"]
    if len(bin_df):
        yt = (bin_df["true_answer"].str.lower() == "yes").astype(int)
        yp = (bin_df["pred_answer"] == "yes").astype(int)
        res["bin_accuracy"] = float(accuracy_score(yt, yp))
        res["bin_precision"] = float(precision_score(yt, yp, zero_division=0))
        res["bin_recall"] = float(recall_score(yt, yp, zero_division=0))
        res["bin_f1"] = float(f1_score(yt, yp, zero_division=0))
 
    # Counting
    cnt_df = df[df["answer_type"] == "count"]
    if len(cnt_df):
        def safe_int(x):
            try: return int(x)
            except: return -1
        yt = cnt_df["true_answer"].apply(safe_int)
        yp = cnt_df["pred_answer"].apply(safe_int)
        mask = (yt >= 0) & (yp >= 0)
        if mask.sum():
            yt_v, yp_v = yt[mask].values, yp[mask].values
            res["count_accuracy"] = float(accuracy_score(yt_v, yp_v))
            res["count_mae"] = float(mean_absolute_error(yt_v, yp_v))
            res["count_rmse"] = float(np.sqrt(mean_squared_error(yt_v, yp_v)))
 
    # Open-ended: BLEU-1 approximation (token overlap)
    open_df = df[df["answer_type"] == "open"]
    if len(open_df):
        def token_f1(true, pred):
            t_tok = set(true.lower().split())
            p_tok = set(pred.lower().split())
            if not p_tok: return 0.0
            prec = len(t_tok & p_tok) / len(p_tok)
            rec = len(t_tok & p_tok) / max(len(t_tok), 1)
            if prec + rec == 0: return 0.0
            return 2 * prec * rec / (prec + rec)
        res["open_token_f1"] = float(
            open_df.apply(lambda r: token_f1(r["true_answer"], r["pred_answer"]), axis=1).mean()
        )
 
    # Print summary
    for k, v in res.items():
        if k in ("model", "cls_confusion", "per_qtype"): continue
        if isinstance(v, float): print(f"  {k:30s} {v:.4f}")
        elif isinstance(v, list): print(f"  {k:30s} {[f'{x:.3f}' for x in v]}")
 
    return res
 
 
# ════════════════════════════════════════════════════════════════
# Distillation efficiency report
# ════════════════════════════════════════════════════════════════
 
def report_efficiency(teacher_res, student_res, baseline_res, dirs):
    t = teacher_res.get("overall_accuracy", 0)
    s = student_res.get("overall_accuracy", 0)
    b = baseline_res.get("overall_accuracy", 0)
    gap = t - b
    eff = (s - b) / gap * 100 if gap > 0 else 0.0
 
    print("\n" + "=" * 60)
    print("  DISTILLATION EFFICIENCY")
    print("=" * 60)
    print(f"  Teacher (Oracle)       : {t:.4f}")
    print(f"  Student (Distilled)    : {s:.4f}")
    print(f"  Baseline (No Distill)  : {b:.4f}")
    print(f"  Gap closed             : {eff:.1f}%")
 
    report = {"teacher_acc": t, "student_acc": s, "baseline_acc": b,
              "gap_closed_pct": eff}
    with open(os.path.join(dirs["metrics"], "distillation_efficiency.json"), "w") as f:
        json.dump(report, f, indent=2)
 
 
# ════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════
 
def main(args):
    cfg_local = Config.load(args.config) if args.config else cfg
 
    meta_dir = os.path.join(cfg_local.project_root, "metadata")
    dirs = {
        "predictions": os.path.join(cfg_local.project_root, "predictions"),
        "metrics": os.path.join(cfg_local.project_root, "metrics"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
 
    all_pairs, splits = load_pairs_and_splits(meta_dir)
    test_ds = StudentVQADataset(all_pairs, splits["test_indices"], cfg_local.image_size)
    logger.info(f"Test set: {len(test_ds)} samples")
 
    all_results = []
    all_preds = {}
 
    # Teacher
    teacher, teacher_proc = load_teacher_for_eval(cfg_local, args.teacher_ckpt)
    t_res, t_preds = evaluate_model(
        teacher, teacher_proc, test_ds, "Teacher (Oracle)", True, cfg_local)
    all_results.append(t_res)
    all_preds["teacher"] = t_preds
    del teacher; torch.cuda.empty_cache()
 
    # Distilled student
    student, s_tok = load_student_for_eval(cfg_local, args.student_ckpt, apply_qlora=True)
    s_res, s_preds = evaluate_model(
        student, s_tok, test_ds, "Student (Distilled)", False, cfg_local)
    all_results.append(s_res)
    all_preds["student"] = s_preds
    del student; torch.cuda.empty_cache()
 
    # Baseline
    baseline, b_tok = load_student_for_eval(cfg_local, args.baseline_ckpt, apply_qlora=True)
    b_res, b_preds = evaluate_model(
        baseline, b_tok, test_ds, "Baseline", False, cfg_local)
    all_results.append(b_res)
    all_preds["baseline"] = b_preds
 
    # Save
    with open(os.path.join(dirs["metrics"], "all_results.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    for name, preds in all_preds.items():
        with open(os.path.join(dirs["predictions"], f"{name}_preds.json"), "w") as f:
            json.dump(preds, f, indent=2)
 
    report_efficiency(t_res, s_res, b_res, dirs)
    logger.info(f"All results saved to {dirs['metrics']}")
 
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--teacher_ckpt", type=str, default=None)
    parser.add_argument("--student_ckpt", type=str, default=None)
    parser.add_argument("--baseline_ckpt", type=str, default=None)
    main(parser.parse_args())
 