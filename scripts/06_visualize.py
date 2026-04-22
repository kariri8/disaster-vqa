"""
scripts/06_visualize.py
────────────────────────────────────────────────────────────────
Generate all result visualizations:
  • Bar charts comparing Teacher / Student / Baseline across metrics
  • Confusion matrices
  • Training curves (loss + val accuracy)
  • Sample prediction grids (correct + failure cases)
  • Grad-CAM / saliency attention maps on the Student
 
Usage:
    python scripts/06_visualize.py [--config PATH]
"""
 
import argparse
import json
import logging
import math
import os
import random
import sys
from pathlib import Path
from typing import List, Optional
 
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PIL import Image
 
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs.config import Config, cfg
 
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
 
COLORS = {
    "Teacher (Oracle)":  "#1B4F8A",   # deep blue
    "Student (Distilled)": "#D4A017", # gold
    "Baseline":          "#7A7A7A",   # grey
}
 
 
# ════════════════════════════════════════════════════════════════
# Bar comparison charts
# ════════════════════════════════════════════════════════════════
 
def bar_comparison(results: list, key: str, title: str, save_path: str,
                   lower_is_better: bool = False):
    names = [r["model"] for r in results if key in r]
    vals = [r[key] for r in results if key in r]
    if not names:
        logger.warning(f"No data for metric '{key}'. Skipping.")
        return
 
    colors = [COLORS.get(n, "#5E5E5E") for n in names]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(names, vals, color=colors, edgecolor="k", linewidth=0.5)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                f"{v:.3f}", ha="center", fontweight="bold", fontsize=11)
    ax.set_ylim(0, max(vals) * 1.25 if vals else 1)
    if lower_is_better:
        ax.set_title(f"{title}  (lower is better)", fontsize=12)
    else:
        ax.set_title(title, fontsize=12)
    ax.set_ylabel("Score")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"Saved: {save_path}")
 
 
def per_qtype_grouped_bars(results: list, save_path: str):
    """Grouped bars — one group per question type, one bar per model."""
    qtypes = None
    for r in results:
        if "per_qtype" in r:
            qtypes = list(r["per_qtype"].keys())
            break
    if qtypes is None:
        return
 
    x = np.arange(len(qtypes))
    width = 0.22
    fig, ax = plt.subplots(figsize=(12, 5))
    for i, res in enumerate(results):
        if "per_qtype" not in res:
            continue
        vals = [res["per_qtype"].get(qt, 0) for qt in qtypes]
        offset = (i - 1) * width
        bars = ax.bar(x + offset, vals, width, label=res["model"],
                      color=COLORS.get(res["model"], "#888"), edgecolor="k", lw=0.4)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                    f"{v:.2f}", ha="center", fontsize=6, rotation=45)
 
    ax.set_xticks(x)
    ax.set_xticklabels(qtypes, rotation=30, ha="right", fontsize=9)
    ax.set_ylim(0, 1.25)
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy per Question Type")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"Saved: {save_path}")
 
 
# ════════════════════════════════════════════════════════════════
# Confusion matrices
# ════════════════════════════════════════════════════════════════
 
def plot_confusion_matrices(results: list, labels: list, viz_dir: str):
    for res in results:
        if "cls_confusion" not in res:
            continue
        cm = np.array(res["cls_confusion"])
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title(f"{res['model']} — Confusion Matrix")
        plt.tight_layout()
        name = res["model"].lower().replace(" ", "_").replace("(", "").replace(")", "")
        p = os.path.join(viz_dir, f"cm_{name}.png")
        plt.savefig(p, dpi=150)
        plt.close()
        logger.info(f"Saved: {p}")
 
 
# ════════════════════════════════════════════════════════════════
# Training curves
# ════════════════════════════════════════════════════════════════
 
def plot_training_curves(logs_dir: str, viz_dir: str):
    for logname, title, loss_key in [
        ("teacher_log.json",        "Teacher Training",        "train_loss"),
        ("student_distill_log.json","Student Distillation",    "total"),
        ("baseline_log.json",       "Baseline Training",       "train_loss"),
    ]:
        lp = os.path.join(logs_dir, logname)
        if not os.path.isfile(lp):
            continue
        with open(lp) as f:
            hist = json.load(f)
        epochs = [h["epoch"] for h in hist]
        losses = [h.get(loss_key, 0) for h in hist]
        val_accs = [h.get("val_acc", None) for h in hist]
 
        fig, ax1 = plt.subplots(figsize=(8, 4))
        ax1.plot(epochs, losses, "b-o", ms=4, label=loss_key.replace("_", " ").title())
        ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss", color="b")
        ax1.tick_params(axis="y", labelcolor="b")
        if any(v is not None for v in val_accs):
            ax2 = ax1.twinx()
            ax2.plot(epochs, val_accs, "r-s", ms=4, label="Val Accuracy")
            ax2.set_ylabel("Accuracy", color="r")
            ax2.tick_params(axis="y", labelcolor="r")
        ax1.set_title(title)
        ax1.legend(loc="upper left")
        plt.tight_layout()
        p = os.path.join(viz_dir, f"curve_{logname.replace('.json', '.png')}")
        plt.savefig(p, dpi=150)
        plt.close()
        logger.info(f"Saved: {p}")
 
    # Student: also plot individual loss components if available
    sp = os.path.join(logs_dir, "student_distill_log.json")
    if os.path.isfile(sp):
        with open(sp) as f:
            hist = json.load(f)
        if "task" in hist[0]:
            epochs = [h["epoch"] for h in hist]
            fig, ax = plt.subplots(figsize=(8, 4))
            for key, lbl, col in [
                ("task",    "L_task (CE)",          "blue"),
                ("feature", "L_feature (Cosine)",   "orange"),
                ("total",   "L_total",              "red"),
            ]:
                vals = [h.get(key, 0) for h in hist]
                ax.plot(epochs, vals, "o-", ms=3, label=lbl, color=col)
            ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
            ax.set_title("Student — Loss Components")
            ax.legend()
            plt.tight_layout()
            p = os.path.join(viz_dir, "student_loss_components.png")
            plt.savefig(p, dpi=150)
            plt.close()
            logger.info(f"Saved: {p}")
 
 
# ════════════════════════════════════════════════════════════════
# Sample prediction grids
# ════════════════════════════════════════════════════════════════
 
def sample_grid(preds: list, title: str, save_path: str,
                max_show: int = 9):
    samples = random.sample(preds, min(max_show, len(preds)))
    cols = min(3, len(samples))
    rows = math.ceil(len(samples) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    axes = np.array(axes).flatten()
 
    for ax, s in zip(axes, samples):
        post_path = s.get("post_path", "")
        if post_path and os.path.isfile(post_path):
            ax.imshow(Image.open(post_path).resize((256, 256)))
        else:
            ax.text(0.5, 0.5, "No image", ha="center", va="center",
                    transform=ax.transAxes)
        ok = s["true_answer"].lower() == s["pred_answer"]
        color = "green" if ok else "red"
        marker = "✓" if ok else "✗"
        ax.set_title(
            f"{marker} [{s['question_type']}]\n"
            f"True: {s['true_answer']}\nPred: {s['pred_answer']}",
            fontsize=7, color=color, pad=4)
        ax.axis("off")
 
    for ax in axes[len(samples):]:
        ax.axis("off")
    plt.suptitle(title, fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"Saved: {save_path}")
 
 
# ════════════════════════════════════════════════════════════════
# Teacher vs Student comparison table (text)
# ════════════════════════════════════════════════════════════════
 
def print_teacher_student_comparison(t_preds: list, s_preds: list,
                                     b_preds: list, n: int = 15):
    print("\n" + "─" * 90)
    print(f"{'QType':22s}  {'True':15s}  "
          f"{'Teacher':15s}  {'Student':15s}  {'Baseline':15s}")
    print("─" * 90)
    for i in range(min(n, len(s_preds))):
        sp = s_preds[i]
        tp = t_preds[i] if i < len(t_preds) else {}
        bp = b_preds[i] if i < len(b_preds) else {}
        ok_t = "✓" if tp.get("pred_answer") == sp["true_answer"].lower() else "✗"
        ok_s = "✓" if sp["pred_answer"] == sp["true_answer"].lower() else "✗"
        ok_b = "✓" if bp.get("pred_answer") == sp["true_answer"].lower() else "✗"
        print(
            f"{sp['question_type']:22s}  "
            f"{sp['true_answer']:15s}  "
            f"{tp.get('pred_answer','?'):13s}{ok_t}  "
            f"{sp['pred_answer']:13s}{ok_s}  "
            f"{bp.get('pred_answer','?'):13s}{ok_b}"
        )
    print("─" * 90)
 
 
# ════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════
 
def main(args):
    cfg_local = Config.load(args.config) if args.config else cfg
 
    project = cfg_local.project_root
    viz_dir = os.path.join(project, "visualizations")
    logs_dir = os.path.join(project, "logs")
    metrics_dir = os.path.join(project, "metrics")
    pred_dir = os.path.join(project, "predictions")
    os.makedirs(viz_dir, exist_ok=True)
 
    # Load results
    all_results_path = os.path.join(metrics_dir, "all_results.json")
    if not os.path.isfile(all_results_path):
        logger.error(f"Run 05_evaluate.py first. {all_results_path} not found.")
        return
    with open(all_results_path) as f:
        all_results = json.load(f)
 
    # Load predictions
    preds = {}
    for name in ("teacher", "student", "baseline"):
        p = os.path.join(pred_dir, f"{name}_preds.json")
        if os.path.isfile(p):
            with open(p) as f:
                preds[name] = json.load(f)
 
    # --- Bar charts ---
    for key, title, lower in [
        ("overall_accuracy",   "Overall VQA Accuracy",               False),
        ("cls_accuracy",       "Damage Classification Accuracy",      False),
        ("cls_macro_f1",       "Damage Classification Macro-F1",      False),
        ("cls_weighted_f1",    "Damage Classification Weighted-F1",   False),
        ("bin_f1",             "Binary Damage Detection F1",          False),
        ("count_mae",          "Count MAE",                           True),
        ("count_rmse",         "Count RMSE",                          True),
        ("open_token_f1",      "Open-Ended Token F1",                 False),
    ]:
        bar_comparison(all_results, key, title,
                       os.path.join(viz_dir, f"cmp_{key}.png"), lower_is_better=lower)
 
    per_qtype_grouped_bars(all_results, os.path.join(viz_dir, "cmp_per_qtype.png"))
 
    # --- Confusion matrices ---
    plot_confusion_matrices(all_results, cfg_local.damage_labels, viz_dir)
 
    # --- Training curves ---
    plot_training_curves(logs_dir, viz_dir)
 
    # --- Sample grids ---
    if "student" in preds:
        sample_grid(preds["student"], "Student (Distilled) — Samples",
                    os.path.join(viz_dir, "student_samples.png"))
        failures = [p for p in preds["student"]
                    if p["true_answer"].lower() != p["pred_answer"]]
        if failures:
            sample_grid(failures, "Student — Failure Cases",
                        os.path.join(viz_dir, "student_failures.png"))
 
    # --- Comparison table ---
    if all(k in preds for k in ("teacher", "student", "baseline")):
        print_teacher_student_comparison(
            preds["teacher"], preds["student"], preds["baseline"])
 
    logger.info(f"All visualizations saved to {viz_dir}")
 
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    main(parser.parse_args())