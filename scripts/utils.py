"""
scripts/utils.py
────────────────────────────────────────────────────────────────
Shared training utilities: EarlyStopping, MetricLogger,
checkpoint saving, answer parsing.
"""

import json
import os
from datetime import datetime
from typing import Optional

import torch


class EarlyStopping:
    def __init__(self, patience: int = 5, mode: str = "min"):
        """mode='min' for loss, 'max' for accuracy."""
        self.patience = patience
        self.mode = mode
        self.counter = 0
        self.best = None
        self.stop = False

    def __call__(self, score: float) -> bool:
        if self.best is None:
            self.best = score
            return False
        better = (score < self.best) if self.mode == "min" else (score > self.best)
        if better:
            self.best = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
        return self.stop


class MetricLogger:
    def __init__(self, path: str):
        self.path = path
        self.data = []

    def log(self, entry: dict):
        self.data.append({**entry, "ts": datetime.now().isoformat()})
        with open(self.path, "w") as f:
            json.dump(self.data, f, indent=2)


def save_ckpt(model, optimizer, epoch: int, metrics: dict,
              path: str, projector=None):
    """Save a standard checkpoint dict."""
    state = {
        "epoch": epoch,
        "metrics": metrics,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    if projector is not None:
        state["projector"] = projector.state_dict()
    torch.save(state, path)
    print(f"    ckpt → {os.path.basename(path)}")


def save_lora_checkpoint(model, ckpt_dir: str, tag: str = "best"):
    """Save a PEFT LoRA adapter using save_pretrained."""
    out_dir = os.path.join(ckpt_dir, f"{tag}_lora")
    model.save_pretrained(out_dir)
    print(f"    LoRA ckpt → {out_dir}")


def parse_answer(text: str) -> str:
    """
    Normalise a raw model output to a canonical answer string.
    Strips common preamble phrases and takes only the first clause.
    """
    text = text.strip().lower()
    for prefix in ("answer:", "the answer is", "a:", "assistant:"): 
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
    # Take only the first sentence / comma-clause
    text = text.split(".")[0].split(",")[0].split("\n")[0].strip()
    return text