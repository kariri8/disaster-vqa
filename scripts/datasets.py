"""
scripts/datasets.py
────────────────────────────────────────────────────────────────
PyTorch Dataset classes for Teacher (bi-temporal) and Student
(post-only) VQA. Loaded from the JSON produced by 01_prepare_dataset.py.
"""

import json
import os
from typing import List, Optional

from PIL import Image
from torch.utils.data import Dataset

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs.config import Config


def load_pairs_and_splits(metadata_dir: str):
    """Load all_pairs.json and splits.json from the metadata directory."""
    with open(os.path.join(metadata_dir, "all_pairs.json")) as f:
        all_pairs = json.load(f)
    with open(os.path.join(metadata_dir, "splits.json")) as f:
        splits = json.load(f)
    return all_pairs, splits


# ════════════════════════════════════════════════════════════════
# Teacher dataset — bi-temporal (pre + post side-by-side)
# ════════════════════════════════════════════════════════════════

class TeacherVQADataset(Dataset):
    """
    Provides side-by-side [pre | post] concatenated images for LLaVA.
    Only includes samples where BOTH pre and post images exist.
    Builds the full LLaVA prompt internally.
    """

    def __init__(self, all_pairs: List[dict], indices: List[int],
                 processor, image_size: int = 336,
                 question_types: Optional[List[str]] = None):
        # Filter: need both images
        self.samples = [
            all_pairs[i] for i in indices
            if all_pairs[i]["pre_ok"] and all_pairs[i]["post_ok"]
        ]
        # Optionally restrict to specific question types
        if question_types is not None:
            self.samples = [s for s in self.samples
                            if s["question_type"] in question_types]
        self.processor = processor
        self.image_size = image_size
        print(f"  TeacherDataset: {len(self.samples)} samples "
              f"(from {len(indices)} indices, needs both images)")

    def __len__(self):
        return len(self.samples)

    def _concat_images(self, pre_path: str, post_path: str) -> Image.Image:
        size = self.image_size
        pre = Image.open(pre_path).convert("RGB").resize((size, size))
        post = Image.open(post_path).convert("RGB").resize((size, size))
        canvas = Image.new("RGB", (size * 2, size))
        canvas.paste(pre, (0, 0))
        canvas.paste(post, (size, 0))
        return canvas

    def _build_prompt(self, question: str, answer: str) -> str:
        return (
            f"USER: <image>\n"
            f"This image shows the same area before (left) and after (right) a disaster.\n"
            f"{question}\n"
            f"ASSISTANT: {answer}"
        )

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        image = self._concat_images(s["pre_path"], s["post_path"])
        prompt = self._build_prompt(s["question"], s["answer"])
        return {
            "image": image,
            "prompt": prompt,
            "question": s["question"],
            "question_type": s["question_type"],
            "answer": s["answer"],
            "answer_idx": s["answer_idx"],
            "answer_type": s["answer_type"],
            "scene_id": s["scene_id"],
            "pre_path": s["pre_path"],
            "post_path": s["post_path"],
        }


# ════════════════════════════════════════════════════════════════
# Student dataset — post-disaster image only
# ════════════════════════════════════════════════════════════════

class StudentVQADataset(Dataset):
    """
    Post-disaster image only. Used for both the distilled Student
    and the Baseline (no distillation) models.
    """

    def __init__(self, all_pairs: List[dict], indices: List[int],
                 image_size: int = 336,
                 question_types: Optional[List[str]] = None):
        self.samples = [
            all_pairs[i] for i in indices
            if all_pairs[i]["post_ok"]
        ]
        if question_types is not None:
            self.samples = [s for s in self.samples
                            if s["question_type"] in question_types]
        self.image_size = image_size
        print(f"  StudentDataset: {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        image = Image.open(s["post_path"]).convert("RGB").resize(
            (self.image_size, self.image_size))
        return {
            "image": image,
            "question": s["question"],
            "question_type": s["question_type"],
            "answer": s["answer"],
            "answer_idx": s["answer_idx"],
            "answer_type": s["answer_type"],
            "scene_id": s["scene_id"],
            "post_path": s["post_path"],
            "pre_path": s.get("pre_path", ""),
            "pre_ok": s.get("pre_ok", False),
        }