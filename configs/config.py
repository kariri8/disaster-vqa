"""
configs/config.py
Central configuration for Inferential Disaster VQA via Feature-Level Distillation.
"""
 
from dataclasses import dataclass, field, asdict
import json, os
 
 
@dataclass
class Config:
    # ── Paths ─────────────────────────────────────────────────
    project_root: str = "/home/dsrc_iskakova/arc/project"
    xbd_root: str = "/home/dsrc_iskakova/arc/project/data/tier3"        # set after download
    xbd_structure: str = "flat"
 
    # ── Experiment ────────────────────────────────────────────
    experiment_name: str = "disaster_vqa_v2"
    seed: int = 42
    debug_mode: bool = False
    debug_scenes: int = 200
 
    # ── Image ─────────────────────────────────────────────────
    image_size: int = 336
 
    # ── Dataset / VQA ─────────────────────────────────────────
    max_count_bin: int = 16
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
 
    # ── Teacher: LLaVA-1.5-7B ─────────────────────────────────
    teacher_model_id: str = "llava-hf/llava-1.5-7b-hf"
    teacher_lora_r: int = 16
    teacher_lora_alpha: int = 32
    teacher_lora_dropout: float = 0.05
    teacher_lora_target_modules: list = field(
        default_factory=lambda: [
            "q_proj", "v_proj", "k_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
    )
    teacher_epochs: int = 10
    teacher_lr: float = 1e-3
    teacher_batch_size: int = 8
    teacher_grad_accum: int = 2
 
    # ── Student / Baseline: Moondream2 ────────────────────────
    student_model_id: str = "llava-hf/llava-interleave-qwen-0.5b-hf"
    student_model_revision: str = "2025-01-09"
    student_epochs: int = 15
    student_lr: float = 3e-4
    student_batch_size: int = 16
    student_grad_accum: int = 1
 
    # Student QLoRA settings
    student_lora_r: int = 16
    student_lora_alpha: int = 32
    student_lora_dropout: float = 0.05
    # Moondream2 uses Phi-1.5 as LM; these are common attention projections
    # student_lora_target_modules: list = field(
    #     default_factory=lambda: ["q_proj", "v_proj", "k_proj", "out_proj"]
    # )

    student_lora_target_modules: list = field(
        default_factory=lambda: ["qkv", "dense"]
    )
 
    baseline_epochs: int = 15
    baseline_lr: float = 3e-4
    baseline_batch_size: int = 16
 
    # ── Distillation loss weights ──────────────────────────────
    # L_total = alpha * L_task + beta * L_feature + gamma * L_KD
    alpha: float = 1.0   # task (CE on answer)
    beta: float = 0.5    # feature MSE (vision-encoder projection)
    gamma: float = 0.0   # KL divergence on logits
    temperature: float = 4.0
 
    # ── Training ──────────────────────────────────────────────
    weight_decay: float = 0.01
    warmup_ratio: float = 0.10
    max_grad_norm: float = 1.0
    max_answer_tokens: int = 32
    num_workers: int = 4
    patience: int = 5
 
    # ── Feature projection ────────────────────────────────────
    # Teacher: LLaVA hidden dim = 4096  (Vicuna-7B)
    # Student: Moondream hidden dim = 2048 (Phi-1.5)
    teacher_hidden: int = 4096
    student_hidden: int = 2048
    projection_dim: int = 1024
 
    # ── Distillation alignment layer ──────────────────────────
    # Which layer to extract from each model for feature alignment.
    # "last_hidden" → mean-pool of the LM's last hidden state (default).
    # "vision_pool"  → mean-pool of vision-encoder output tokens.
    # Using "last_hidden" captures both visual and textual understanding.
    distill_layer: str = "last_hidden"
 
    # ── Label definitions ─────────────────────────────────────
    damage_labels: list = field(
        default_factory=lambda: ["no-damage", "minor-damage", "major-damage", "destroyed"]
    )
    binary_labels: list = field(default_factory=lambda: ["no", "yes"])
 
    # ── Question types ────────────────────────────────────────
    # Extended set — includes open-ended descriptive questions
    question_types: list = field(
        default_factory=lambda: [
            "overall_damage",       # classification
            "destroyed_count",      # count
            "major_count",          # count
            "minor_count",          # count
            "total_damaged_count",  # count
            "binary_damage",        # binary yes/no
            "describe_scene",       # open-ended: what do you see?
            "damage_extent",        # open-ended: describe damage extent
            "infrastructure",       # open-ended: roads/bridges visible?
        ]
    )
 
    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)
 
    @classmethod
    def load(cls, path: str) -> "Config":
        with open(path) as f:
            data = json.load(f)
        obj = cls()
        for k, v in data.items():
            if hasattr(obj, k):
                setattr(obj, k, v)
        return obj
 
 
# Singleton used by all scripts when imported directly
cfg = Config()
