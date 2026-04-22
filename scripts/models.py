import torch
import torch.nn as nn
from transformers import (
    LlavaForConditionalGeneration,
    LlavaProcessor,
    BitsAndBytesConfig,
    get_linear_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs.config import Config

# ════════════════════════════════════════════════════════════════
# GPU / dtype helpers
# ════════════════════════════════════════════════════════════════

def get_device_info():
    if not torch.cuda.is_available():
        return torch.device("cpu"), torch.float32, False
    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability()[0]
    dtype = torch.bfloat16 if cap >= 8 else torch.float16
    # High-end GPUs have enough VRAM to skip 4-bit for smaller models
    high_end = any(x in name for x in ("H200", "H100", "A100"))
    use_4bit = not high_end
    device = torch.device("cuda")
    if cap >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    return device, dtype, use_4bit

DEVICE, DTYPE, USE_4BIT = get_device_info()

# ════════════════════════════════════════════════════════════════
# Feature projector (shared logic remains the same)
# ════════════════════════════════════════════════════════════════

class FeatureProjector(nn.Module):
    def __init__(self, input_dim: int, proj_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, proj_dim),
            nn.GELU(),
            nn.LayerNorm(proj_dim),
            nn.Linear(proj_dim, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

def build_projectors(cfg: Config, student_hidden: int) -> tuple:
    print(f"  Projectors: teacher={cfg.teacher_hidden}d → {cfg.projection_dim}d, "
          f"student={student_hidden}d → {cfg.projection_dim}d")
    t_proj = FeatureProjector(cfg.teacher_hidden, cfg.projection_dim).to(DEVICE, dtype=DTYPE)
    s_proj = FeatureProjector(student_hidden, cfg.projection_dim).to(DEVICE, dtype=DTYPE)
    return t_proj, s_proj

# ════════════════════════════════════════════════════════════════
# Teacher: LLaVA-1.5-7B
# ════════════════════════════════════════════════════════════════

def load_teacher(cfg: Config, apply_lora: bool = True):
    print(f"Loading Teacher: {cfg.teacher_model_id} …")
    processor = LlavaProcessor.from_pretrained(cfg.teacher_model_id)

    bnb_cfg = None
    if USE_4BIT:
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=DTYPE,
            bnb_4bit_use_double_quant=True,
        )

    model = LlavaForConditionalGeneration.from_pretrained(
        cfg.teacher_model_id,
        quantization_config=bnb_cfg,
        device_map="auto",
        torch_dtype=DTYPE,
    )

    if USE_4BIT:
        model = prepare_model_for_kbit_training(model)

    if apply_lora:
        lora_cfg = LoraConfig(
            r=cfg.teacher_lora_r,
            lora_alpha=cfg.teacher_lora_alpha,
            lora_dropout=cfg.teacher_lora_dropout,
            target_modules=cfg.teacher_lora_target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_cfg)

    return model, processor

# ════════════════════════════════════════════════════════════════
# Student: Tiny-LLaVA (Phi-2 or TinyLlama based)
# ════════════════════════════════════════════════════════════════

def get_student_vision_hidden(model) -> int:
    """Infer the vision encoder output dim from the loaded model."""
    actual = model.base_model.model if hasattr(model, "base_model") else model
    
    # Navigating the Hugging Face LLaVA hierarchy safely
    if hasattr(actual, "model") and hasattr(actual.model, "vision_tower"):
        vision_tower = actual.model.vision_tower
    else:
        vision_tower = actual.vision_tower

    # Try common attribute paths
    for attr in ["hidden_size", "config.hidden_size", "embed_dim"]:
        try:
            parts = attr.split(".")
            obj = vision_tower
            for p in parts:
                obj = getattr(obj, p)
            return int(obj)
        except AttributeError:
            continue
            
    # Fallback: run a dummy forward
    import torch
    dummy = torch.zeros(1, 3, 336, 336, dtype=next(vision_tower.parameters()).dtype,
                        device=next(vision_tower.parameters()).device)
    with torch.no_grad():
        out = vision_tower(dummy, output_hidden_states=True)
    return out.last_hidden_state.shape[-1]

def load_student(cfg: Config, apply_qlora: bool = True):
    # Recommended: "tiny-llava/tiny-llava-phi-2-siglip-2.7b"
    # or "bczhou/tiny-llava-v1-hf" (for a ~1.5B version)
    student_id = cfg.student_model_id 
    
    tag = "Student" if apply_qlora else "Baseline"
    print(f"Loading {tag}: {student_id} …")

    # Load Processor (Use same class as Teacher)
    processor = LlavaProcessor.from_pretrained(student_id)

    # processor.patch_size = 14
    # processor.vision_feature_select_strategy = "default"
    # processor.image_processor.patch_size = 14
    # processor.image_processor.vision_feature_select_strategy = "default"

    if not hasattr(processor, "patch_size") or processor.patch_size is None:
        # Standard for CLIP-ViT-L/14 used in LLaVA/TinyLLaVA
        processor.patch_size = 14 
    if not hasattr(processor, "vision_feature_select_strategy") or processor.vision_feature_select_strategy is None:
        processor.vision_feature_select_strategy = "default"

    bnb_cfg = None
    if apply_qlora:
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=DTYPE,
            bnb_4bit_use_double_quant=True,
        )

    # Use LlavaForConditionalGeneration instead of AutoModel
    model = LlavaForConditionalGeneration.from_pretrained(
        student_id,
        quantization_config=bnb_cfg,
        device_map={"": DEVICE},
        torch_dtype=DTYPE,
    )

    if apply_qlora:
        model = prepare_model_for_kbit_training(model)
        
        # Standard LLaVA target modules
        lora_cfg = LoraConfig(
            r=cfg.student_lora_r,
            lora_alpha=cfg.student_lora_alpha,
            lora_dropout=cfg.student_lora_dropout,
            target_modules=["q_proj", "v_proj", "k_proj", "out_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()

    return model, processor

# ════════════════════════════════════════════════════════════════
# Optimizer / Scheduler remains the same
# ════════════════════════════════════════════════════════════════

def build_optimizer_scheduler(params, n_samples: int, batch_size: int,
                               grad_accum: int, epochs: int, cfg: Config):
    optimizer = torch.optim.AdamW(params, lr=cfg.student_lr, weight_decay=cfg.weight_decay)
    steps_per_epoch = max(n_samples // (batch_size * grad_accum), 1)
    total_steps = steps_per_epoch * epochs
    warmup_steps = int(total_steps * cfg.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    return optimizer, scheduler