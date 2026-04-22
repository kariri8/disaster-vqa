"""
scripts/07_demo.py
────────────────────────────────────────────────────────────────
Interactive terminal demo for the distilled student model.
Prints sample Q&A from the test set, then lets you load any
image and chat with the model.

Usage:
    python scripts/07_demo.py [--config PATH] [--ckpt PATH] [--n_samples 10]
"""

import argparse
import os
import random
import sys
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs.config import Config, cfg
from scripts.datasets import StudentVQADataset, load_pairs_and_splits
from scripts.models import DEVICE, load_student
from scripts.utils import parse_answer

BOLD  = "\033[1m"
CYAN  = "\033[96m"
GREEN = "\033[92m"
YELLOW= "\033[93m"
RED   = "\033[91m"
RESET = "\033[0m"


def load_model(cfg_local, ckpt_path):
    from scripts.models import load_teacher
    from peft import PeftModel
    model, processor = load_teacher(cfg_local)
    lora_dir = ckpt_path or os.path.join(
        cfg_local.project_root, "checkpoints/teacher/best_lora")
    if os.path.isdir(lora_dir):
        model = PeftModel.from_pretrained(model, lora_dir)
        print(f"{GREEN}✓ Loaded LoRA: {lora_dir}{RESET}")
    else:
        print(f"{YELLOW}⚠ No LoRA found at {lora_dir}, using base weights{RESET}")
    model.eval()
    return model, processor


def ask(model, processor, image: Image.Image, question: str, cfg_local) -> str:
    prompt = (
        f"USER: <image>\n"
        f"This image shows a post-disaster satellite image.\n"
        f"{question}\nASSISTANT:"
    )
    inputs = processor(
        text=[prompt], images=[image],
        return_tensors="pt", padding=True, max_length=1024,
    ).to(DEVICE)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=cfg_local.max_answer_tokens,
            do_sample=False,
            repetition_penalty=1.3,
        )
    raw = processor.decode(
        output_ids[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )
    for stop in ["ASSISTANT:", "USER:", "\n"]:
        if stop in raw:
            raw = raw.split(stop)[0]
    return raw.strip()


def print_samples(test_ds, model, tokenizer, cfg_local, n):
    print(f"\n{BOLD}{'═'*65}{RESET}")
    print(f"{BOLD}  SAMPLE PREDICTIONS FROM TEST SET{RESET}")
    print(f"{BOLD}{'═'*65}{RESET}")

    indices = random.sample(range(len(test_ds)), min(n, len(test_ds)))
    for i, idx in enumerate(indices):
        item = test_ds[idx]
        pred_raw = ask(model, tokenizer, item["image"], item["question"], cfg_local)
        pred = parse_answer(pred_raw)
        correct = pred == item["answer"].lower()
        marker = f"{GREEN}✓{RESET}" if correct else f"{RED}✗{RESET}"

        print(f"\n  {CYAN}[{i+1}] {item['question_type'].upper()}{RESET}")
        print(f"  Q : {item['question']}")
        print(f"  A : {GREEN}{item['answer']}{RESET}")
        print(f"  P : {pred_raw}  {marker}")
        print(f"  Image: {item.get('post_path', 'N/A')}")

    print(f"\n{BOLD}{'═'*65}{RESET}")


def chat_loop(model, tokenizer, cfg_local):
    print(f"\n{BOLD}  INTERACTIVE CHAT{RESET}")
    print(f"  Commands:")
    print(f"    {YELLOW}load <path>{RESET}  — load a new image")
    print(f"    {YELLOW}quit{RESET}         — exit\n")

    current_image = None
    current_path = None

    while True:
        try:
            user_input = input(f"{CYAN}> {RESET}").strip()
        except (KeyboardInterrupt, EOFError):
            print(f"\n{YELLOW}Bye!{RESET}")
            break

        if not user_input:
            continue

        if user_input.lower() == "quit":
            print(f"{YELLOW}Bye!{RESET}")
            break

        if user_input.lower().startswith("load "):
            path = user_input[5:].strip().strip('"')
            if not os.path.isfile(path):
                print(f"{RED}  File not found: {path}{RESET}")
                continue
            try:
                current_image = Image.open(path).convert("RGB")
                current_path = path
                print(f"{GREEN}  ✓ Loaded: {path}{RESET}")
            except Exception as e:
                print(f"{RED}  Failed to load image: {e}{RESET}")
            continue

        if current_image is None:
            print(f"{YELLOW}  No image loaded. Use: load <path>{RESET}")
            continue

        print(f"  {YELLOW}thinking...{RESET}", end="\r")
        answer = ask(model, tokenizer, current_image, user_input, cfg_local)
        print(f"  {GREEN}{answer}{RESET}          ")  # spaces clear "thinking..."


def main(args):
    cfg_local = Config.load(args.config) if args.config else cfg

    meta_dir = os.path.join(cfg_local.project_root, "metadata")
    all_pairs, splits = load_pairs_and_splits(meta_dir)
    test_ds = StudentVQADataset(
        all_pairs, splits["test_indices"], cfg_local.image_size)

    print(f"{BOLD}Loading student model...{RESET}")
    model, tokenizer = load_model(cfg_local, args.ckpt)

    print_samples(test_ds, model, tokenizer, cfg_local, args.n_samples)
    chat_loop(model, tokenizer, cfg_local)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",    type=str, default=None)
    parser.add_argument("--ckpt",      type=str, default=None,
                        help="Path to student checkpoint (default: checkpoints/student/best.pt)")
    parser.add_argument("--n_samples", type=int, default=10,
                        help="Number of test samples to show before chat")
    main(parser.parse_args())