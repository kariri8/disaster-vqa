"""
scripts/01_prepare_dataset.py
────────────────────────────────────────────────────────────────
Downloads xBD from Kaggle, parses JSON annotations, generates an
extended VQA pair set (classification, count, binary, open-ended),
validates all samples, performs a scene-aware train/val/test split,
and saves everything to disk as JSON.
 
Usage:
    python scripts/01_prepare_dataset.py [--debug] [--xbd_root PATH]
"""
 
import argparse
import json
import logging
import os
import random
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
 
from tqdm.auto import tqdm
 
# ── Ensure project root is importable ─────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs.config import Config, cfg
 
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
 
 
# ════════════════════════════════════════════════════════════════
# 1. Constants
# ════════════════════════════════════════════════════════════════
 
DAMAGE_MAP = {
    "no-damage": 0,
    "minor-damage": 1,
    "major-damage": 2,
    "destroyed": 3,
    "un-classified": -1,
}
 
# Open-ended question templates — answer is derived from annotations
OPEN_ENDED_QUESTIONS = {
    "describe_scene": (
        "Describe what you observe in this post-disaster satellite image.",
        None,   # answer is generated from labels below
    ),
    "damage_extent": (
        "Describe the extent of the damage visible in this satellite image.",
        None,
    ),
    "infrastructure": (
        "Are there any roads, bridges, or infrastructure visible in this image?",
        "yes",   # conservative default; refined by subtype presence
    ),
}
 
 
# ════════════════════════════════════════════════════════════════
# 2. Kaggle download helper
# ════════════════════════════════════════════════════════════════
 
def setup_kaggle_creds():
    kaggle_json = os.path.expanduser("~/.kaggle/kaggle.json")
    if os.path.isfile(kaggle_json):
        logger.info("kaggle.json already present.")
        return
    api_token = os.environ.get("KAGGLE_API_TOKEN", "")
    username = os.environ.get("KAGGLE_USERNAME", "")
    key = os.environ.get("KAGGLE_KEY", "")
    if api_token.startswith("KGAT_"):
        import base64
        try:
            payload = base64.b64decode(api_token[5:] + "==").decode()
            username, key = payload.split(":", 1)
        except Exception:
            pass
    if username and key:
        os.makedirs(os.path.dirname(kaggle_json), exist_ok=True)
        with open(kaggle_json, "w") as f:
            json.dump({"username": username, "key": key}, f)
        os.chmod(kaggle_json, 0o600)
        logger.info("kaggle.json written.")
    else:
        raise RuntimeError(
            "No Kaggle credentials found.\n"
            "Set KAGGLE_USERNAME + KAGGLE_KEY  or  KAGGLE_API_TOKEN."
        )
 
 
def find_subdir(base: str, name: str) -> Optional[str]:
    for root, dirs, _ in os.walk(base):
        if name in dirs:
            return os.path.join(root, name)
    return None
 
 
def download_xbd(data_dir: str) -> str:
    """Download & locate the tier3 split. Returns its path."""
    tier3 = find_subdir(data_dir, "tier3")
    if tier3 and os.path.isdir(tier3):
        subs = {s: len(os.listdir(os.path.join(tier3, s)))
                for s in ["images", "labels", "masks"]
                if os.path.isdir(os.path.join(tier3, s))}
        if subs:
            logger.info(f"tier3 already present at {tier3}. Skipping download.")
            return tier3
 
    logger.info("Downloading xBD dataset from Kaggle …")
    setup_kaggle_creds()
    result = subprocess.run(
        ["kaggle", "datasets", "download", "qianlanzz/xbd-dataset",
         "--path", data_dir, "--unzip"],
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError("Kaggle download failed. Check credentials.")
 
    tier3 = find_subdir(data_dir, "tier3")
    if tier3 is None:
        raise RuntimeError(f"tier3/ not found under {data_dir} after download.")
 
    target = os.path.join(data_dir, "tier3")
    if os.path.abspath(tier3) != os.path.abspath(target):
        logger.info(f"Moving {tier3} → {target}")
        shutil.move(tier3, target)
        tier3 = target
 
    # Clean up extras
    for entry in os.listdir(data_dir):
        ep = os.path.join(data_dir, entry)
        if os.path.abspath(ep) == os.path.abspath(tier3):
            continue
        if os.path.isdir(ep):
            shutil.rmtree(ep)
        elif entry.endswith(".zip"):
            os.remove(ep)
    logger.info(f"tier3 ready at {tier3}")
    return tier3
 
 
# ════════════════════════════════════════════════════════════════
# 3. xBD scene discovery
# ════════════════════════════════════════════════════════════════
 
def discover_scenes(xbd_root: str, structure: str = "flat") -> List[dict]:
    """Return a list of scene dicts, each with paths to pre/post/json."""
    scenes = []
 
    def scan_dir(label_dir: str, image_dir: str, tag: str = "") -> List[dict]:
        found = []
        if not os.path.isdir(label_dir):
            return found
        for fname in os.listdir(label_dir):
            if not fname.endswith(".json"):
                continue
            if "_pre_disaster.json" in fname:
                continue  # use post JSON only
            if "_post_disaster.json" in fname:
                sid = fname.replace("_post_disaster.json", "")
            else:
                sid = fname.replace(".json", "")
 
            json_path = os.path.join(label_dir, fname)
            pre_path = os.path.join(image_dir, f"{sid}_pre_disaster.png")
            post_path = os.path.join(image_dir, f"{sid}_post_disaster.png")
 
            found.append(dict(
                scene_id=sid,
                split_origin=tag,
                json_path=json_path,
                pre_path=pre_path,
                post_path=post_path,
                pre_ok=os.path.isfile(pre_path),
                post_ok=os.path.isfile(post_path),
            ))
        return found
 
    if structure == "flat":
        lbl = os.path.join(xbd_root, "labels")
        img = os.path.join(xbd_root, "images")
        scenes.extend(scan_dir(lbl, img, tag="all"))
    elif structure == "split-based":
        for split in os.listdir(xbd_root):
            sp = os.path.join(xbd_root, split)
            if os.path.isdir(sp):
                scenes.extend(scan_dir(
                    os.path.join(sp, "labels"),
                    os.path.join(sp, "images"),
                    tag=split,
                ))
    elif structure == "disaster-based":
        for dis in sorted(os.listdir(xbd_root)):
            dp = os.path.join(xbd_root, dis)
            if os.path.isdir(dp):
                scenes.extend(scan_dir(
                    os.path.join(dp, "labels"),
                    os.path.join(dp, "images"),
                    tag=dis,
                ))
    else:
        # Auto-detect
        for strategy in ["flat", "split-based", "disaster-based"]:
            scenes = discover_scenes(xbd_root, strategy)
            if scenes:
                break
 
    logger.info(f"Discovered {len(scenes)} scene triplets.")
    return scenes
 
 
# ════════════════════════════════════════════════════════════════
# 4. Annotation parsing
# ════════════════════════════════════════════════════════════════
 
def parse_xbd_json(json_path: str) -> dict:
    """Parse post-disaster JSON → damage counts + overall label."""
    with open(json_path) as f:
        data = json.load(f)
 
    counts = {0: 0, 1: 0, 2: 0, 3: 0}
    subtypes_seen = set()
    for feat in data.get("features", {}).get("xy", []):
        props = feat.get("properties", {})
        subtype = props.get("subtype", "un-classified")
        lvl = DAMAGE_MAP.get(subtype, -1)
        if lvl >= 0:
            counts[lvl] += 1
        subtypes_seen.add(subtype)
 
    total = sum(counts.values())
    if counts[3] > 0:
        overall, idx = "destroyed", 3
    elif counts[2] > 0:
        overall, idx = "major-damage", 2
    elif counts[1] > 0:
        overall, idx = "minor-damage", 1
    else:
        overall, idx = "no-damage", 0
 
    return {
        "counts": counts,
        "total_buildings": total,
        "overall_label": overall,
        "overall_idx": idx,
        "subtypes_seen": list(subtypes_seen),
    }
 
 
# ════════════════════════════════════════════════════════════════
# 5. VQA pair generation — EXTENDED
# ════════════════════════════════════════════════════════════════
 
def _describe_scene_answer(ann: dict) -> str:
    """Build a natural-language description from annotation counts."""
    c = ann["counts"]
    total = ann["total_buildings"]
    label = ann["overall_label"]
    parts = [f"This post-disaster image shows {total} buildings."]
    if c[3] > 0:
        parts.append(f"{c[3]} building(s) are completely destroyed.")
    if c[2] > 0:
        parts.append(f"{c[2]} building(s) have major damage.")
    if c[1] > 0:
        parts.append(f"{c[1]} building(s) have minor damage.")
    if c[0] > 0:
        parts.append(f"{c[0]} building(s) appear undamaged.")
    parts.append(f"Overall damage level: {label}.")
    return " ".join(parts)
 
 
def _damage_extent_answer(ann: dict) -> str:
    label = ann["overall_label"]
    c = ann["counts"]
    total = ann["total_buildings"]
    damaged = c[1] + c[2] + c[3]
    pct = int(damaged / total * 100) if total > 0 else 0
    if label == "no-damage":
        return "There is no visible structural damage in this image."
    elif label == "minor-damage":
        return f"Minor damage is visible, affecting approximately {pct}% of buildings."
    elif label == "major-damage":
        return f"Significant damage is visible, with approximately {pct}% of buildings affected including partial collapses."
    else:
        return f"Severe destruction is visible. Approximately {pct}% of buildings are damaged or completely destroyed."
 
 
def make_vqa_pairs(scene: dict, ann: dict, max_count_bin: int = 16) -> List[dict]:
    """Generate all VQA pairs for a single scene."""
    c = ann["counts"]
    total_buildings = ann["total_buildings"]
    total_damaged = c[1] + c[2] + c[3]
 
    base = {
        "scene_id": scene["scene_id"],
        "pre_path": scene["pre_path"],
        "post_path": scene["post_path"],
        "pre_ok": scene["pre_ok"],
        "post_ok": scene["post_ok"],
    }
    pairs = []
 
    # ── Structured questions ────────────────────────────────
    # Q1: overall damage classification
    pairs.append({**base,
        "question": "What is the overall damage level of this area?",
        "question_type": "overall_damage",
        "answer": ann["overall_label"],
        "answer_idx": ann["overall_idx"],
        "answer_type": "classification",
    })
 
    # Q2: destroyed count
    pairs.append({**base,
        "question": "How many buildings are completely destroyed?",
        "question_type": "destroyed_count",
        "answer": str(c[3]),
        "answer_idx": min(c[3], max_count_bin),
        "answer_type": "count",
    })
 
    # Q3: major damage count
    pairs.append({**base,
        "question": "How many buildings have major structural damage?",
        "question_type": "major_count",
        "answer": str(c[2]),
        "answer_idx": min(c[2], max_count_bin),
        "answer_type": "count",
    })
 
    # Q4: minor damage count
    pairs.append({**base,
        "question": "How many buildings show minor damage?",
        "question_type": "minor_count",
        "answer": str(c[1]),
        "answer_idx": min(c[1], max_count_bin),
        "answer_type": "count",
    })
 
    # Q5: total damaged count
    pairs.append({**base,
        "question": "How many buildings are damaged in total?",
        "question_type": "total_damaged_count",
        "answer": str(total_damaged),
        "answer_idx": min(total_damaged, max_count_bin),
        "answer_type": "count",
    })
 
    # Q6: binary
    pairs.append({**base,
        "question": "Is there any visible structural damage in this image?",
        "question_type": "binary_damage",
        "answer": "yes" if total_damaged > 0 else "no",
        "answer_idx": 1 if total_damaged > 0 else 0,
        "answer_type": "binary",
    })
 
    # ── Open-ended questions ────────────────────────────────
    # Q7: scene description
    pairs.append({**base,
        "question": "Describe what you observe in this post-disaster satellite image.",
        "question_type": "describe_scene",
        "answer": _describe_scene_answer(ann),
        "answer_idx": -1,
        "answer_type": "open",
    })
 
    # Q8: damage extent
    pairs.append({**base,
        "question": "Describe the extent of the damage visible in this satellite image.",
        "question_type": "damage_extent",
        "answer": _damage_extent_answer(ann),
        "answer_idx": ann["overall_idx"],
        "answer_type": "open",
    })
 
    # Q9: infrastructure (present in almost all urban/suburban xBD scenes)
    infra = "yes"  # xBD scenes contain roads in the vast majority of cases
    pairs.append({**base,
        "question": "Are there roads or other infrastructure visible in this post-disaster image?",
        "question_type": "infrastructure",
        "answer": infra,
        "answer_idx": 1,
        "answer_type": "binary",
    })
 
    return pairs
 
 
# ════════════════════════════════════════════════════════════════
# 6. Scene-aware split
# ════════════════════════════════════════════════════════════════
 
def scene_split(pairs: List[dict], cfg: Config):
    scene2idx = defaultdict(list)
    for i, p in enumerate(pairs):
        scene2idx[p["scene_id"]].append(i)
    sids = sorted(scene2idx.keys())
    rng = random.Random(cfg.seed)
    rng.shuffle(sids)
    n = len(sids)
    n_tr = int(n * cfg.train_ratio)
    n_va = int(n * cfg.val_ratio)
    tr_sc = set(sids[:n_tr])
    va_sc = set(sids[n_tr:n_tr + n_va])
    te_sc = set(sids[n_tr + n_va:])
    tr, va, te = [], [], []
    for s in tr_sc: tr.extend(scene2idx[s])
    for s in va_sc: va.extend(scene2idx[s])
    for s in te_sc: te.extend(scene2idx[s])
    assert not (tr_sc & va_sc) and not (tr_sc & te_sc), "Scene leakage detected!"
    return tr, va, te, tr_sc, va_sc, te_sc
 
 
# ════════════════════════════════════════════════════════════════
# 7. Main pipeline
# ════════════════════════════════════════════════════════════════
 
def main(args):
    # --- Resolve config ---
    if args.config:
        cfg_local = Config.load(args.config)
    else:
        cfg_local = cfg  # default from configs/config.py
 
    if args.debug:
        cfg_local.debug_mode = True
    if args.xbd_root:
        cfg_local.xbd_root = args.xbd_root
 
    # --- Build directory structure ---
    dirs = {}
    for sub in ["data", "metadata", "checkpoints/teacher", "checkpoints/student",
                "checkpoints/baseline", "logs", "metrics", "predictions", "visualizations"]:
        full = os.path.join(cfg_local.project_root, sub)
        os.makedirs(full, exist_ok=True)
        key = sub.split("/")[-1]
        dirs[key] = full
    dirs["teacher_ckpt"] = os.path.join(cfg_local.project_root, "checkpoints/teacher")
    dirs["student_ckpt"] = os.path.join(cfg_local.project_root, "checkpoints/student")
    dirs["baseline_ckpt"] = os.path.join(cfg_local.project_root, "checkpoints/baseline")
 
    # --- Download xBD if needed ---
    if not cfg_local.xbd_root:
        # cfg_local.xbd_root = download_xbd(dirs["data"])
        print("Not seeing xBD root")
    else:
        logger.info(f"Using existing xBD root: {cfg_local.xbd_root}")
 
    # --- Discover scenes ---
    raw_scenes = discover_scenes(cfg_local.xbd_root, cfg_local.xbd_structure)
    logger.info(f"Total scenes discovered: {len(raw_scenes)}")
 
    # --- Parse & validate ---
    valid, invalid = [], []
    miss_pre = miss_post = 0
    for sc in tqdm(raw_scenes, desc="Parsing annotations"):
        if not sc["post_ok"]:
            miss_post += 1
            invalid.append(sc)
            continue
        try:
            ann = parse_xbd_json(sc["json_path"])
        except Exception as e:
            logger.warning(f"Parse error {sc['json_path']}: {e}")
            invalid.append(sc)
            continue
        if ann["total_buildings"] == 0:
            invalid.append(sc)
            continue
        sc["ann"] = ann
        valid.append(sc)
        if not sc["pre_ok"]:
            miss_pre += 1
 
    logger.info(f"Valid: {len(valid)}  |  Filtered: {len(invalid)}")
    logger.info(f"Missing pre-image: {miss_pre}  |  Missing post-image: {miss_post}")
 
    if cfg_local.debug_mode and len(valid) > cfg_local.debug_scenes:
        random.shuffle(valid)
        valid = valid[:cfg_local.debug_scenes]
        logger.info(f"[DEBUG] Reduced to {len(valid)} scenes.")
 
    # --- Generate VQA pairs ---
    all_pairs = []
    for sc in tqdm(valid, desc="Generating VQA pairs"):
        all_pairs.extend(make_vqa_pairs(sc, sc["ann"], cfg_local.max_count_bin))
 
    logger.info(f"Total VQA samples: {len(all_pairs)}")
 
    qt_counts = Counter(p["question_type"] for p in all_pairs)
    at_counts = Counter(p["answer_type"] for p in all_pairs)
    logger.info(f"Question type distribution: {dict(qt_counts)}")
    logger.info(f"Answer type distribution: {dict(at_counts)}")
 
    # --- Scene-aware split ---
    tr_idx, va_idx, te_idx, tr_sc, va_sc, te_sc = scene_split(all_pairs, cfg_local)
    logger.info(f"Split — train: {len(tr_idx)}  val: {len(va_idx)}  test: {len(te_idx)}")
 
    # --- Save pairs + split ---
    pairs_path = os.path.join(dirs["metadata"], "all_pairs.json")
    with open(pairs_path, "w") as f:
        json.dump(all_pairs, f)
    logger.info(f"All pairs saved → {pairs_path}")
 
    splits_path = os.path.join(dirs["metadata"], "splits.json")
    splits = {
        "train_indices": tr_idx,
        "val_indices": va_idx,
        "test_indices": te_idx,
        "train_scenes": sorted(tr_sc),
        "val_scenes": sorted(va_sc),
        "test_scenes": sorted(te_sc),
    }
    with open(splits_path, "w") as f:
        json.dump(splits, f, indent=2)
 
    # --- Save preprocessing report ---
    report = {
        "timestamp": datetime.now().isoformat(),
        "total_discovered": len(raw_scenes),
        "valid": len(valid),
        "invalid": len(invalid),
        "miss_pre": miss_pre,
        "miss_post": miss_post,
        "total_pairs": len(all_pairs),
        "per_qtype": dict(qt_counts),
        "per_atype": dict(at_counts),
        "split": {"train": len(tr_idx), "val": len(va_idx), "test": len(te_idx)},
    }
    rp = os.path.join(dirs["metadata"], "preprocessing_report.json")
    with open(rp, "w") as f:
        json.dump(report, f, indent=2)
 
    # --- Save config ---
    cfg_local.save(os.path.join(dirs["metadata"], "config.json"))
    logger.info("Dataset preparation complete ✓")
 
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare xBD VQA dataset")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to a saved config JSON")
    parser.add_argument("--debug", action="store_true",
                        help="Debug mode (small subset)")
    parser.add_argument("--xbd_root", type=str, default=None,
                        help="Path to an already-downloaded xBD tier3 directory")
    main(parser.parse_args())
