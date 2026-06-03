"""Step 2 — Loss function ablation training script.

Runs five compact fine-tuning experiments comparing loss configurations
on CoNLL-2003 training data, evaluating each checkpoint on WNUT-17.

Configs (in order):
    bce          — alpha=-1, gamma=0  (plain BCE, baseline)
    focal_025    — alpha=0.25, gamma=2
    focal_070    — alpha=0.70, gamma=2  (bi-encoder paper defaults)
    dice         — SpanDiceLoss, gamma=1.0
    dice_width   — SpanDiceLoss + span-width weighting

Usage:
    python scripts/train_ablation.py \
        --base_model urchade/gliner-multitask-large-v0.5 \
        --max_steps 200 \
        --output_dir results/ablation
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch

try:
    from datasets import load_dataset
except ImportError:
    sys.exit("pip install datasets")

try:
    from gliner import GLiNER
    from gliner.training import TrainingArguments
except ImportError:
    sys.exit("pip install -e '.[training]' from repo root")


# ---------------------------------------------------------------------------
# BIO → GLiNER format
# ---------------------------------------------------------------------------

WNUT17_TAG_TO_LABEL = {
    1: "corporation",  2: "corporation",
    3: "creative-work", 4: "creative-work",
    5: "group",        6: "group",
    7: "location",     8: "location",
    9: "person",      10: "person",
    11: "product",    12: "product",
}
WNUT17_LABELS = ["person", "location", "corporation", "creative-work", "group", "product"]

CONLL_TAG_TO_LABEL = {
    1: "person",        2: "person",
    3: "organization",  4: "organization",
    5: "location",      6: "location",
    7: "miscellaneous", 8: "miscellaneous",
}


def bio_to_spans(tokens: list, tags: list, tag_to_label: dict) -> list:
    spans, start, label = [], None, None
    for i, tag in enumerate(tags):
        lbl = tag_to_label.get(tag)
        if lbl is None:
            if start is not None:
                spans.append([start, i - 1, label])
                start, label = None, None
            continue
        if tag % 2 == 1 or label != lbl:
            if start is not None:
                spans.append([start, i - 1, label])
            start, label = i, lbl
    if start is not None:
        spans.append([start, len(tags) - 1, label])
    return spans


def hf_to_gliner(examples: list, tag_to_label: dict) -> list:
    return [
        {"tokenized_text": ex["tokens"],
         "ner": bio_to_spans(ex["tokens"], ex["ner_tags"], tag_to_label)}
        for ex in examples
    ]


# ---------------------------------------------------------------------------
# Ablation configurations
# ---------------------------------------------------------------------------

@dataclass
class AblationCfg:
    name: str
    description: str
    loss_type: str
    focal_alpha: float
    focal_gamma: float
    dice_gamma: float
    use_span_width_weight: bool


ABLATION_CONFIGS = [
    AblationCfg("bce",        "Plain BCE (alpha=-1, gamma=0)",                  "focal", -1.0, 0.0, 1.0, False),
    AblationCfg("focal_025",  "Focal (alpha=0.25, gamma=2)",                    "focal",  0.25, 2.0, 1.0, False),
    AblationCfg("focal_070",  "Focal (alpha=0.70, gamma=2) — bi-encoder paper", "focal",  0.70, 2.0, 1.0, False),
    AblationCfg("dice",       "Span Dice Loss (gamma=1.0)",                     "dice",  -1.0, 0.0, 1.0, False),
    AblationCfg("dice_width", "Dice + span-width weighting",                    "dice",  -1.0, 0.0, 1.0, True),
]


# ---------------------------------------------------------------------------
# Single config run
# ---------------------------------------------------------------------------

def run_one(cfg: AblationCfg, base_model: str, train_data: list, eval_data: list,
            output_dir: Path, max_steps: int, batch_size: int, threshold: float) -> dict:
    print(f"\n{'─'*60}")
    print(f"  Config: {cfg.name}  —  {cfg.description}")
    print(f"{'─'*60}")

    run_dir = output_dir / cfg.name
    run_dir.mkdir(parents=True, exist_ok=True)

    model = GLiNER.from_pretrained(base_model)

    training_args = TrainingArguments(
        output_dir=str(run_dir),
        max_steps=max_steps,
        per_device_train_batch_size=batch_size,
        learning_rate=5e-5,
        weight_decay=0.01,
        others_lr=1e-4,
        others_weight_decay=0.0,
        # ── Loss function config ──────────────────────────────────────────
        loss_type=cfg.loss_type,
        focal_loss_alpha=cfg.focal_alpha,
        focal_loss_gamma=cfg.focal_gamma,
        dice_gamma=cfg.dice_gamma,
        use_span_width_weight=cfg.use_span_width_weight,
        # ──────────────────────────────────────────────────────────────────
        negatives=1.0,
        masking="global",
        loss_reduction="sum",
        logging_steps=50,
        save_steps=max_steps,
        save_total_limit=1,
        report_to="none",
        dataloader_num_workers=0,
    )

    t0 = time.perf_counter()
    model.train_model(
        train_dataset=train_data,
        eval_dataset=None,
        training_args=training_args,
    )
    train_time = time.perf_counter() - t0
    print(f"  Trained in {train_time:.0f}s")

    print("  Evaluating on WNUT-17...")
    wnut_data = hf_to_gliner(eval_data, WNUT17_TAG_TO_LABEL)
    _, f1 = model.evaluate(
        wnut_data,
        flat_ner=True,
        threshold=threshold,
        batch_size=8,
        entity_types=WNUT17_LABELS,
    )
    f1 = float(f1)
    print(f"  WNUT-17 F1: {f1*100:.2f}%")

    return {
        "config":              cfg.name,
        "description":         cfg.description,
        "loss_type":           cfg.loss_type,
        "focal_alpha":         cfg.focal_alpha,
        "focal_gamma":         cfg.focal_gamma,
        "dice_gamma":          cfg.dice_gamma,
        "use_span_width_weight": cfg.use_span_width_weight,
        "wnut17_f1":           round(f1, 4),
        "wnut17_f1_pct":       round(f1 * 100, 2),
        "max_steps":           max_steps,
        "train_time_s":        round(train_time, 1),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", default="urchade/gliner-multitask-large-v0.5")
    p.add_argument("--output_dir", default="results/ablation")
    p.add_argument("--max_steps",  type=int, default=200,
                   help="Steps per config. 200 is enough to see loss function differences.")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--train_max",  type=int, default=1500,
                   help="Cap training examples for speed")
    p.add_argument("--threshold",  type=float, default=0.5)
    p.add_argument("--configs",    nargs="+", default=None,
                   help="Subset of config names. Omit to run all 5.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("  GLiNER-Robust — Step 2: Loss Function Ablation")
    print(f"{'='*60}\n")

    print("Loading datasets...")
    raw_train = list(load_dataset("conll2003", split="train", trust_remote_code=True))[:args.train_max]
    raw_eval  = list(load_dataset("wnut_17",   split="test",  trust_remote_code=True))
    train_data = hf_to_gliner(raw_train, CONLL_TAG_TO_LABEL)
    print(f"  Train: {len(train_data)} examples (CoNLL-2003)")
    print(f"  Eval:  {len(raw_eval)} examples (WNUT-17)\n")

    configs = ABLATION_CONFIGS
    if args.configs:
        configs = [c for c in ABLATION_CONFIGS if c.name in args.configs]

    results = []
    for cfg in configs:
        row = run_one(cfg, args.base_model, train_data, raw_eval,
                      output_dir, args.max_steps, args.batch_size, args.threshold)
        results.append(row)

    csv_path = output_dir / "ablation_results.csv"
    if results:
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)

    print(f"\n{'='*60}  ABLATION RESULTS — WNUT-17 F1")
    print(f"  {'Config':<14} {'F1 (%)':>8}  Description")
    print(f"  {'-'*14} {'-'*8}  {'-'*35}")
    for r in results:
        print(f"  {r['config']:<14} {r['wnut17_f1_pct']:>7.2f}%  {r['description']}")
    print(f"\n  Saved → {csv_path}\n")


if __name__ == "__main__":
    main()
