"""Step 1 — Baseline evaluation script.

Zero-shot evaluation of a pretrained GLiNER checkpoint on WNUT-17 and CoNLL-2003.
Also measures CPU inference latency and computes the positive-span imbalance ratio
that mathematically motivates the Focal / Dice loss work in Step 2.

Usage:
    python scripts/baseline_eval.py \
        --model knowledgator/gliner-bi-small-v1.0 \
        --output results/baseline_table.csv
"""
from __future__ import annotations

# KMP_DUPLICATE_LIB_OK suppresses the macOS ARM OpenMP abort that occurs when
# PyTorch (libiomp5) and accelerate/transformers (libomp) are both in-process.
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Optional

import torch

try:
    from datasets import load_dataset
except ImportError:
    sys.exit("pip install datasets")

try:
    from gliner import GLiNER
except ImportError:
    sys.exit("pip install -e '.[training]' from repo root")


# ---------------------------------------------------------------------------
# Dataset tag maps
# ---------------------------------------------------------------------------

DATASETS = {
    "wnut17": {
        "hf_name": "wnut_17",
        "split": "test",
        "labels": ["person", "location", "corporation", "creative-work", "group", "product"],
        "tag_to_label": {
            1: "corporation", 2: "corporation",
            3: "creative-work", 4: "creative-work",
            5: "group", 6: "group",
            7: "location", 8: "location",
            9: "person", 10: "person",
            11: "product", 12: "product",
        },
    },
    "conll2003": {
        "hf_name": "conll2003",
        "split": "test",
        "labels": ["person", "organization", "location", "miscellaneous"],
        "tag_to_label": {
            1: "person", 2: "person",
            3: "organization", 4: "organization",
            5: "location", 6: "location",
            7: "miscellaneous", 8: "miscellaneous",
        },
    },
}


# ---------------------------------------------------------------------------
# BIO → GLiNER span format
# ---------------------------------------------------------------------------

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
    """Convert HF NER rows → GLiNER internal format {tokenized_text, ner}."""
    out = []
    for ex in examples:
        out.append({
            "tokenized_text": ex["tokens"],
            "ner": bio_to_spans(ex["tokens"], ex["ner_tags"], tag_to_label),
        })
    return out


# ---------------------------------------------------------------------------
# Span imbalance analysis
# ---------------------------------------------------------------------------

def compute_span_imbalance(examples: list, tag_to_label: dict, max_width: int = 12) -> dict:
    total_cands, total_pos = 0, 0
    for ex in examples:
        L = len(ex["tokens"])
        if L == 0:
            continue
        k_max = min(max_width, L)
        total_cands += sum(L - k + 1 for k in range(1, k_max + 1))
        spans = bio_to_spans(ex["tokens"], ex["ner_tags"], tag_to_label)
        total_pos += sum(1 for s, e, _ in spans if (e - s + 1) <= max_width)
    ratio = total_pos / total_cands if total_cands > 0 else 0.0
    return {
        "total_candidate_spans": total_cands,
        "total_positive_spans": total_pos,
        "positive_ratio": ratio,
        "imbalance_factor": (total_cands - total_pos) / max(total_pos, 1),
    }


# ---------------------------------------------------------------------------
# Latency measurement — single sentence, batch_size=1
# ---------------------------------------------------------------------------

def measure_latency(model: GLiNER, sentence: str, labels: list,
                    n_warmup: int = 5, n_repeats: int = 30) -> dict:
    for _ in range(n_warmup):
        model.predict_entities(sentence, labels, threshold=0.5)
    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        model.predict_entities(sentence, labels, threshold=0.5)
        times.append((time.perf_counter() - t0) * 1000.0)
    ts = sorted(times)
    return {
        "latency_mean_ms": round(sum(times) / len(times), 2),
        "latency_p50_ms":  round(ts[len(ts) // 2], 2),
        "latency_p95_ms":  round(ts[int(len(ts) * 0.95)], 2),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model",    default="knowledgator/gliner-bi-small-v1.0")
    p.add_argument("--datasets", nargs="+", default=["wnut17", "conll2003"])
    p.add_argument("--output",   default="results/baseline_table.csv")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--max_width", type=int,   default=12)
    p.add_argument("--latency_repeats", type=int, default=30)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("  GLiNER-Robust — Step 1: Baseline Evaluation")
    print(f"{'='*60}")

    print("\nLoading model...")
    model = GLiNER.from_pretrained(args.model)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    param_mb  = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2
    print(f"  Params: {n_params/1e6:.1f}M  |  Memory: {param_mb:.0f} MB (FP32)\n")

    all_rows = []

    for ds_name in args.datasets:
        cfg = DATASETS[ds_name]
        print(f"{'─'*60}")
        print(f"  Dataset: {ds_name.upper()}")
        print(f"{'─'*60}")

        raw = list(load_dataset(cfg["hf_name"], split=cfg["split"], trust_remote_code=True))
        gliner_data = hf_to_gliner(raw, cfg["tag_to_label"])
        print(f"  Loaded {len(raw)} examples")

        # Span imbalance
        print("  Computing span imbalance...")
        imb = compute_span_imbalance(raw, cfg["tag_to_label"], args.max_width)
        print(f"    Positive spans    : {imb['total_positive_spans']:,}")
        print(f"    Total candidates  : {imb['total_candidate_spans']:,}")
        print(f"    Positive ratio    : {imb['positive_ratio']:.4%}")
        print(f"    Imbalance factor  : {imb['imbalance_factor']:.0f}× more negatives")

        # Latency
        sample_sentence = " ".join(raw[0]["tokens"])
        print(f"\n  Measuring latency ({args.latency_repeats} runs, batch_size=1)...")
        lat = measure_latency(model, sample_sentence, cfg["labels"],
                              n_repeats=args.latency_repeats)
        print(f"    Mean: {lat['latency_mean_ms']} ms  |  P50: {lat['latency_p50_ms']} ms  |  P95: {lat['latency_p95_ms']} ms")

        # NER F1 via model.evaluate()
        print(f"\n  Running zero-shot NER evaluation (labels: {cfg['labels']})...")
        _, f1 = model.evaluate(
            gliner_data,
            flat_ner=True,
            threshold=args.threshold,
            batch_size=8,
            entity_types=cfg["labels"],
        )
        print(f"    F1: {f1*100:.2f}%\n")

        all_rows.append({
            "dataset":               ds_name,
            "model":                 args.model,
            "f1":                    round(float(f1), 4),
            "f1_pct":                round(float(f1) * 100, 2),
            **lat,
            "model_params_M":        round(n_params / 1e6, 1),
            "model_memory_MB":       round(param_mb, 1),
            "total_candidate_spans": imb["total_candidate_spans"],
            "total_positive_spans":  imb["total_positive_spans"],
            "positive_span_ratio":   round(imb["positive_ratio"], 6),
            "imbalance_factor":      round(imb["imbalance_factor"], 1),
            "threshold":             args.threshold,
        })

    # Write CSV
    if all_rows:
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"  Results → {args.output}")

    # Summary
    print(f"\n{'='*60}  SUMMARY")
    print(f"  {'Dataset':<12} {'F1':>8} {'Latency (ms)':>14} {'Imbalance':>12}")
    print(f"  {'-'*12} {'-'*8} {'-'*14} {'-'*12}")
    for r in all_rows:
        print(f"  {r['dataset']:<12} {r['f1_pct']:>7.2f}% "
              f"{r['latency_mean_ms']:>12.1f}ms {r['imbalance_factor']:>10.0f}×")
    print()
    print("  ★  Imbalance factor = negatives/positives — the mathematical")
    print("     justification for Focal + Dice loss. Paste into the paper.\n")


if __name__ == "__main__":
    main()
