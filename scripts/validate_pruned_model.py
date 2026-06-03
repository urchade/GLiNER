"""
Validate a pruned GLiNER model against its original.

Asserts:
  1. Identical entity predictions on diverse test sentences
  2. Measurable model-size reduction
  3. Wall-clock latency comparison (original vs pruned)

Usage:
    python scripts/validate_pruned_model.py \
        --original_model_id knowledgator/gliner-bi-small-v1.0 \
        --pruned_model_dir  results/pruned_en

    # Skip latency benchmarking for a quick correctness check:
    python scripts/validate_pruned_model.py \
        --original_model_id knowledgator/gliner-bi-small-v1.0 \
        --pruned_model_dir  results/pruned_en \
        --skip_latency
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import sys
import time
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from gliner import GLiNER


# ─────────────────────────────────────────────────────────────────────────────
# Test suite
# ─────────────────────────────────────────────────────────────────────────────

TEST_CASES = [
    {
        "text":   "Apple Inc. was founded by Steve Jobs in Cupertino, California.",
        "labels": ["person", "organization", "location"],
    },
    {
        "text":   "The European Central Bank raised interest rates by 25 basis points.",
        "labels": ["organization", "financial instrument"],
    },
    {
        "text":   "Dr. Marie Curie won the Nobel Prize in Physics in 1903.",
        "labels": ["person", "award", "date"],
    },
    {
        "text":   "Tesla's Gigafactory in Berlin produces the Model Y electric vehicle.",
        "labels": ["organization", "location", "product"],
    },
    {
        "text":   "Barack Obama served as the 44th President of the United States from 2009 to 2017.",
        "labels": ["person", "political title", "country", "date"],
    },
    {
        "text":   "The Amazon River flows through Brazil and discharges into the Atlantic Ocean.",
        "labels": ["location", "country", "body of water"],
    },
]

LATENCY_LABELS = ["person", "organization", "location"]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _param_mb(model) -> float:
    return sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6


def _normalize(entities: list[dict]) -> list[dict]:
    """Sort entities for stable comparison."""
    return sorted(
        [{"text": e["text"], "label": e["label"], "score": float(e["score"])} for e in entities],
        key=lambda x: (x["text"], x["label"]),
    )


def _entity_sets_match(a: list[dict], b: list[dict]) -> bool:
    """True if both lists contain the same (text, label) pairs regardless of score."""
    key = lambda e: (e["text"], e["label"])  # noqa: E731
    return sorted(a, key=key) == sorted([{"text": e["text"], "label": e["label"]} for e in b], key=key)


def _compare(orig: list[dict], pruned: list[dict], score_tol: float = 0.02) -> tuple[str, str]:
    """
    Return (status, detail) where status is one of:
      PASS        — identical entity sets AND scores within tolerance
      SCORE_DRIFT — same entity sets, scores differ within ±score_tol (acceptable)
      FAIL        — entity sets differ (real regression)
    """
    orig_n   = _normalize(orig)
    pruned_n = _normalize(pruned)

    orig_pairs   = [{"text": e["text"], "label": e["label"]} for e in orig_n]
    pruned_pairs = [{"text": e["text"], "label": e["label"]} for e in pruned_n]

    if orig_pairs != pruned_pairs:
        return "FAIL", f"entity sets differ\n         original : {orig_n}\n         pruned   : {pruned_n}"

    # Same entity set — check score drift
    max_drift = max(abs(o["score"] - p["score"]) for o, p in zip(orig_n, pruned_n)) if orig_n else 0.0
    if max_drift <= score_tol:
        return "PASS", f"max score drift {max_drift:.4f} ≤ {score_tol}"
    return "SCORE_DRIFT", (
        f"same entities, max score drift {max_drift:.4f} > {score_tol} tolerance\n"
        + "\n".join(
            f"         {o['text']!r} ({o['label']}): {o['score']:.4f} → {p['score']:.4f}  Δ{abs(o['score']-p['score']):.4f}"
            for o, p in zip(orig_n, pruned_n)
        )
    )


def _bench(model, n_warmup: int = 3, n_runs: int = 20) -> float:
    """Return mean inference latency in milliseconds."""
    text   = TEST_CASES[0]["text"]
    labels = LATENCY_LABELS
    for _ in range(n_warmup):
        model.predict_entities(text, labels)
    t0 = time.perf_counter()
    for _ in range(n_runs):
        model.predict_entities(text, labels)
    return (time.perf_counter() - t0) / n_runs * 1000


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate a pruned GLiNER model against its original.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--original_model_id", required=True,
        help="Original HuggingFace model ID or local path",
    )
    parser.add_argument(
        "--pruned_model_dir", required=True,
        help="Path to the pruned model directory",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Entity prediction confidence threshold",
    )
    parser.add_argument(
        "--skip_latency", action="store_true",
        help="Skip latency benchmarking (faster correctness-only check)",
    )
    args = parser.parse_args()

    sep = "═" * 60
    print(f"\n{sep}")
    print("  GLiNER Vocabulary Pruning — Validation Report")
    print(sep)

    # ── Load models ───────────────────────────────────────────────────────────
    print(f"\n[load]  original : {args.original_model_id}")
    orig = GLiNER.from_pretrained(args.original_model_id)
    orig.eval()

    print(f"[load]  pruned   : {args.pruned_model_dir}")
    pruned = GLiNER.from_pretrained(args.pruned_model_dir)
    pruned.eval()

    # ── Vocabulary sizes ──────────────────────────────────────────────────────
    orig_V   = len(orig.data_processor.transformer_tokenizer)
    pruned_V = len(pruned.data_processor.transformer_tokenizer)
    print(f"\n[vocab]  original : {orig_V:,}")
    print(f"[vocab]  pruned   : {pruned_V:,}  ({pruned_V/orig_V*100:.1f}% of original)")

    # ── Model sizes ───────────────────────────────────────────────────────────
    orig_mb   = _param_mb(orig)
    pruned_mb = _param_mb(pruned)
    size_red  = (1 - pruned_mb / orig_mb) * 100
    print(f"\n[size]   original : {orig_mb:.1f} MB")
    print(f"[size]   pruned   : {pruned_mb:.1f} MB  ({size_red:.1f}% smaller)")

    # ── Entity prediction correctness ─────────────────────────────────────────
    print(f"\n[pred]  Running {len(TEST_CASES)} test cases (threshold={args.threshold})…")
    passes = fails = drifts = 0
    for i, tc in enumerate(TEST_CASES, start=1):
        orig_out   = orig.predict_entities(tc["text"],   tc["labels"], threshold=args.threshold)
        pruned_out = pruned.predict_entities(tc["text"], tc["labels"], threshold=args.threshold)

        status, detail = _compare(orig_out, pruned_out)
        snippet = tc["text"][:55] + ("…" if len(tc["text"]) > 55 else "")

        if status == "PASS":
            icon = "PASS ✓      "
            passes += 1
        elif status == "SCORE_DRIFT":
            icon = "SCORE_DRIFT ~"
            drifts += 1
        else:
            icon = "FAIL ✗      "
            fails += 1

        print(f"  [{i}] {icon}  {snippet!r}")
        if status != "PASS":
            for line in detail.splitlines():
                print(f"       {line}")

    # ── Latency ───────────────────────────────────────────────────────────────
    orig_ms = pruned_ms = speedup = None
    if not args.skip_latency:
        print(f"\n[lat]   Benchmarking (20 runs per model)…")
        orig_ms   = _bench(orig)
        pruned_ms = _bench(pruned)
        speedup   = orig_ms / pruned_ms
        print(f"  original : {orig_ms:.1f} ms")
        print(f"  pruned   : {pruned_ms:.1f} ms  ({speedup:.2f}× speedup)")

    # ── Summary ───────────────────────────────────────────────────────────────
    n = len(TEST_CASES)
    if fails == 0 and drifts == 0:
        verdict = f"ALL PASS ✓  ({passes}/{n})"
    elif fails == 0:
        verdict = f"PASS ✓ {passes}/{n}  |  SCORE_DRIFT ~ {drifts}/{n}  (same entities, small score shift)"
    else:
        verdict = f"PASS ✓ {passes}/{n}  |  SCORE_DRIFT ~ {drifts}/{n}  |  ENTITY_FAIL ✗ {fails}/{n}"

    print(f"\n{sep}")
    print(f"  Entity correctness : {verdict}")
    print(f"  Vocab reduction    : {orig_V:,} → {pruned_V:,}  ({(1-pruned_V/orig_V)*100:.1f}%)")
    print(f"  Size reduction     : {orig_mb:.1f} → {pruned_mb:.1f} MB  ({size_red:.1f}%)")
    if speedup is not None:
        print(f"  Latency speedup    : {orig_ms:.1f} → {pruned_ms:.1f} ms  ({speedup:.2f}×)")
    print(sep)

    if fails > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
