"""
Benchmark FlashDeBERTa vs standard DeBERTa attention across sequence lengths.

Usage:
    # Standard vs Flash at various token lengths
    python scripts/benchmark_flash_attention.py \
        --model_id urchade/gliner_multi-v2.1 \
        --output_dir results/flash_benchmark

    # Skip Flash (measure standard only, e.g. flashdeberta not installed)
    python scripts/benchmark_flash_attention.py \
        --model_id urchade/gliner_multi-v2.1 \
        --no_flash
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import sys
import time
import json
import argparse
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from gliner import GLiNER
from gliner.utils import is_module_available

LABELS = ["person", "organization", "location", "date", "product"]

# Token lengths to benchmark (approximate — text is padded/truncated to hit these)
TOKEN_LENGTHS = [128, 256, 384, 512, 768, 1024]

WARMUP_RUNS = 3
TIMED_RUNS = 10


def _make_text(n_words: int) -> str:
    """Generate a synthetic NER-like text of approximately n_words words."""
    base = (
        "Apple Inc. was founded by Steve Jobs in Cupertino California. "
        "The European Central Bank raised interest rates by 25 basis points. "
        "Dr Marie Curie won the Nobel Prize in Physics in 1903. "
    )
    repeats = (n_words // len(base.split())) + 1
    return " ".join((base * repeats).split()[:n_words])


def _bench_model(model, text: str, n_runs: int) -> dict:
    """Run n_runs inferences and return timing stats."""
    # Warmup
    for _ in range(WARMUP_RUNS):
        model.predict_entities(text, LABELS)

    latencies = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        model.predict_entities(text, LABELS)
        latencies.append((time.perf_counter() - t0) * 1000)

    latencies.sort()
    return {
        "mean_ms":   round(sum(latencies) / len(latencies), 2),
        "p50_ms":    round(latencies[len(latencies) // 2], 2),
        "p95_ms":    round(latencies[int(len(latencies) * 0.95)], 2),
        "min_ms":    round(latencies[0], 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark FlashDeBERTa vs standard DeBERTa across sequence lengths.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model_id", required=True, help="HuggingFace model ID or local path")
    parser.add_argument("--output_dir", default="results/flash_benchmark",
                        help="Directory to save benchmark CSV and plot")
    parser.add_argument("--no_flash", action="store_true",
                        help="Skip FlashDeBERTa benchmark (measure standard only)")
    parser.add_argument("--n_runs", type=int, default=TIMED_RUNS,
                        help="Number of timed inference runs per configuration")
    parser.add_argument("--max_token_length", type=int, default=1024,
                        help="Maximum token length to test")
    args = parser.parse_args()

    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    flash_available = is_module_available("flashdeberta")
    if not flash_available and not args.no_flash:
        print("[warn]  flashdeberta not installed. Install with: pip install flashdeberta")
        print("[warn]  Running standard-only benchmark.")
        args.no_flash = True

    token_lengths = [l for l in TOKEN_LENGTHS if l <= args.max_token_length]

    # ── Load models ───────────────────────────────────────────────────────────
    print(f"\n[load]  Standard DeBERTa: {args.model_id}")
    model_std = GLiNER.from_pretrained(args.model_id, flash_attention=False)
    model_std.eval()

    model_flash = None
    if not args.no_flash:
        print(f"[load]  FlashDeBERTa:     {args.model_id}")
        model_flash = GLiNER.from_pretrained(args.model_id, flash_attention=True)
        model_flash.eval()

    # ── Benchmark ─────────────────────────────────────────────────────────────
    rows = []
    sep = "─" * 70
    header = f"{'Tokens':>8} │ {'Std mean':>10} {'Std P95':>10} │ {'Flash mean':>10} {'Flash P95':>10} {'Speedup':>8}"
    print(f"\n{sep}\n{header}\n{sep}")

    for n_tokens in token_lengths:
        n_words = max(1, n_tokens - 2)  # approximate: 1 word ≈ 1 token for English
        text = _make_text(n_words)

        try:
            std_stats = _bench_model(model_std, text, args.n_runs)
        except Exception as e:
            print(f"  [skip] std at {n_tokens} tokens: {e}")
            continue

        if model_flash is not None:
            try:
                flash_stats = _bench_model(model_flash, text, args.n_runs)
                speedup = round(std_stats["mean_ms"] / flash_stats["mean_ms"], 2)
                flash_mean = f"{flash_stats['mean_ms']:>10.1f}"
                flash_p95  = f"{flash_stats['p95_ms']:>10.1f}"
                speedup_s  = f"{speedup:>7.2f}×"
            except Exception as e:
                flash_stats = {}
                flash_mean = flash_p95 = f"{'ERROR':>10}"
                speedup = None
                speedup_s = f"{'—':>8}"
        else:
            flash_stats = {}
            flash_mean = flash_p95 = f"{'—':>10}"
            speedup = None
            speedup_s = f"{'—':>8}"

        print(f"  {n_tokens:>6} │ {std_stats['mean_ms']:>10.1f} {std_stats['p95_ms']:>10.1f} │ {flash_mean} {flash_p95} {speedup_s}")

        row = {
            "n_tokens":       n_tokens,
            "std_mean_ms":    std_stats["mean_ms"],
            "std_p50_ms":     std_stats["p50_ms"],
            "std_p95_ms":     std_stats["p95_ms"],
            "flash_mean_ms":  flash_stats.get("mean_ms"),
            "flash_p50_ms":   flash_stats.get("p50_ms"),
            "flash_p95_ms":   flash_stats.get("p95_ms"),
            "speedup":        speedup,
        }
        rows.append(row)

    print(sep)

    # ── Save CSV ──────────────────────────────────────────────────────────────
    import csv
    csv_path = out_path / "flash_attention_benchmark.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[save]  CSV  → {csv_path}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    try:
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        xs = [r["n_tokens"] for r in rows]
        std_ms = [r["std_mean_ms"] for r in rows]
        flash_ms = [r["flash_mean_ms"] for r in rows if r["flash_mean_ms"] is not None]
        flash_xs = [r["n_tokens"] for r in rows if r["flash_mean_ms"] is not None]
        speedups = [r["speedup"] for r in rows if r["speedup"] is not None]

        ax1.plot(xs, std_ms, "o-", label="Standard DeBERTa", color="#e05252")
        if flash_xs:
            ax1.plot(flash_xs, flash_ms, "s-", label="FlashDeBERTa", color="#4caf50")
        ax1.set_xlabel("Sequence length (tokens)")
        ax1.set_ylabel("Latency (ms)")
        ax1.set_title("Inference Latency vs Sequence Length")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        if speedups:
            ax2.bar([str(x) for x in flash_xs], speedups, color="#4caf50", alpha=0.8)
            ax2.axhline(1.0, color="gray", linestyle="--", linewidth=1)
            ax2.set_xlabel("Sequence length (tokens)")
            ax2.set_ylabel("Speedup (×)")
            ax2.set_title("FlashDeBERTa Speedup over Standard")
            ax2.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        plot_path = out_path / "flash_attention_benchmark.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[save]  Plot → {plot_path}")
    except ImportError:
        print("[skip]  matplotlib not installed — skipping plot")

    # ── Summary ───────────────────────────────────────────────────────────────
    if any(r["speedup"] for r in rows):
        max_speedup = max(r["speedup"] for r in rows if r["speedup"])
        best_len = next(r["n_tokens"] for r in rows if r["speedup"] == max_speedup)
        print(f"\n  Peak speedup: {max_speedup:.2f}× at {best_len} tokens")


if __name__ == "__main__":
    main()
