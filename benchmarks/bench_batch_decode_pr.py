"""
Rigorous benchmark: batch-level _decode_batch vs per-item _decode_batch_item loop.

Methodology (per project benchmarking rules):
- Interleaved A/B within the same process to avoid warm-cache bias
- n=50 replications per condition after 10 warmup iterations
- Batch sizes: 1, 8, 16, 32
- Input lengths: short (~20 tokens), medium (~80), long (~200), very_long (~500)
- CPU and GPU
- Reports mean, median, stdev, Welch's t-test
- Outputs structured JSON for analysis script
"""

import json
import time
import statistics
import sys
import platform
import torch
import random
import numpy as np
from scipy import stats
from unittest.mock import Mock
from gliner.decoding.decoder import SpanDecoder


N_WARMUP = 10
N_REPS = 50
BATCH_SIZES = [1, 8, 16, 32]
INPUT_LENGTHS = [
    ("short", 20),
    ("medium", 80),
    ("long", 200),
    ("very_long", 500),
]
MAX_WIDTH = 12
NUM_CLASSES = 8
THRESHOLD = 0.5


def make_inputs(batch_size, seq_len, max_width, num_classes, device):
    """Create realistic decoder inputs.

    Real GLiNER models produce mostly low logits with sparse high-confidence
    spans. We bias logits negative so ~5-50 spans per item pass threshold=0.5.
    """
    logits = torch.randn(batch_size, seq_len, max_width, num_classes, device=device) - 3.0
    num_hot = max(5, seq_len // 10)
    for b in range(batch_size):
        for _ in range(num_hot):
            s = random.randint(0, seq_len - 1)
            k = random.randint(0, min(max_width - 1, seq_len - s - 1))
            c = random.randint(0, num_classes - 1)
            logits[b, s, k, c] = random.uniform(1.0, 6.0)
    probs = torch.sigmoid(logits)
    tokens = [[f"tok_{j}" for j in range(seq_len)] for _ in range(batch_size)]
    id_to_classes = {i + 1: f"CLASS_{i}" for i in range(num_classes)}
    return probs, tokens, id_to_classes


def run_old_path(decoder, probs, tokens, id_to_classes, K):
    B = probs.shape[0]
    spans = []
    for i in range(B):
        span_i = decoder._decode_batch_item(
            probs_i=probs[i],
            tokens_i=tokens[i],
            id_to_class_i=id_to_classes,
            K=K,
            threshold=THRESHOLD,
            flat_ner=True,
            multi_label=False,
            span_label_map={},
        )
        spans.append(span_i)
    return spans


def run_new_path(decoder, probs, tokens, id_to_classes, K):
    B = probs.shape[0]
    return decoder._decode_batch(
        probs=probs,
        tokens=tokens,
        id_to_classes=id_to_classes,
        K=K,
        threshold=THRESHOLD,
        flat_ner=True,
        multi_label=False,
        span_label_maps=[{} for _ in range(B)],
    )


def benchmark_condition(decoder, probs, tokens, id_to_classes, K, device):
    old_times = []
    new_times = []

    for rep in range(N_WARMUP + N_REPS):
        # Alternate which runs first to avoid ordering bias
        order = ["old", "new"] if rep % 2 == 0 else ["new", "old"]

        for which in order:
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            if which == "old":
                run_old_path(decoder, probs, tokens, id_to_classes, K)
            else:
                run_new_path(decoder, probs, tokens, id_to_classes, K)

            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - t0) * 1000

            if rep >= N_WARMUP:
                (old_times if which == "old" else new_times).append(elapsed_ms)

    return old_times, new_times


def count_spans(decoder, probs, tokens, id_to_classes, K):
    """Count how many spans pass threshold for this condition (for reporting)."""
    B = probs.shape[0]
    total = 0
    for i in range(B):
        spans = decoder._decode_batch_item(
            probs_i=probs[i], tokens_i=tokens[i], id_to_class_i=id_to_classes,
            K=K, threshold=THRESHOLD, flat_ner=True, multi_label=False,
            span_label_map={},
        )
        total += len(spans)
    return total


def verify_correctness(decoder, probs, tokens, id_to_classes, K):
    """Verify old and new paths produce identical output."""
    B = probs.shape[0]
    old = run_old_path(decoder, probs, tokens, id_to_classes, K)
    new = run_new_path(decoder, probs, tokens, id_to_classes, K)
    assert len(old) == len(new) == B
    for i in range(B):
        o = [(s.start, s.end, s.entity_type, round(s.score, 8)) for s in old[i]]
        n = [(s.start, s.end, s.entity_type, round(s.score, 8)) for s in new[i]]
        assert o == n, f"Mismatch at batch {i}: {o} != {n}"


def main():
    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)

    config = Mock()
    config.max_width = MAX_WIDTH
    decoder = SpanDecoder(config)

    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))

    results = []
    env_info = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "cpu": platform.processor() or "unknown",
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "n_warmup": N_WARMUP,
        "n_reps": N_REPS,
        "max_width": MAX_WIDTH,
        "num_classes": NUM_CLASSES,
        "threshold": THRESHOLD,
    }

    total = len(devices) * len(INPUT_LENGTHS) * len(BATCH_SIZES)
    done = 0

    for device in devices:
        device_name = "GPU" if device.type == "cuda" else "CPU"

        for len_label, seq_len in INPUT_LENGTHS:
            for bs in BATCH_SIZES:
                done += 1
                label = f"{device_name} bs={bs:>2} {len_label:>10}"
                print(f"[{done}/{total}] {label} ...", end="", flush=True)

                probs, tokens, id_to_classes = make_inputs(
                    bs, seq_len, MAX_WIDTH, NUM_CLASSES, device
                )

                # Verify correctness before benchmarking
                verify_correctness(decoder, probs, tokens, id_to_classes, MAX_WIDTH)

                span_count = count_spans(decoder, probs, tokens, id_to_classes, MAX_WIDTH)

                old_times, new_times = benchmark_condition(
                    decoder, probs, tokens, id_to_classes, MAX_WIDTH, device
                )

                old_mean = statistics.mean(old_times)
                old_med = statistics.median(old_times)
                old_std = statistics.stdev(old_times)
                new_mean = statistics.mean(new_times)
                new_med = statistics.median(new_times)
                new_std = statistics.stdev(new_times)
                t_stat, p_val = stats.ttest_ind(old_times, new_times, equal_var=False)
                pct_median = (old_med - new_med) / old_med * 100
                pct_mean = (old_mean - new_mean) / old_mean * 100

                r = {
                    "device": device_name,
                    "batch_size": bs,
                    "seq_len": seq_len,
                    "len_label": len_label,
                    "span_count": span_count,
                    "old_mean": old_mean, "old_median": old_med, "old_stdev": old_std,
                    "new_mean": new_mean, "new_median": new_med, "new_stdev": new_std,
                    "pct_change_median": pct_median,
                    "pct_change_mean": pct_mean,
                    "t_stat": t_stat, "p_value": p_val,
                    "old_times": old_times,
                    "new_times": new_times,
                }
                results.append(r)
                sig = "*" if p_val < 0.05 else " "
                print(
                    f" old={old_med:.3f}ms  new={new_med:.3f}ms  "
                    f"{pct_median:+.1f}%{sig}  ({span_count} spans)"
                )

    output = {"env": env_info, "results": results}
    out_path = "benchmarks/bench_batch_decode_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
