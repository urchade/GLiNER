#!/usr/bin/env python3
"""Benchmark quantization quality and speed for GLiNER.

Measures:
1. NER quality (strict F1) — does quantization degrade accuracy?
2. Inference latency — how much faster is quantized inference?

Test conditions:
- CPU fp32 (baseline)
- CPU int8 (torch.ao.quantization.quantize_dynamic)
- GPU fp32 (baseline)
- GPU fp16 (model.half() fallback)
- GPU int8 (bitsandbytes Linear8bitLt)

Datasets: CoNLL-2003 and WNUT-2017 test sets via knowledge_engine loaders.
Evaluation: nervaluate strict matching (exact boundary + type).

Usage:
    python benchmarks/bench_quantization.py
    python benchmarks/bench_quantization.py --sample-size 100 --speed-iters 10
"""

import gc
import json
import sys
import time
import warnings
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from statistics import mean, median, stdev

import numpy as np
import torch
from scipy import stats

# Add local GLiNER and knowledge_engine to path
GLINER_ROOT = Path(__file__).resolve().parent.parent
KE_ROOT = GLINER_ROOT.parent / "knowledge_engine"
sys.path.insert(0, str(GLINER_ROOT))
sys.path.insert(0, str(KE_ROOT))

from gliner import GLiNER

from src.core.entities import Document, Entity
from src.core.types import EntityType
from src.datasets.loaders import load_conll2003, load_wnut2017
from src.evaluation import evaluate_entities, extract_error_breakdown

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_ID = "urchade/gliner_medium-v2.1"
LABELS = ["person", "organization", "location"]
ENTITY_TYPES = ["PERSON", "ORGANIZATION", "LOCATION"]
THRESHOLD = 0.5

LABEL_TO_TYPE = {
    "person": EntityType.PERSON,
    "organization": EntityType.ORGANIZATION,
    "location": EntityType.LOCATION,
}

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class QualityResult:
    condition: str
    dataset: str
    f1: float
    precision: float
    recall: float
    num_docs: int
    num_gold: int
    num_pred: int


@dataclass
class SpeedResult:
    condition: str
    n_iters: int
    n_docs: int
    warmup_iters: int
    times_sec: list[float] = field(default_factory=list)

    @property
    def mean_sec(self):
        return mean(self.times_sec)

    @property
    def median_sec(self):
        return median(self.times_sec)

    @property
    def stdev_sec(self):
        return stdev(self.times_sec) if len(self.times_sec) > 1 else 0.0

    @property
    def throughput(self):
        """Docs per second based on median."""
        return self.n_docs / self.median_sec if self.median_sec > 0 else float("inf")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def predict_all(model: GLiNER, docs: list[tuple[Document, list[Entity]]]) -> list[tuple[Document, list[Entity]]]:
    """Run GLiNER inference on all documents and return (doc, entities) pairs."""
    predictions = []
    for doc, _ in docs:
        raw_preds = model.predict_entities(doc.text, LABELS, threshold=THRESHOLD)
        entities = []
        for p in raw_preds:
            label_key = p.get("label", "").lower()
            etype = LABEL_TO_TYPE.get(label_key)
            if etype is None:
                continue
            start, end = p.get("start", 0), p.get("end", 0)
            if start >= end:
                continue
            entities.append(
                Entity(text=p.get("text", ""), label=etype, start=start, end=end, confidence=p.get("score", 1.0))
            )
        predictions.append((doc, entities))
    return predictions


def evaluate_quality(
    model: GLiNER,
    gold_data: list[tuple[Document, list[Entity]]],
    condition: str,
    dataset_name: str,
) -> QualityResult:
    """Run evaluation, return QualityResult."""
    preds = predict_all(model, gold_data)
    results = evaluate_entities(gold_data, preds, entity_types=ENTITY_TYPES)
    strict = results.get("strict", {})
    return QualityResult(
        condition=condition,
        dataset=dataset_name,
        f1=strict.get("f1", 0.0),
        precision=strict.get("precision", 0.0),
        recall=strict.get("recall", 0.0),
        num_docs=len(gold_data),
        num_gold=sum(len(e) for _, e in gold_data),
        num_pred=sum(len(e) for _, e in preds),
    )


def measure_speed(
    model: GLiNER,
    docs: list[tuple[Document, list[Entity]]],
    condition: str,
    n_iters: int = 30,
    warmup_iters: int = 5,
) -> SpeedResult:
    """Measure inference latency over multiple iterations."""
    texts = [doc.text for doc, _ in docs]

    # Warmup
    for _ in range(warmup_iters):
        model.inference(texts, LABELS, threshold=THRESHOLD, batch_size=len(texts))

    if torch.cuda.is_available() and next(model.model.parameters(), torch.tensor(0)).is_cuda:
        torch.cuda.synchronize()

    times = []
    for _ in range(n_iters):
        if torch.cuda.is_available() and next(model.model.parameters(), torch.tensor(0)).is_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        model.inference(texts, LABELS, threshold=THRESHOLD, batch_size=len(texts))
        if torch.cuda.is_available() and next(model.model.parameters(), torch.tensor(0)).is_cuda:
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    return SpeedResult(
        condition=condition,
        n_iters=n_iters,
        n_docs=len(docs),
        warmup_iters=warmup_iters,
        times_sec=times,
    )


def load_model(device: str, quantize: bool = False, bnb_int8: bool = False) -> GLiNER:
    """Load model with specified device and quantization."""
    model = GLiNER.from_pretrained(MODEL_ID, map_location=device, quantize=quantize)
    if bnb_int8:
        model.quantize_bnb_int8()
    model.eval()
    return model


def free_model(model):
    """Delete model and free memory."""
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark GLiNER quantization quality and speed")
    parser.add_argument("--sample-size", type=int, default=200, help="Docs per dataset for quality evaluation")
    parser.add_argument("--speed-docs", type=int, default=30, help="Docs for speed benchmark")
    parser.add_argument("--speed-iters", type=int, default=30, help="Iterations for speed benchmark")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations for speed benchmark")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    parser.add_argument("--cpu-only", action="store_true", help="Skip GPU conditions")
    args = parser.parse_args()

    has_cuda = torch.cuda.is_available() and not args.cpu_only

    print("=" * 72)
    print("GLINER QUANTIZATION BENCHMARK")
    print("=" * 72)
    print(f"Timestamp:   {datetime.now().isoformat()}")
    print(f"Model:       {MODEL_ID}")
    print(f"Labels:      {LABELS}")
    print(f"Quality set: {args.sample_size} docs/dataset")
    print(f"Speed set:   {args.speed_docs} docs x {args.speed_iters} iters ({args.warmup} warmup)")
    print(f"CUDA:        {torch.cuda.get_device_name(0) if has_cuda else 'N/A'}")
    print(f"PyTorch:     {torch.__version__}")
    if has_cuda:
        try:
            import bitsandbytes as bnb

            print(f"BnB:         {bnb.__version__}")
        except ImportError:
            print("BnB:         not installed")
    print()

    # ------------------------------------------------------------------
    # Load datasets
    # ------------------------------------------------------------------
    print("Loading datasets...")
    conll_data = load_conll2003("test", sample_size=args.sample_size)
    wnut_data = load_wnut2017("test", sample_size=args.sample_size)
    print(f"  CoNLL-2003: {len(conll_data)} docs, {sum(len(e) for _, e in conll_data)} entities")
    print(f"  WNUT-2017:  {len(wnut_data)} docs, {sum(len(e) for _, e in wnut_data)} entities")

    # Subset for speed benchmark
    speed_data = conll_data[: args.speed_docs]

    # ------------------------------------------------------------------
    # Define conditions
    # ------------------------------------------------------------------
    # Each condition: (name, device, quantize_flag, post_action, bnb_int8)
    conditions = [
        ("cpu-fp32", "cpu", False, None, False),
        ("cpu-quantized", "cpu", True, None, False),
    ]
    if has_cuda:
        conditions += [
            ("gpu-fp32", "cuda", False, None, False),
            ("gpu-quantized", "cuda", True, None, False),
            ("gpu-bnb-int8", "cuda", False, None, True),
        ]

    quality_results: list[QualityResult] = []
    speed_results: list[SpeedResult] = []

    # ------------------------------------------------------------------
    # Run each condition
    # ------------------------------------------------------------------
    for cond_name, device, quant, post_action, bnb_int8 in conditions:
        print("\n" + "=" * 72)
        print(f"CONDITION: {cond_name}")
        print("=" * 72)

        # Load model
        print(f"  Loading model (device={device}, quantize={quant}, bnb_int8={bnb_int8})...")
        t0 = time.perf_counter()
        model = load_model(device, quantize=quant, bnb_int8=bnb_int8)
        load_time = time.perf_counter() - t0
        print(f"  Loaded in {load_time:.1f}s")

        # Quality evaluation
        for dname, ddata in [("CoNLL-2003", conll_data), ("WNUT-2017", wnut_data)]:
            print(f"  Evaluating quality on {dname}...")
            qr = evaluate_quality(model, ddata, cond_name, dname)
            quality_results.append(qr)
            print(f"    F1={qr.f1:.4f}  P={qr.precision:.4f}  R={qr.recall:.4f}")

        # Speed benchmark
        print(f"  Speed benchmark ({args.speed_docs} docs x {args.speed_iters} iters)...")
        sr = measure_speed(model, speed_data, cond_name, n_iters=args.speed_iters, warmup_iters=args.warmup)
        speed_results.append(sr)
        print(f"    Median: {sr.median_sec:.4f}s  Mean: {sr.mean_sec:.4f}s  Stdev: {sr.stdev_sec:.4f}s")
        print(f"    Throughput: {sr.throughput:.1f} docs/sec")

        free_model(model)

    # ------------------------------------------------------------------
    # Quality summary table
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("QUALITY RESULTS (Strict F1)")
    print("=" * 72)

    datasets = ["CoNLL-2003", "WNUT-2017"]
    cond_names = [c[0] for c in conditions]

    header = f"| {'Condition':<16} |"
    for ds in datasets:
        header += f" {ds:^14} |"
    print(header)
    print("|" + "-" * 18 + "|" + ("-" * 16 + "|") * len(datasets))

    # Find baselines for delta computation
    cpu_baseline = {ds: None for ds in datasets}
    gpu_baseline = {ds: None for ds in datasets}
    for qr in quality_results:
        if qr.condition == "cpu-fp32":
            cpu_baseline[qr.dataset] = qr.f1
        if qr.condition == "gpu-fp32":
            gpu_baseline[qr.dataset] = qr.f1

    for cond in cond_names:
        row = f"| {cond:<16} |"
        for ds in datasets:
            qr = next((q for q in quality_results if q.condition == cond and q.dataset == ds), None)
            if qr:
                # Show delta vs appropriate baseline
                baseline = gpu_baseline[ds] if cond.startswith("gpu") else cpu_baseline[ds]
                if baseline is not None and cond not in ("cpu-fp32", "gpu-fp32"):
                    delta = qr.f1 - baseline
                    row += f" {qr.f1:.4f} ({delta:+.4f})|"
                else:
                    row += f" {qr.f1:.4f}        |"
            else:
                row += f" {'N/A':^14} |"
        print(row)

    # ------------------------------------------------------------------
    # Speed summary table
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print(f"SPEED RESULTS ({args.speed_docs} docs, {args.speed_iters} iterations)")
    print("=" * 72)

    print(
        f"| {'Condition':<16} | {'Median(s)':>10} | {'Mean(s)':>10} | {'Stdev(s)':>10} "
        f"| {'Docs/sec':>10} | {'Speedup':>10} |"
    )
    print(f"|{'-' * 18}|{'-' * 12}|{'-' * 12}|{'-' * 12}|{'-' * 12}|{'-' * 12}|")

    # Baselines for speedup
    cpu_base_time = next((s.median_sec for s in speed_results if s.condition == "cpu-fp32"), None)
    gpu_base_time = next((s.median_sec for s in speed_results if s.condition == "gpu-fp32"), None)

    for sr in speed_results:
        baseline = gpu_base_time if sr.condition.startswith("gpu") else cpu_base_time
        if baseline and baseline > 0:
            speedup = baseline / sr.median_sec
            speedup_str = f"{speedup:.2f}x"
        else:
            speedup_str = "1.00x"

        print(
            f"| {sr.condition:<16} | {sr.median_sec:>10.4f} | {sr.mean_sec:>10.4f} | {sr.stdev_sec:>10.4f} "
            f"| {sr.throughput:>10.1f} | {speedup_str:>10} |"
        )

    # ------------------------------------------------------------------
    # Statistical significance (Welch's t-test: quantized vs baseline)
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("STATISTICAL TESTS (Welch's t-test, quantized vs fp32 baseline)")
    print("=" * 72)

    comparisons = [
        ("cpu-quantized", "cpu-fp32"),
    ]
    if has_cuda:
        comparisons += [
            ("gpu-quantized", "gpu-fp32"),
            ("gpu-bnb-int8", "gpu-fp32"),
        ]

    for quant_cond, base_cond in comparisons:
        sq = next((s for s in speed_results if s.condition == quant_cond), None)
        sb = next((s for s in speed_results if s.condition == base_cond), None)
        if sq and sb:
            t_stat, p_value = stats.ttest_ind(sb.times_sec, sq.times_sec, equal_var=False)
            sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
            direction = "FASTER" if mean(sq.times_sec) < mean(sb.times_sec) else "SLOWER"
            print(f"  {quant_cond} vs {base_cond}: t={t_stat:.3f}, p={p_value:.2e} {sig} ({direction})")

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "model": MODEL_ID,
        "labels": LABELS,
        "threshold": THRESHOLD,
        "pytorch_version": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(0) if has_cuda else None,
        "quality_sample_size": args.sample_size,
        "speed_docs": args.speed_docs,
        "speed_iters": args.speed_iters,
        "warmup_iters": args.warmup,
        "quality": [asdict(qr) for qr in quality_results],
        "speed": [
            {
                "condition": sr.condition,
                "n_iters": sr.n_iters,
                "n_docs": sr.n_docs,
                "warmup_iters": sr.warmup_iters,
                "times_sec": sr.times_sec,
                "mean_sec": sr.mean_sec,
                "median_sec": sr.median_sec,
                "stdev_sec": sr.stdev_sec,
                "throughput_docs_per_sec": sr.throughput,
            }
            for sr in speed_results
        ],
    }

    if args.output:
        out_path = Path(args.output)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = Path(__file__).parent / f"quantization_benchmark_{ts}.json"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to: {out_path}")

    print("\n" + "=" * 72)
    print("BENCHMARK COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    main()
