"""Accuracy and latency of ONNX int8 dynamic quantization for a GLiNER model.

Exports the model to fp32 ONNX, builds two int8 variants with
``onnxruntime.quantization.quantize_dynamic`` and compares them on PII text:

* ``int8_uint8_per_tensor``: unsigned, per-tensor weights (what
  ``export_to_onnx(quantize=True)`` produced before this change)
* ``int8_int8_per_channel``: signed, per-channel weights (what it produces now,
  built through ``BaseGLiNER._maybe_quantize_onnx``)

Metrics:

* agreement with the fp32 ONNX model (entity-level precision / recall / F1 on
  exact ``(start, end, label)`` matches), i.e. "does quantization change what
  the model finds";
* precision / recall / F1 against the gold spans of the ai4privacy
  ``pii-masking-300k`` English validation split, for labels with an
  unambiguous natural-language name (given/last names are skipped);
* median single-text latency on the ONNX Runtime CPU provider, interleaved
  across variants for ``--rounds`` passes;
* file size.

Usage:
    python benchmarks/bench_onnx_int8.py --model urchade/gliner_small-v2.1 --n 200
"""

from __future__ import annotations

import os
import json
import time
import argparse
import tempfile
import warnings
import statistics
from pathlib import Path

from huggingface_hub import hf_hub_download
from onnxruntime.quantization import QuantType, quantize_dynamic

from gliner import GLiNER

GOLD_LABELS = {
    "EMAIL": "email",
    "TEL": "phone number",
    "IP": "ip address",
    "SOCIALNUMBER": "social security number",
    "PASSPORT": "passport number",
    "DRIVERLICENSE": "driver license number",
    "IDCARD": "id card number",
    "USERNAME": "username",
    "BOD": "date of birth",
    "CITY": "city",
    "POSTCODE": "postal code",
    "PASS": "password",
}


def load_texts(n: int) -> tuple[list[str], list[set]]:
    path = hf_hub_download(
        "ai4privacy/pii-masking-300k",
        "data/validation/1english_openpii_8k.jsonl",
        repo_type="dataset",
    )
    texts, gold = [], []
    with open(path) as f:
        for line in f:
            if len(texts) == n:
                break
            row = json.loads(line)
            texts.append(row["source_text"])
            gold.append(
                {
                    (m["start"], m["end"], GOLD_LABELS[m["label"]])
                    for m in row["privacy_mask"]
                    if m["label"] in GOLD_LABELS
                }
            )
    return texts, gold


def prf(pred: list[set], ref: list[set]) -> dict[str, float]:
    tp = sum(len(a & b) for a, b in zip(pred, ref, strict=True))
    n_pred, n_ref = sum(map(len, pred)), sum(map(len, ref))
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_ref if n_ref else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", default="urchade/gliner_small-v2.1")
    parser.add_argument("--n", type=int, default=200, help="Number of validation texts.")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--rounds", type=int, default=3, help="Interleaved latency passes.")
    parser.add_argument("--workdir", default=None, help="Where to write ONNX files (default: temp dir).")
    parser.add_argument("--output", default="benchmarks/bench_onnx_int8_results.json")
    args = parser.parse_args()
    warnings.filterwarnings("ignore")

    workdir = Path(args.workdir or tempfile.mkdtemp(prefix="gliner_int8_"))
    model = GLiNER.from_pretrained(args.model)
    if not (workdir / "model.onnx").exists():
        model.export_to_onnx(workdir, quantize=False)

    files = {"fp32": "model.onnx"}
    legacy = workdir / "int8_uint8_per_tensor.onnx"
    if not legacy.exists():
        quantize_dynamic(
            model_input=str(workdir / "model.onnx"),
            model_output=str(legacy),
            weight_type=QuantType.QUInt8,
        )
    files["int8_uint8_per_tensor"] = legacy.name
    current = workdir / "int8_int8_per_channel.onnx"
    if not current.exists():
        model._maybe_quantize_onnx(workdir / "model.onnx", current, quantize=True)
    files["int8_int8_per_channel"] = current.name
    del model

    texts, gold = load_texts(args.n)
    labels = sorted(set(GOLD_LABELS.values()))
    runtimes = {
        name: GLiNER.from_pretrained(str(workdir), load_onnx_model=True, load_tokenizer=True, onnx_model_file=fname)
        for name, fname in files.items()
    }

    preds = {name: [] for name in files}
    latencies = {name: [] for name in files}
    for rt in runtimes.values():  # warm up
        rt.inference(texts[:4], labels, threshold=args.threshold, batch_size=1)
    for rnd in range(args.rounds):
        for text in texts:
            for name, rt in runtimes.items():
                t0 = time.perf_counter()
                out = rt.inference([text], labels, threshold=args.threshold, batch_size=1)[0]
                latencies[name].append(time.perf_counter() - t0)
                if rnd == 0:
                    preds[name].append({(e["start"], e["end"], e["label"]) for e in out})

    results = {}
    for name, fname in files.items():
        results[name] = {
            "size_mb": round(os.path.getsize(workdir / fname) / 1e6, 1),
            "median_latency_ms": round(statistics.median(latencies[name]) * 1000, 1),
            "entities": sum(map(len, preds[name])),
            "agreement_with_fp32": prf(preds[name], preds["fp32"]),
            "vs_gold": prf(preds[name], gold),
        }
    report = {
        "model": args.model,
        "n_texts": args.n,
        "threshold": args.threshold,
        "latency_rounds": args.rounds,
        "dataset": "ai4privacy/pii-masking-300k, data/validation/1english_openpii_8k.jsonl (first n rows)",
        "labels": labels,
        "results": results,
    }
    print(json.dumps(report, indent=2))  # noqa: T201
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
