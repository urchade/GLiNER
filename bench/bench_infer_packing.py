#!/usr/bin/env python
"""Benchmark harness for GLiNER inference-time packing."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from typing import Dict, List

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from gliner.infer_packing import InferencePackingConfig, pack_requests


@dataclass
class BenchmarkStats:
    tokens_per_s: float
    examples_per_s: float
    padding_ratio: float


def _format_table(result: Dict[str, object]) -> str:
    lines = []
    header = f"{'mode':<10} {'tokens/s':>15} {'examples/s':>15} {'padding':>12}"
    lines.append(header)
    lines.append("-" * len(header))
    for mode in ("baseline", "packed"):
        stats: BenchmarkStats = result[mode]  # type: ignore[assignment]
        lines.append(
            f"{mode:<10} {stats.tokens_per_s:>15.2e} {stats.examples_per_s:>15.2f} {stats.padding_ratio:>11.2%}"
        )
    lines.append("")
    lines.append(f"Speedup (tokens/s): {result['speedup_tokens_per_s']:.2f}x")
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark GLiNER inference packing.")
    parser.add_argument("--model", type=str, default="roberta-base", help="Model name or path")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=64, help="Number of requests per batch")
    parser.add_argument(
        "--scenario",
        type=str,
        default="short_uniform",
        choices=["short_uniform", "short_zipf", "mixed_tail", "flat_long"],
        help="Length distribution scenario",
    )
    parser.add_argument("--device", type=str, default="cpu", help="Device to benchmark on")
    parser.add_argument("--warmup", type=int, default=10, help="Number of warmup iterations")
    parser.add_argument("--iters", type=int, default=100, help="Number of timed iterations")
    return parser.parse_args()


def _generate_lengths(args: argparse.Namespace) -> List[int]:
    batch = args.batch_size
    max_length = args.max_length

    if args.scenario == "short_uniform":
        rng = np.random.default_rng(1337)
        values = rng.integers(8, min(64, max_length) + 1, size=batch)
        return values.astype(int).tolist()
    if args.scenario == "short_zipf":
        rng = np.random.default_rng(2024)
        lengths = rng.zipf(1.2, size=batch)
        clipped = np.clip(lengths, 8, min(128, max_length))
        return clipped.astype(int).tolist()
    if args.scenario == "mixed_tail":
        rng = np.random.default_rng(314)
        longs = [min(256, max_length)]
        if batch > 1:
            shorts = rng.integers(8, min(48, max_length) + 1, size=batch - 1)
            return longs + shorts.astype(int).tolist()
        return longs
    if args.scenario == "flat_long":
        return [min(256, max_length)] * batch

    raise ValueError(f"Unsupported scenario: {args.scenario}")


def _build_requests(lengths: List[int], vocab_size: int, pad_token_id: int) -> List[Dict[str, List[int]]]:
    requests: List[Dict[str, List[int]]] = []
    token = 0
    for length in lengths:
        actual_len = max(1, min(int(length), vocab_size - 1))
        sequence: List[int] = []
        for _ in range(actual_len):
            value = token % vocab_size
            if value == pad_token_id:
                value = (value + 1) % vocab_size
            sequence.append(value)
            token += 1
        requests.append({"input_ids": sequence})
    return requests


def _collate_baseline(requests: List[Dict[str, List[int]]], pad_token_id: int) -> Dict[str, torch.Tensor]:
    max_len = max(len(req["input_ids"]) for req in requests)
    batch = len(requests)
    input_ids = torch.full((batch, max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((batch, max_len), dtype=torch.long)
    for row, req in enumerate(requests):
        tokens = req["input_ids"]
        length = len(tokens)
        input_ids[row, :length] = torch.tensor(tokens, dtype=torch.long)
        attention_mask[row, :length] = 1
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def _measure(
    model: AutoModel,
    inputs: Dict[str, torch.Tensor],
    *,
    warmup: int,
    iters: int,
    device: torch.device,
) -> float:
    with torch.inference_mode():
        for _ in range(max(0, warmup)):
            model(**inputs)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(max(1, iters)):
            model(**inputs)
        if device.type == "cuda":
            torch.cuda.synchronize()
    return time.perf_counter() - start


def main() -> None:
    args = _parse_args()
    if args.max_length <= 0:
        raise ValueError("--max_length must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive")

    device = torch.device(args.device)
    torch.manual_seed(1337)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(1337)
        torch.backends.cudnn.deterministic = True

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model)
    model.to(device)
    model.eval()

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id or 0
    vocab_size = getattr(tokenizer, "vocab_size", len(tokenizer))
    if vocab_size is None:
        vocab_size = len(tokenizer)
    vocab_size = int(vocab_size)
    if vocab_size <= 1:
        raise ValueError("Tokenizer vocabulary size must exceed 1")
    lengths = _generate_lengths(args)
    lengths = [min(length, args.max_length) for length in lengths]
    requests = _build_requests(lengths, vocab_size, pad_token_id)
    real_tokens = sum(len(req["input_ids"]) for req in requests)

    baseline_inputs = _collate_baseline(requests, pad_token_id)
    baseline_inputs = {k: v.to(device) for k, v in baseline_inputs.items()}

    cfg = InferencePackingConfig(
        max_length=args.max_length,
        sep_token_id=tokenizer.sep_token_id,
        streams_per_batch=1,
    )
    packed = pack_requests(requests, cfg, pad_token_id)
    mask_dtype = baseline_inputs["attention_mask"].dtype
    packed_inputs = {
        "input_ids": packed.input_ids.to(device),
        "attention_mask": packed.pair_attention_mask.to(device=device, dtype=mask_dtype),
    }

    warmup = args.warmup
    iters = args.iters

    baseline_time = _measure(model, baseline_inputs, warmup=warmup, iters=iters, device=device)
    packed_time = _measure(model, packed_inputs, warmup=warmup, iters=iters, device=device)

    padded_tokens = baseline_inputs["input_ids"].size(1) * len(requests)
    baseline_stats = BenchmarkStats(
        tokens_per_s=(real_tokens * iters) / baseline_time,
        examples_per_s=(len(requests) * iters) / baseline_time,
        padding_ratio=1.0 - (real_tokens / padded_tokens) if padded_tokens else 0.0,
    )

    packed_tokens = packed.input_ids.size(1) * packed.input_ids.size(0)
    packed_stats = BenchmarkStats(
        tokens_per_s=(real_tokens * iters) / packed_time,
        examples_per_s=(len(requests) * iters) / packed_time,
        padding_ratio=1.0 - (real_tokens / packed_tokens) if packed_tokens else 0.0,
    )

    result = {
        "device": device.type,
        "model": args.model,
        "scenario": args.scenario,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "baseline": baseline_stats,
        "packed": packed_stats,
        "speedup_tokens_per_s": packed_stats.tokens_per_s / baseline_stats.tokens_per_s,
        "streams": packed.input_ids.size(0),
    }

    json_payload = {
        **{k: v for k, v in result.items() if k not in {"baseline", "packed"}},
        "baseline": asdict(baseline_stats),
        "packed": asdict(packed_stats),
    }

    print(json.dumps(json_payload, indent=2))
    print()
    print(_format_table(result))


if __name__ == "__main__":
    main()

