#!/usr/bin/env python3
"""Benchmark GLiNER StreamingSpan inference across sequence lengths and devices.

Normal mode processes the complete text in one stateless call. Streaming mode
feeds one model-split word per call while retaining the model's session KV cache.

Example:
    python benchmarks/bench_streaming.py models/checkpoint-15000 \
        --devices cpu,cuda --lengths 16,32,64,128 --repeats 5
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from uuid import uuid4

import torch

from gliner import GLiNER


DEFAULT_LABELS = ["person", "organization", "location", "date"]
CORPUS_WORDS = (
    "Alice Johnson joined Acme Corporation in London on Monday before meeting "
    "Robert Smith from Global Research Institute in Paris to discuss a new "
    "technology project supported by the European Commission and Stanford University"
).split()


@dataclass(frozen=True)
class Result:
    device: str
    mode: str
    requested_words: int
    model_words: int
    transformer_tokens: int
    median_seconds: float
    mean_seconds: float
    stdev_seconds: float
    words_per_second: float
    tokens_per_second: float
    repeats: int


def parse_csv(value: str, cast=str) -> list:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError("expected a non-empty comma-separated list")
    try:
        return [cast(item) for item in items]
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def make_text(word_count: int) -> str:
    words = [CORPUS_WORDS[index % len(CORPUS_WORDS)] for index in range(word_count)]
    return " ".join(words)


def word_chunks(model, text: str) -> list[str]:
    """Split text into appendable chunks using GLiNER's own word splitter."""
    token_batches, _, end_batches = model.prepare_inputs([text])
    tokens = token_batches[0]
    ends = end_batches[0]
    chunks: list[str] = []
    previous_end = 0
    for index, end in enumerate(ends):
        chunk_end = len(text) if index == len(tokens) - 1 else end
        chunks.append(text[previous_end:chunk_end])
        previous_end = chunk_end
    return chunks


def transformer_token_count(model, text: str) -> int:
    tokenizer = model.data_processor.transformer_tokenizer
    encoded = tokenizer(text, add_special_tokens=False)
    input_ids = encoded["input_ids"]
    if input_ids and isinstance(input_ids[0], list):
        input_ids = input_ids[0]
    return len(input_ids)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def run_once(model, mode: str, text: str, chunks: list[str], labels: list[str], threshold: float) -> None:
    if mode == "normal":
        model.inference([text], labels, threshold=threshold)
        return

    session_id = f"benchmark-{uuid4().hex}"
    model.clear_session(session_id)
    try:
        for chunk in chunks:
            model.inference([chunk], labels, session_id=[session_id], threshold=threshold)
    finally:
        model.clear_session(session_id)


def measure(
    model,
    device: torch.device,
    mode: str,
    text: str,
    chunks: list[str],
    labels: list[str],
    threshold: float,
    warmups: int,
    repeats: int,
) -> list[float]:
    for _ in range(warmups):
        run_once(model, mode, text, chunks, labels, threshold)
    synchronize(device)

    samples = []
    for _ in range(repeats):
        synchronize(device)
        started = time.perf_counter()
        run_once(model, mode, text, chunks, labels, threshold)
        synchronize(device)
        samples.append(time.perf_counter() - started)
    return samples


def resolve_devices(requested: list[str]) -> list[torch.device]:
    devices: list[torch.device] = []
    for name in requested:
        if name == "auto":
            name = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(name)
        if device.type == "cuda" and not torch.cuda.is_available():
            print(f"Skipping {name}: CUDA is not available")
            continue
        if device.type == "mps" and not torch.backends.mps.is_available():
            print(f"Skipping {name}: MPS is not available")
            continue
        devices.append(device)
    if not devices:
        raise SystemExit("None of the requested devices is available")
    return devices


def print_results(results: list[Result]) -> None:
    print()
    print(
        f"{'device':<9} {'mode':<10} {'words':>7} {'tokens':>7} "
        f"{'median ms':>11} {'words/s':>12} {'tokens/s':>12}"
    )
    print("-" * 76)
    for result in results:
        print(
            f"{result.device:<9} {result.mode:<10} {result.model_words:>7} "
            f"{result.transformer_tokens:>7} {result.median_seconds * 1000:>11.2f} "
            f"{result.words_per_second:>12.2f} {result.tokens_per_second:>12.2f}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="Local path or Hugging Face ID for a StreamingSpan checkpoint")
    parser.add_argument("--devices", default="cpu,cuda", help="Comma-separated devices (default: cpu,cuda)")
    parser.add_argument("--lengths", default="16,32,64,128", help="Comma-separated input word counts")
    parser.add_argument("--labels", default=",".join(DEFAULT_LABELS), help="Comma-separated entity labels")
    parser.add_argument("--warmups", type=int, default=1, help="Warmup runs per mode and length")
    parser.add_argument("--repeats", type=int, default=5, help="Measured runs per mode and length")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--json", type=Path, help="Optionally write detailed results as JSON")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    lengths = parse_csv(args.lengths, int)
    labels = parse_csv(args.labels)
    requested_devices = parse_csv(args.devices)
    if any(length < 1 for length in lengths):
        raise SystemExit("All --lengths values must be positive")
    if args.warmups < 0 or args.repeats < 1:
        raise SystemExit("--warmups must be >= 0 and --repeats must be >= 1")
    if not 0.0 <= args.threshold <= 1.0:
        raise SystemExit("--threshold must be between 0 and 1")

    results: list[Result] = []
    for device in resolve_devices(requested_devices):
        print(f"Loading {args.model!r} on {device} ...", flush=True)
        model = GLiNER.from_pretrained(
            args.model,
            load_tokenizer=True,
            local_files_only=args.local_files_only,
            map_location=str(device),
        ).to(device).eval()
        if getattr(model.config, "model_type", None) != "gliner_streaming_span":
            raise SystemExit("The checkpoint must use model_type='gliner_streaming_span'")

        for requested_words in lengths:
            text = make_text(requested_words)
            chunks = word_chunks(model, text)
            token_count = transformer_token_count(model, text)
            print(
                f"Benchmarking {device}: requested={requested_words}, "
                f"model_words={len(chunks)}, transformer_tokens={token_count}",
                flush=True,
            )
            for mode in ("normal", "streaming"):
                samples = measure(
                    model, device, mode, text, chunks, labels, args.threshold, args.warmups, args.repeats
                )
                median = statistics.median(samples)
                results.append(
                    Result(
                        device=str(device),
                        mode=mode,
                        requested_words=requested_words,
                        model_words=len(chunks),
                        transformer_tokens=token_count,
                        median_seconds=median,
                        mean_seconds=statistics.mean(samples),
                        stdev_seconds=statistics.stdev(samples) if len(samples) > 1 else 0.0,
                        words_per_second=len(chunks) / median,
                        tokens_per_second=token_count / median,
                        repeats=len(samples),
                    )
                )

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print_results(results)
    if args.json:
        args.json.write_text(json.dumps([asdict(result) for result in results], indent=2) + "\n")
        print(f"\nWrote {args.json}")


if __name__ == "__main__":
    main()
