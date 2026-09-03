#!/usr/bin/env python3
"""Compare stateless, looped, fixed-batch, and async StreamingSpan inference."""

from __future__ import annotations

import time
import asyncio
import argparse
import statistics
from uuid import uuid4

import torch
from bench_streaming import DEFAULT_LABELS, make_text, synchronize, word_chunks

from gliner import GLiNER


def _session_ids(prefix: str, count: int) -> list[str]:
    run_id = uuid4().hex
    return [f"{prefix}-{run_id}-{index}" for index in range(count)]


def run_stateless(model, text, labels, sessions, threshold):
    model.inference([text] * sessions, labels, batch_size=sessions, threshold=threshold)


def run_loop(model, chunks, labels, sessions, threshold):
    session_ids = _session_ids("loop", sessions)
    try:
        for chunk in chunks:
            for session_id in session_ids:
                model.inference(
                    [chunk],
                    labels,
                    session_id=[session_id],
                    threshold=threshold,
                )
    finally:
        model.clear_session(session_ids)


def run_fixed_batch(model, chunks, labels, sessions, threshold):
    with model.create_streaming_batch(
        _session_ids("fixed", sessions),
        labels,
    ) as stream:
        for chunk in chunks:
            stream.append([chunk] * sessions, threshold=threshold)


async def _run_async(model, chunks, labels, sessions, threshold):
    session_ids = _session_ids("async", sessions)
    async with model.create_async_streaming_engine(
        max_batch_size=sessions,
        batch_wait_timeout_ms=0,
    ) as engine:
        for chunk in chunks:
            await asyncio.gather(
                *(
                    engine.append(
                        session_id,
                        chunk,
                        labels,
                        threshold=threshold,
                    )
                    for session_id in session_ids
                )
            )
    model.clear_session(session_ids)


def run_async(model, chunks, labels, sessions, threshold):
    asyncio.run(_run_async(model, chunks, labels, sessions, threshold))


def run_mode(model, mode, text, chunks, labels, sessions, threshold):
    if mode == "stateless":
        run_stateless(model, text, labels, sessions, threshold)
    elif mode == "loop":
        run_loop(model, chunks, labels, sessions, threshold)
    elif mode == "fixed_batch":
        run_fixed_batch(model, chunks, labels, sessions, threshold)
    elif mode == "async":
        run_async(model, chunks, labels, sessions, threshold)
    else:
        raise ValueError(f"Unknown benchmark mode: {mode}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--sessions", type=int, default=8)
    parser.add_argument("--words", type=int, default=32)
    parser.add_argument("--labels", default=",".join(DEFAULT_LABELS))
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    if args.sessions < 1 or args.words < 1 or args.repeats < 1 or args.warmups < 0:
        parser.error("sessions, words, and repeats must be positive; warmups must be non-negative")
    labels = [label.strip() for label in args.labels.split(",") if label.strip()]
    if not labels:
        parser.error("at least one label is required")

    device = torch.device(args.device)
    model = GLiNER.from_pretrained(
        args.model,
        local_files_only=args.local_files_only,
        map_location=str(device),
    ).to(device).eval()
    if getattr(model.config, "model_type", None) != "gliner_streaming_span":
        parser.error("the checkpoint must use model_type='gliner_streaming_span'")

    text = make_text(args.words)
    chunks = word_chunks(model, text)
    modes = ("stateless", "loop", "fixed_batch", "async")
    print(f"device={device} sessions={args.sessions} words/session={len(chunks)}")  # noqa: T201
    print(f"{'mode':<14} {'median ms':>12} {'steps/s':>12} {'words/s':>12}")  # noqa: T201
    for mode in modes:
        for _ in range(args.warmups):
            run_mode(model, mode, text, chunks, labels, args.sessions, args.threshold)
        synchronize(device)
        samples = []
        for _ in range(args.repeats):
            synchronize(device)
            started = time.perf_counter()
            run_mode(model, mode, text, chunks, labels, args.sessions, args.threshold)
            synchronize(device)
            samples.append(time.perf_counter() - started)
        median = statistics.median(samples)
        total_steps = args.sessions if mode == "stateless" else args.sessions * len(chunks)
        print(  # noqa: T201
            f"{mode:<14} {median * 1000:>12.2f} "
            f"{total_steps / median:>12.2f} "
            f"{args.sessions * len(chunks) / median:>12.2f}"
        )


if __name__ == "__main__":
    main()
