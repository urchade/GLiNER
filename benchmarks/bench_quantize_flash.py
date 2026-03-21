#!/usr/bin/env python3
"""Benchmark GLiNER inference: baseline vs quantize vs compiled+quantized vs compiled+flashdeberta.

Compares:
  1. Baseline (fp32)
  2. Quantized (fp16)
  3. Compiled + Quantized
  4. Compiled + FlashDeBERTa

Each configuration is tested at sequence lengths 128 and 1024.
"""

import os
import sys
import time
import pathlib

import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gliner import GLiNER

MODEL_NAME = "gliner-community/gliner_small-v2.5"
DEVICE = "cuda"
LABELS = [
    "Person", "Organization", "Location", "Event",
    "Date", "Money", "Product", "Technology",
]
THRESHOLD = 0.5
WARMUP_RUNS = 3
TIMED_RUNS = 10

# Short text (~128 tokens after prompt+labels)
SHORT_TEXT = (
    "OpenAI launched GPT-4o in San Francisco, while Sam Altman discussed "
    "future plans on CNBC. NASA announced that the Artemis II mission will "
    "send astronauts around the Moon in 2025."
)

# Long text (~1024 tokens after prompt+labels)
LONG_TEXT = (
    "In 2023, OpenAI released GPT-4o at a major event in San Francisco, with CEO Sam Altman "
    "joining a panel on CNBC to discuss the company's ambitions for artificial general intelligence. "
    "At nearly the same time, Google hosted its I/O conference in Mountain View, unveiling "
    "breakthroughs in translation and search while Sundar Pichai emphasized responsible AI. "
    "Meanwhile, Microsoft completed its $69 billion acquisition of Activision Blizzard, reshaping "
    "the gaming industry and prompting regulators in Brussels and Washington, D.C. to raise "
    "antitrust concerns.\n\n"
    "Elsewhere, NASA announced that the Artemis II mission, scheduled for 2025, would send "
    "astronauts around the Moon for the first time in decades, while SpaceX prepared a Starship "
    "launch from Boca Chica, Texas. In Europe, the European Union finalized a sweeping AI Act "
    "in Brussels in 2024, hailed as the most comprehensive technology regulation since GDPR in "
    "2018. At the same time, the United Nations hosted the COP28 climate summit in Dubai, where "
    "leaders including Emmanuel Macron, Narendra Modi, and Joe Biden pledged trillions of dollars "
    "in green investment.\n\n"
    "In sports, Lionel Messi shocked the world by signing with Inter Miami in 2023 after leaving "
    "Paris Saint-Germain, while Cristiano Ronaldo continued his career with Al-Nassr in Saudi "
    "Arabia. The 2022 FIFA World Cup in Qatar had already set records for attendance and "
    "sponsorship revenue, with Adidas and Coca-Cola reporting billions in sales tied to the event. "
    "Meanwhile, the International Olympic Committee prepared for the Paris 2024 Summer Games, "
    "investing heavily in infrastructure projects across France.\n\n"
    "On the cultural front, Taylor Swift's Eras Tour began in Glendale in 2023 before expanding "
    "across Europe in 2024, generating over a billion dollars in ticket sales and boosting local "
    "economies from London to Berlin. Netflix, still riding the success of Stranger Things and "
    "The Crown, announced partnerships with South Korean studios in Seoul, investing $2.5 billion "
    "in new dramas by 2027. In Hollywood, the 2022 Academy Awards saw CODA win Best Picture, "
    "while the 2023 ceremony honored Everything Everywhere All at Once, with Michelle Yeoh "
    "becoming the first Asian woman to win Best Actress.\n\n"
    "Meanwhile, in global finance, Bitcoin surged to an all-time high of $69,000 in November 2021 "
    "before crashing below $20,000 in 2022, causing turmoil for crypto exchanges like FTX, which "
    "filed for bankruptcy in Delaware after revelations about Sam Bankman-Fried's empire. By 2024, "
    "BlackRock and Fidelity were filing ETF applications with the U.S. Securities and Exchange "
    "Commission, betting on mainstream adoption. In Asia, Alibaba and Tencent continued to expand "
    "their digital payment systems, while India's UPI network processed more than 10 billion "
    "transactions in a single month."
)*2

TINY_TEXT = "Elon Musk founded SpaceX in Los Angeles."

TEXTS_BY_SEQ_LEN = {
    8: TINY_TEXT,
    128: SHORT_TEXT,
    1024: LONG_TEXT,
}


def sync_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def bench_inference(model, text, n_warmup=WARMUP_RUNS, n_timed=TIMED_RUNS):
    """Run warmup + timed inference, return average time in ms."""
    for _ in range(n_warmup):
        model.predict_entities(text, LABELS, threshold=THRESHOLD)
        sync_cuda()

    times = []
    for _ in range(n_timed):
        sync_cuda()
        t0 = time.perf_counter()
        model.predict_entities(text, LABELS, threshold=THRESHOLD)
        sync_cuda()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    return sum(times) / len(times)


def load_model(quantize=False, compile_model=False, flash_deberta=False):
    """Load a fresh model with the given settings."""
    # Flash deberta is toggled via env var before model load
    if flash_deberta:
        os.environ["USE_FLASHDEBERTA"] = "1"
    else:
        os.environ.pop("USE_FLASHDEBERTA", None)

    model = GLiNER.from_pretrained(MODEL_NAME, map_location=DEVICE)

    if quantize:
        model.quantize()
    if compile_model:
        model.compile()
    
    if flash_deberta:
        model.half()

    model.eval()
    return model


#                       name                          quantize  compile  flash
CONFIGS = [
    ("Baseline (fp32)",                                False,    False,   False),
    ("Quantized (fp16)",                               True,     False,   False),
    ("Compiled + Quantized",                           True,     True,    False),
    ("FlashDeBERTa",                                   False,    False,   True),
    ("Compiled + FlashDeBERTa",                        False,    True,    True),
]

SEQ_LENGTHS = [8, 128, 1024]


def main():
    if not torch.cuda.is_available():
        print("ERROR: CUDA is required for this benchmark.")
        sys.exit(1)

    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Model: {MODEL_NAME}")
    print(f"Warmup runs: {WARMUP_RUNS}, Timed runs: {TIMED_RUNS}")
    print()

    # results[config_name][seq_len] = avg_ms
    results = {}

    for config_name, quantize, compile_model, flash in CONFIGS:
        print(f"=== {config_name} ===")
        results[config_name] = {}

        try:
            model = load_model(
                quantize=quantize,
                compile_model=compile_model,
                flash_deberta=flash,
            )
        except Exception as e:
            print(f"  SKIPPED: {e}\n")
            for sl in SEQ_LENGTHS:
                results[config_name][sl] = None
            continue

        for sl in SEQ_LENGTHS:
            text = TEXTS_BY_SEQ_LEN[sl]
            avg_ms = bench_inference(model, text)
            results[config_name][sl] = avg_ms
            print(f"  seq_len={sl:>4d}: {avg_ms:>8.1f} ms/inference")

        # Free GPU memory before loading next config
        del model
        torch.cuda.empty_cache()
        print()

    # Summary table
    baseline_name = CONFIGS[0][0]
    hdr_times = "".join(f" {'seq=' + str(sl):>12}" for sl in SEQ_LENGTHS)
    hdr_speedups = "".join(f" {'spd@' + str(sl):>10}" for sl in SEQ_LENGTHS)
    width = 32 + len(hdr_times) + len(hdr_speedups)
    print("=" * width)
    print(f"{'Configuration':<32}{hdr_times}{hdr_speedups}")
    print("-" * width)

    for config_name, *_ in CONFIGS:
        row = f"{config_name:<32}"
        for sl in SEQ_LENGTHS:
            ms = results[config_name].get(sl)
            if ms is not None:
                row += f" {ms:>10.1f}ms"
            else:
                row += f" {'N/A':>11s}"

        for sl in SEQ_LENGTHS:
            base_ms = results[baseline_name].get(sl)
            cur_ms = results[config_name].get(sl)
            if base_ms and cur_ms:
                row += f" {base_ms / cur_ms:>8.2f}x"
            else:
                row += f" {'N/A':>9s}"

        print(row)

    print("=" * 78)


if __name__ == "__main__":
    main()
