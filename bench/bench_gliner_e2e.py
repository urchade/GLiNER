#!/usr/bin/env python
"""Compare GLiNER NER predictions with and without inference packing."""

from __future__ import annotations

import argparse
import pathlib
import sys
import textwrap
import time
from typing import List, Sequence
import traceback

import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gliner import GLiNER, InferencePackingConfig


DEFAULT_TEXTS: List[str] = [
    "OpenAI launched GPT-4o in San Francisco, while Sam Altman discussed future plans on CNBC.",
    "NASA announced that the Artemis II mission will send astronauts around the Moon in 2025.",
    "Amazon acquired Whole Foods for $13.7 billion and expanded grocery delivery across the United States.",
    "Taylor Swift kicked off her Eras Tour in Glendale before headlining shows across Europe in 2024.",
    "Elon Musk unveiled Tesla’s Cybertruck at a launch event in Los Angeles in 2019.",
    "Apple introduced the Vision Pro headset at WWDC 2023 in Cupertino.",
    "Google I/O 2022 showcased advances in AI, including new features for Google Translate.",
    "Lionel Messi signed with Inter Miami in 2023, changing the face of Major League Soccer in the United States.",
    "The FIFA World Cup 2022 was held in Qatar, attracting millions of fans worldwide.",
    "Meta rebranded from Facebook in October 2021 to emphasize its focus on the Metaverse.",
    "Microsoft acquired Activision Blizzard in 2022 for nearly $69 billion, one of the largest deals in gaming history.",
    "Serena Williams announced her retirement after the 2022 U.S. Open in New York.",
    "The Paris Climate Agreement was signed in 2016, bringing together nations to fight global warming.",
    "Pfizer and BioNTech developed the first widely distributed COVID-19 vaccine in 2020.",
    "The 2008 financial crisis was triggered by the collapse of Lehman Brothers on Wall Street.",
    "Barack Obama was elected President of the United States in 2008, becoming the first African-American president.",
    "India landed Chandrayaan-3 on the lunar south pole in 2023, marking a historic achievement for ISRO.",
    "The Nobel Peace Prize in 2014 was awarded to Malala Yousafzai for her fight for girls’ education.",
    "Cristiano Ronaldo scored his 800th career goal during a match at Old Trafford in 2021.",
    "The United Nations was founded in San Francisco in 1945 after World War II.",
    "Beyoncé performed at Coachella 2018, an event later dubbed 'Beychella' by fans.",
    "The Great Recession officially ended in June 2009 after massive stimulus spending.",
    "Alibaba raised $25 billion in its 2014 IPO on the New York Stock Exchange.",
    "Jeff Bezos stepped down as Amazon CEO in July 2021, handing over to Andy Jassy.",
    "The Tokyo 2020 Olympics were postponed to 2021 due to the COVID-19 pandemic.",
    "The Euro 2016 football championship was held across France, with Portugal emerging as the winner.",
    "Netflix premiered 'Stranger Things' in 2016, a show that became a global pop culture phenomenon.",
    "The iPhone was first introduced by Steve Jobs in January 2007 in San Francisco.",
    "The Berlin Wall fell in 1989, symbolizing the end of the Cold War in Europe.",
    "Queen Elizabeth II passed away in September 2022, ending the longest reign in British history.",
    "Google acquired YouTube in 2006 for $1.65 billion, reshaping the digital media landscape.",
]

long_seq = """In 2023, OpenAI released GPT-4o at a major event in San Francisco, with CEO Sam Altman joining a panel on CNBC to discuss the company’s ambitions for artificial general intelligence. At nearly the same time, Google hosted its I/O conference in Mountain View, unveiling breakthroughs in translation and search while Sundar Pichai emphasized responsible AI. Meanwhile, Microsoft completed its $69 billion acquisition of Activision Blizzard, reshaping the gaming industry and prompting regulators in Brussels and Washington, D.C. to raise antitrust concerns.  

Elsewhere, NASA announced that the Artemis II mission, scheduled for 2025, would send astronauts around the Moon for the first time in decades, while SpaceX prepared a Starship launch from Boca Chica, Texas. In Europe, the European Union finalized a sweeping AI Act in Brussels in 2024, hailed as the most comprehensive technology regulation since GDPR in 2018. At the same time, the United Nations hosted the COP28 climate summit in Dubai, where leaders including Emmanuel Macron, Narendra Modi, and Joe Biden pledged trillions of dollars in green investment.  

In sports, Lionel Messi shocked the world by signing with Inter Miami in 2023 after leaving Paris Saint-Germain, while Cristiano Ronaldo continued his career with Al-Nassr in Saudi Arabia. The 2022 FIFA World Cup in Qatar had already set records for attendance and sponsorship revenue, with Adidas and Coca-Cola reporting billions in sales tied to the event. Meanwhile, the International Olympic Committee prepared for the Paris 2024 Summer Games, investing heavily in infrastructure projects across France.  

On the cultural front, Taylor Swift’s Eras Tour began in Glendale in 2023 before expanding across Europe in 2024, generating over a billion dollars in ticket sales and boosting local economies from London to Berlin. Netflix, still riding the success of “Stranger Things” and “The Crown,” announced partnerships with South Korean studios in Seoul, investing $2.5 billion in new dramas by 2027. In Hollywood, the 2022 Academy Awards saw “CODA” win Best Picture, while the 2023 ceremony honored “Everything Everywhere All at Once,” with Michelle Yeoh becoming the first Asian woman to win Best Actress.  

Meanwhile, in global finance, Bitcoin surged to an all-time high of $69,000 in November 2021 before crashing below $20,000 in 2022, causing turmoil for crypto exchanges like FTX, which filed for bankruptcy in Delaware after revelations about Sam Bankman-Fried’s empire. By 2024, BlackRock and Fidelity were filing ETF applications with the U.S. Securities and Exchange Commission, betting on mainstream adoption. In Asia, Alibaba and Tencent continued to expand their digital payment systems, while India’s UPI network processed more than 10 billion transactions in a single month.  

Finally, political headlines dominated the world stage. In September 2022, Queen Elizabeth II passed away in Balmoral, prompting a global outpouring of grief and the accession of King Charles III. In the United States, the 2024 presidential race heated up with debates in Washington and rallies in key swing states like Pennsylvania, Michigan, and Arizona. Meanwhile, leaders gathered at the G20 Summit in New Delhi in 2023, where Prime Minister Narendra Modi welcomed counterparts from China, Japan, and the European Union, emphasizing collaboration in technology, trade, and climate resilience. Across all these events, the interplay of innovation, regulation, culture, and geopolitics underscored how interconnected the twenty-first century has become.
"""
DEFAULT_TEXTS.append(long_seq)

DEFAULT_LABELS: List[str] = [
    "Person",
    "Organization",
    "Location",
    "Event",
    "Date",
    "Money",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a single GLiNER NER batch with and without inference packing.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="urchade/gliner_small-v2.1",
        help="Model name or local path to load with GLiNER.from_pretrained",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run inference on (e.g. 'cpu', 'cuda', 'cuda:0')",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size to use for the GLiNER dataloader",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Confidence threshold passed to GLiNER.run",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="Override maximum packed sequence length (defaults to model.config.max_len)",
    )
    parser.add_argument(
        "--streams-per-batch",
        type=int,
        default=32,
        help="Number of packed streams to generate per batch",
    )
    return parser.parse_args()


def _sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _format_prediction(example_id: int, text: str, entities: Sequence[dict]) -> str:
    header = f"Example {example_id}"
    body_lines = [textwrap.fill(text, width=100, initial_indent="  ", subsequent_indent="  ")]
    if not entities:
        body_lines.append("  (no entities above threshold)")
    else:
        for ent in entities:
            snippet = ent["text"].replace("\n", " ")
            label = ent["label"]
            score = ent.get("score", 0.0)
            start = ent.get("start")
            end = ent.get("end")
            body_lines.append(
                f"  - '{snippet}' ({label}, score={score:.2f}, span={start}:{end})"
            )
    return "\n".join([header] + body_lines)


def _display_predictions(texts: Sequence[str], predictions: Sequence[Sequence[dict]]) -> None:
    print("NER predictions:")
    for idx, (text, ents) in enumerate(zip(texts, predictions), start=1):
        print(_format_prediction(idx, text, ents))
        print()


def _run_once(
    model: GLiNER,
    *,
    texts: Sequence[str],
    labels: Sequence[str],
    batch_size: int,
    threshold: float,
    device: torch.device,
    packing_config: InferencePackingConfig | None,
) -> tuple[List[List[dict]], float]:
    _sync_if_cuda(device)
    start = time.perf_counter()
    predictions = model.run(
        list(texts),
        list(labels),
        batch_size=batch_size,
        threshold=threshold,
        packing_config=packing_config,
    )
    _sync_if_cuda(device)
    elapsed = time.perf_counter() - start
    # The model returns a tuple when num_gen_sequences > 1; enforce list typing for consistency.
    return [list(example) for example in predictions], elapsed


def main() -> None:
    args = _parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.streams_per_batch < 1:
        raise ValueError("--streams-per-batch must be >= 1")

    texts: List[str] = list(DEFAULT_TEXTS)
    labels: List[str] = list(DEFAULT_LABELS)

    device = torch.device(args.device)
    try:
        model = GLiNER.from_pretrained(args.model, map_location=str(device))
    except Exception as exc:  # pragma: no cover - network / I/O failures
        print(traceback.format_exc())
        raise SystemExit(
            "Failed to load GLiNER model. Use --model with a local path or ensure network access."
        ) from exc
    model.to(device)
    model.eval()

    tokenizer = model.data_processor.transformer_tokenizer
    sep_token_id = getattr(tokenizer, "sep_token_id", None)
    if sep_token_id is None:
        sep_token_id = getattr(tokenizer, "eos_token_id", None)
    if sep_token_id is None:
        sep_token_id = getattr(tokenizer, "pad_token_id", None)

    max_length = args.max_length or getattr(model.config, "max_len", None)
    if max_length is None:
        raise ValueError("Unable to infer max sequence length for packing; please pass --max-length")

    packing_config = InferencePackingConfig(
        max_length=int(max_length),
        sep_token_id=sep_token_id,
        streams_per_batch=args.streams_per_batch,
    )

    print("Running baseline (no packing)...")
    baseline_preds, baseline_time = _run_once(
        model,
        texts=texts,
        labels=labels,
        batch_size=args.batch_size,
        threshold=args.threshold,
        device=device,
        packing_config=None,
    )

    print("Running with inference packing...")
    packed_preds, packed_time = _run_once(
        model,
        texts=texts,
        labels=labels,
        batch_size=args.batch_size,
        threshold=args.threshold,
        device=device,
        packing_config=packing_config,
    )
    # _display_predictions(texts, packed_preds)

    identical = baseline_preds == packed_preds

    print("Timing summary:")
    print(f"  Without packing: {baseline_time:.3f} s")
    print(f"  With packing   : {packed_time:.3f} s")
    if packed_time > 0:
        print(f"  Speedup        : {baseline_time / packed_time:.2f}x")
    print(f"Predictions identical: {identical}")

    # if not identical:
    #     print("WARNING: Predictions differ between baseline and packed runs!")
    #     print(baseline_preds)
    #     print("===================="*10)
    #     print(packed_preds)


if __name__ == "__main__":
    main()