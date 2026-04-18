"""Evaluate raw vs compressed-prompt GLiNER on knowledgator/biomed_NER."""

import argparse
import random
import time

import torch
from datasets import load_dataset

from gliner import GLiNER


def predictions_to_ner(text, preds):
    """Map char-offset predictions from model.inference to word-level ner tuples."""
    ent_dicts = [{"start": p["start"], "end": p["end"], "class": p["label"]} for p in preds]
    return char_to_word_sample(text, ent_dicts)


def distill_finetune(model, distill_data, *, epochs, lr, batch_size, output_dir):
    """Fine-tune `model` on pseudo-labeled `distill_data` via GLiNER.train_model."""
    # Attach the full label set so the collator uses it with prepare_labels=True.
    model.train_model(
        train_dataset=distill_data,
        eval_dataset=None,
        output_dir=output_dir,
        num_train_epochs=epochs,
        max_steps=-1,  # override create_training_args' default (10000) so num_train_epochs wins
        per_device_train_batch_size=batch_size,
        learning_rate=lr,
        save_strategy="no",
        report_to="none",
        logging_steps=10,
        remove_unused_columns=False,
    )
    model.eval()


def timed_evaluate(model, eval_data, *, warmup, repeats, device, **eval_kwargs):
    """Run model.evaluate once for metrics and `repeats` times for timing."""
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    out, f1 = model.evaluate(eval_data, **eval_kwargs)

    for _ in range(warmup):
        model.evaluate(eval_data, **eval_kwargs)

    if device.startswith("cuda"):
        torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        model.evaluate(eval_data, **eval_kwargs)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    mean = sum(times) / len(times)
    return out, f1, mean, min(times)


def char_to_word_sample(text, entities):
    """Convert {text, entities:[{class,start,end}]} to {tokenized_text, ner}.

    Uses whitespace tokenization and aligns char offsets to word indices.
    Entities that don't align to word boundaries are dropped.
    """
    words = text.split()
    # Build char-start index for each word (assuming single-space separation of split()).
    char_starts, char_ends = [], []
    cursor = 0
    remaining = text
    for w in words:
        idx = remaining.find(w)
        abs_start = cursor + idx
        char_starts.append(abs_start)
        char_ends.append(abs_start + len(w))
        cursor = abs_start + len(w)
        remaining = text[cursor:]

    start_to_widx = {s: i for i, s in enumerate(char_starts)}
    end_to_widx = {e: i for i, e in enumerate(char_ends)}

    ner = []
    for ent in entities:
        s, e, cls = ent["start"], ent["end"], ent["class"].lower()
        # Tolerate leading/trailing whitespace inside span
        span_text = text[s:e]
        ls = len(span_text) - len(span_text.lstrip())
        le = len(span_text) - len(span_text.rstrip())
        s2, e2 = s + ls, e - le
        if s2 in start_to_widx and e2 in end_to_widx:
            ner.append((start_to_widx[s2], end_to_widx[e2], cls))
    return {"tokenized_text": words, "ner": ner}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gliner-community/gliner_small-v2.5")
    parser.add_argument("--dataset", default="knowledgator/biomed_NER")
    parser.add_argument("--split", default="train")
    parser.add_argument("--eval_size", type=int, default=3000)
    parser.add_argument("--compress_size", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bench_warmup", type=int, default=1)
    parser.add_argument("--bench_repeats", type=int, default=1)
    parser.add_argument("--distill", action="store_true",
                        help="Fine-tune the compressed model on raw-model pseudo-labels.")
    parser.add_argument("--distill_size", type=int, default=1000,
                        help="Number of texts to use for distillation (drawn after compress slice).")
    parser.add_argument("--distill_epochs", type=int, default=3)
    parser.add_argument("--distill_lr", type=float, default=1e-5)
    parser.add_argument("--distill_threshold", type=float, default=0.3)
    parser.add_argument("--distill_output_dir", type=str, default="./distill_ckpt")
    args = parser.parse_args()

    random.seed(args.seed)

    print(f"Loading dataset {args.dataset} [{args.split}]...")
    ds = load_dataset(args.dataset, split=args.split)

    processed = [char_to_word_sample(r["text"], r["entities"]) for r in ds]
    processed = [p for p in processed if p["ner"]]  # drop empties

    labels = sorted({t for p in processed for _, _, t in p["ner"]})
    print(f"{len(processed)} samples, {len(labels)} labels: {labels}")

    random.shuffle(processed)
    # Pin the full label set on every sample so raw and compressed evaluations
    # share an identical label space. Without this, raw eval would derive
    # labels per-sample (only the positives present) and be unfairly easier
    # than the compressed path, which always classifies over all labels.
    for p in processed:
        p["ner_labels"] = labels
    eval_data = processed[: args.eval_size]
    compress_slice = processed[args.eval_size : args.eval_size + args.compress_size]
    if not compress_slice:
        compress_slice = processed[: args.compress_size]
    compress_texts = [" ".join(p["tokenized_text"]) for p in compress_slice]

    distill_start = args.eval_size + args.compress_size
    distill_slice = processed[distill_start : distill_start + args.distill_size] if args.distill else []

    print(f"Loading model {args.model}...")
    model = GLiNER.from_pretrained(args.model).to(args.device)

    eval_kwargs = dict(flat_ner=True, threshold=args.threshold, batch_size=args.batch_size)
    n = len(eval_data)

    print("=== Raw GLiNER evaluation ===")
    raw_out, raw_f1, raw_mean, raw_best = timed_evaluate(
        model, eval_data, warmup=args.bench_warmup, repeats=args.bench_repeats,
        device=args.device, **eval_kwargs,
    )
    print(raw_out)
    print(f"Raw F1: {raw_f1:.4f}")
    print(f"Raw timing (n={n}, bs={args.batch_size}, repeats={args.bench_repeats}): "
          f"mean {raw_mean:.3f}s | best {raw_best:.3f}s | "
          f"{n / raw_mean:.1f} samples/s")

    distill_data = None
    if args.distill and distill_slice:
        print(f"Generating pseudo-labels from raw model on {len(distill_slice)} distillation texts...")
        distill_texts = [" ".join(p["tokenized_text"]) for p in distill_slice]
        preds = model.inference(
            distill_texts, labels, flat_ner=True,
            threshold=args.distill_threshold, batch_size=args.batch_size,
        )
        distill_data = [predictions_to_ner(t, p) for t, p in zip(distill_texts, preds)]
        kept = sum(1 for d in distill_data if d["ner"])
        print(f"  {kept}/{len(distill_data)} samples carry at least one pseudo-label")

    print(f"Compressing prompt embeddings over {len(compress_texts)} texts...")
    model.compress_prompt_embeddings(
        texts=compress_texts, labels=labels, batch_size=args.batch_size
    )
    model.config.precomputed_prompts_mode = True

    if distill_data:
        print(f"Fine-tuning compressed model on pseudo-labels "
              f"(epochs={args.distill_epochs}, lr={args.distill_lr})...")
        distill_finetune(
            model, distill_data,
            epochs=args.distill_epochs, lr=args.distill_lr,
            batch_size=args.batch_size, output_dir=args.distill_output_dir,
        )

    print("=== Compressed GLiNER evaluation ===")
    comp_out, comp_f1, comp_mean, comp_best = timed_evaluate(
        model, eval_data, warmup=args.bench_warmup, repeats=args.bench_repeats,
        device=args.device, **eval_kwargs,
    )
    print(comp_out)
    print(f"Compressed F1: {comp_f1:.4f}")
    print(f"Compressed timing (n={n}, bs={args.batch_size}, repeats={args.bench_repeats}): "
          f"mean {comp_mean:.3f}s | best {comp_best:.3f}s | "
          f"{n / comp_mean:.1f} samples/s")

    print("\n=== Summary ===")
    print(f"Raw        F1: {raw_f1:.4f}  | mean {raw_mean:.3f}s | {n / raw_mean:.1f} samples/s")
    print(f"Compressed F1: {comp_f1:.4f}  | mean {comp_mean:.3f}s | {n / comp_mean:.1f} samples/s")
    print(f"Delta F1     : {comp_f1 - raw_f1:+.4f}")
    print(f"Speedup      : {raw_mean / comp_mean:.2f}x")


if __name__ == "__main__":
    main()
