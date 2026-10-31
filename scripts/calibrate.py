"""Calibrate a trained GLiNER model's entity probabilities on a labelled dataset.

The backbone is frozen. Two methods:

* ``--method temperature`` (default): fit one temperature ``T`` (and optionally a
  bias ``b``) so that ``sigmoid(logit / T + b)`` matches observed frequencies.
  The values go to the config (``logit_temperature`` / ``logit_bias``).
* ``--method head``: fit a calibration head (as in GLiClass) that predicts an
  inverse temperature ``beta`` and bias ``b`` for every candidate (span, label)
  pair from the span / label representations and score statistics:
  ``sigmoid(beta * logit + b)``. It starts from, and is regularised towards, a
  global Platt / temperature fit, and is early-stopped on a held-out part of
  ``--data``. The head is saved with the model and applied on every inference.

Both report entry-level metrics and the reliability of decoded spans (of the
spans with score in a bin, what share is correct) on ``--eval-data`` (or the
held-out part of ``--data``).

Dataset format (GLiNER JSON list, or JSONL with one example per line):

    [{"tokenized_text": ["Barack", "Obama", "visited", "Paris", "."],
      "ner": [[0, 1, "person"], [3, 3, "location"]]}, ...]

Examples:

    # Fit temperature on a dev set, compare before/after, save a calibrated copy
    python scripts/calibrate.py --model path/to/model --data dev.json --output-dir path/to/model-calibrated

    # Also fit a bias (shifts which spans pass a fixed threshold) and only report
    python scripts/calibrate.py --model urchade/gliner_small-v2.1 --data dev.json --fit-bias --no-save

    # Calibration head, evaluated on a separate test set
    python scripts/calibrate.py --model path/to/model --data dev.json --eval-data test.json \
        --method head --output-dir path/to/model-calibrated

Use held-out data (not the training set): training logits are overconfident
and would give a temperature that is too low.
"""

from __future__ import annotations

import sys
import json
import random
import logging
import argparse
from pathlib import Path

import torch

from gliner import GLiNER
from gliner.calibration import (
    freeze_model,
    balance_table,
    fit_calibrator,
    span_reliability,
    assign_label_sets,
    candidate_entries,
    calibration_metrics,
    dataset_entity_types,
    collect_entity_logits,
    collect_head_examples,
    fit_logit_calibration,
    get_logit_calibration,
    set_logit_calibration,
    evaluate_head_candidates,
)

AUTO_DATASET_LABELS_LIMIT = 50


def load_dataset(path: str) -> list:
    """Load a GLiNER-format dataset from JSON or JSONL."""
    with open(path, encoding="utf-8") as f:
        if path.endswith(".jsonl"):
            data = [json.loads(line) for line in f if line.strip()]
        else:
            data = json.load(f)
    data = [ex for ex in data if ex.get("tokenized_text")]
    for ex in data:
        ex.setdefault("ner", [])
    return data


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="Path or Hub id of a trained GLiNER model.")
    parser.add_argument("--data", required=True, help="GLiNER-format calibration dataset (.json or .jsonl).")
    parser.add_argument("--output-dir", help="Where to save the calibrated model (weights unchanged, config updated).")
    parser.add_argument("--no-save", action="store_true", help="Only fit and report; do not save.")
    parser.add_argument("--fit-bias", action="store_true", help="Fit a bias b as well as the temperature T.")
    parser.add_argument(
        "--label-mode",
        choices=["auto", "dataset", "example"],
        default="auto",
        help=(
            "'dataset': give every example the full label set of the dataset (mirrors inference). "
            "'example': training-style labels (gold types + sampled negatives; honours ner_labels / ner_negatives). "
            f"'auto': 'dataset' if it has <= {AUTO_DATASET_LABELS_LIMIT} labels, else 'example'."
        ),
    )
    parser.add_argument(
        "--labels", nargs="+", help="Explicit label set for every example (implies --label-mode dataset)."
    )
    parser.add_argument(
        "--labels-per-pass", type=int, help="Split the label set into chunks of this size per forward pass."
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-examples", type=int, help="Use at most this many (randomly chosen) examples.")
    parser.add_argument(
        "--neg-keep-prob",
        type=float,
        default=1.0,
        help="Keep each negative (span, class) entry with this probability to save memory; kept ones are reweighted.",
    )
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for the reported precision/recall/F1.")
    parser.add_argument("--n-bins", type=int, default=15, help="Bins for expected calibration error.")
    parser.add_argument(
        "--eval-data", help="Optional separate GLiNER-format set for the final report (decoded-span reliability)."
    )
    parser.add_argument(
        "--span-eval",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Report decoded-span reliability (runs inference once per variant).",
    )
    parser.add_argument("--span-eval-min-score", type=float, default=0.05, help="Lowest decoded score evaluated.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    head = parser.add_argument_group("calibration head (--method head)")
    parser.add_argument("--method", choices=["temperature", "head"], default="temperature")
    head.add_argument("--eval-ratio", type=float, default=0.2, help="Share of --data held out for early stopping.")
    head.add_argument("--candidate-threshold", type=float, default=0.01, help="Raw probability for a candidate pair.")
    head.add_argument("--hidden-size", type=int, default=256)
    head.add_argument("--proj-size", type=int, default=128)
    head.add_argument("--beta-max", type=float, default=5.0, help="Upper bound of the inverse temperature.")
    head.add_argument("--dropout", type=float, default=0.1)
    head.add_argument(
        "--use-bias",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also predict a per-pair bias (fixes base-rate miscalibration; may flip decisions at 0.5).",
    )
    head.add_argument("--prior-weight", type=float, default=0.01, help="Weight of (log beta - log beta_0)^2.")
    head.add_argument("--bias-prior-weight", type=float, default=0.01, help="Weight of (b - b_0)^2.")
    head.add_argument(
        "--balance",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Reweight candidates so every (raw confidence bin, correct / wrong) cell has equal mass.",
    )
    head.add_argument("--balance-bins", type=int, default=5)
    head.add_argument("--balance-max-weight", type=float, default=10.0)
    head.add_argument("--num-epochs", type=int, default=20)
    head.add_argument("--head-batch-size", type=int, default=16, help="Examples per head training step.")
    head.add_argument("--lr", type=float, default=1e-3)
    head.add_argument("--weight-decay", type=float, default=0.01)
    head.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args(argv)
    if not args.no_save and not args.output_dir:
        parser.error("--output-dir is required unless --no-save is given")
    return args


def format_metrics(name: str, m: dict) -> str:
    return (
        f"{name:<12} NLL={m['nll']:.5f}  Brier={m['brier']:.5f}  ECE={m['ece']:.5f}  "
        f"ECE(p>=0.1)={m['ece_confident']:.4f}  P={m['precision']:.4f}  R={m['recall']:.4f}  F1={m['f1']:.4f}"
    )


def main(argv=None) -> int:
    args = parse_args(argv)
    sys.stdout.reconfigure(line_buffering=True)
    logging.basicConfig(format="%(message)s")
    logging.getLogger("gliner.calibration").setLevel(logging.INFO)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    data = load_dataset(args.data)
    if args.max_examples and len(data) > args.max_examples:
        data = random.sample(data, args.max_examples)
    if not data:
        print("No usable examples in dataset.", file=sys.stderr)
        return 1

    print(f"Loading model: {args.model}")
    model = GLiNER.from_pretrained(args.model, local_files_only=args.local_files_only)
    orig_dtype = next(model.parameters()).dtype
    model = model.to(device=args.device, dtype=torch.float32)
    freeze_model(model)
    if any(p.requires_grad for p in model.parameters()):
        raise RuntimeError("Model parameters are not frozen.")

    old_t, old_b = get_logit_calibration(model.config)
    print(f"Current calibration: T={old_t:.4f}, b={old_b:.4f} (fitting starts from raw logits)")

    if args.labels:
        mode, all_labels = "dataset", list(args.labels)
    else:
        all_labels = dataset_entity_types(data)
        mode = args.label_mode
        if mode == "auto":
            mode = "dataset" if len(all_labels) <= AUTO_DATASET_LABELS_LIMIT else "example"
        print(f"Label mode: {mode} ({len(all_labels)} labels in dataset)")

    eval_data = None
    if args.eval_data:
        eval_data = load_dataset(args.eval_data)
        eval_pool = all_labels if args.labels else dataset_entity_types(eval_data)
        eval_data = assign_label_sets(eval_data, mode, eval_pool, seed=args.seed + 1)

    if args.method == "head":
        return run_head(args, model, orig_dtype, data, eval_data, mode, all_labels)

    entity_types = all_labels if mode == "dataset" else None

    print(f"Collecting logits on {len(data)} examples (model frozen)...")
    logits, labels, weights = collect_entity_logits(
        model,
        data,
        batch_size=args.batch_size,
        entity_types=entity_types,
        labels_per_pass=args.labels_per_pass,
        neg_keep_prob=args.neg_keep_prob,
        seed=args.seed,
    )
    num_pos = int((labels > 0.5).sum())
    print(f"Collected {logits.numel()} (span, class) entries, {num_pos} positives")
    if num_pos == 0:
        print("No positive entries found; check the 'ner' annotations and label set.", file=sys.stderr)
        return 1

    temperature, bias = fit_logit_calibration(logits, labels, weights, fit_bias=args.fit_bias, device=args.device)
    print(f"Fitted calibration: T={temperature:.4f}, b={bias:.4f}")

    metric_kwargs = dict(weights=weights, n_bins=args.n_bins, threshold=args.threshold)
    before = calibration_metrics(logits, labels, temperature=1.0, bias=0.0, **metric_kwargs)
    after = calibration_metrics(logits, labels, temperature=temperature, bias=bias, **metric_kwargs)
    print("Entry-level metrics on the calibration set:")
    print(format_metrics("uncalibrated", before))
    print(format_metrics("calibrated", after))
    if (old_t, old_b) != (1.0, 0.0):
        previous = calibration_metrics(logits, labels, temperature=old_t, bias=old_b, **metric_kwargs)
        print(format_metrics("previous", previous))

    report = {
        "model": args.model,
        "data": args.data,
        "method": "temperature",
        "num_examples": len(data),
        "fit_bias": args.fit_bias,
        "logit_temperature": temperature,
        "logit_bias": bias,
        "previous": {"logit_temperature": old_t, "logit_bias": old_b},
        "metrics_uncalibrated": before,
        "metrics_calibrated": after,
    }

    if args.span_eval and eval_data:
        set_logit_calibration(model.config, 1.0, 0.0)
        with model.calibrator_disabled():
            report["span_reliability"] = {"uncalibrated": span_reliability_report(model, eval_data, args)}
            set_logit_calibration(model.config, temperature, bias)
            report["span_reliability"]["calibrated"] = span_reliability_report(model, eval_data, args)
        print_span_reliability(report["span_reliability"])

    if args.no_save:
        print(json.dumps(report, indent=2))
        return 0

    set_logit_calibration(model.config, temperature, bias)
    model = model.to(dtype=orig_dtype)
    output_dir = Path(args.output_dir)
    model.save_pretrained(output_dir)
    with open(output_dir / "calibration_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Saved calibrated model and calibration_report.json to {output_dir}")
    return 0


def span_reliability_report(model, data, args) -> dict:
    return span_reliability(
        model, data, min_score=args.span_eval_min_score, threshold=args.threshold, batch_size=args.batch_size
    )


def print_span_reliability(results: dict) -> None:
    print("Decoded-span reliability (share of spans in each score bin that are correct):")
    names = list(results)
    print("  bin        " + "".join(f"{name:>18}" for name in names))
    for i, row in enumerate(results[names[0]]["bins"]):
        cells = []
        for name in names:
            b = results[name]["bins"][i]
            cells.append(f"{b['accuracy']:6.1%} (n={b['count']:>5})" if b["count"] else f"{'-':>16}")
        print(f"  {row['bin'][0]:.1f}-{row['bin'][1]:.1f}    " + "".join(f"{c:>18}" for c in cells))
    t = next(k for k in results[names[0]] if k.startswith("f1@")).split("@")[1]
    for name in names:
        r = results[name]
        print(
            f"  {name:<15} ECE={r['ece']:.4f}  spans={r['num_spans']}  "
            f"P@{t}={r[f'precision@{t}']:.3f}  R@{t}={r[f'recall@{t}']:.3f}  F1@{t}={r[f'f1@{t}']:.3f}"
        )


def run_head(args, model, orig_dtype, data, eval_data, mode, all_labels) -> int:
    """Fit a calibration head with the backbone frozen."""
    data = assign_label_sets(data, mode, all_labels, seed=args.seed)
    data = list(data)
    random.Random(args.seed).shuffle(data)
    num_held_out = max(1, int(len(data) * args.eval_ratio))
    held_out, train = data[:num_held_out], data[num_held_out:]
    print(f"Calibration examples: {len(train)} train, {len(held_out)} held-out for early stopping")

    if model.calibrator is not None:
        print("Replacing the existing calibration head")
        model.remove_calibrator()
    set_logit_calibration(model.config, 1.0, 0.0)  # the head replaces global scaling

    print("Caching frozen outputs...")
    train_cached = collect_head_examples(model, train, batch_size=args.batch_size)
    held_cached = collect_head_examples(model, held_out, batch_size=args.batch_size)
    if not train_cached:
        print("No usable training examples.", file=sys.stderr)
        return 1

    calibrator = model.add_calibrator(
        word_dim=train_cached[0].words.shape[1],
        label_dim=train_cached[0].prompts.shape[1],
        hidden_size=args.hidden_size,
        proj_size=args.proj_size,
        beta_max=args.beta_max,
        dropout=args.dropout,
        use_bias=args.use_bias,
        candidate_threshold=args.candidate_threshold,
    ).float()

    z, y = candidate_entries(calibrator, train_cached, args.device)
    num_pos = int((y > 0.5).any(-1).sum()) if y.dim() == 2 else int((y > 0.5).sum())
    print(f"Training candidates (raw p >= {args.candidate_threshold}): {z.shape[0]}, positives: {num_pos}")
    if num_pos == 0:
        print("No positive candidates; check annotations / label sets.", file=sys.stderr)
        return 1

    temperature, base_bias = fit_logit_calibration(
        z.reshape(-1), y.reshape(-1).clamp(min=0), fit_bias=args.use_bias, device=args.device
    )
    base_beta = 1.0 / temperature
    print(f"Global {'Platt' if args.use_bias else 'temperature'} scaling: beta_0={base_beta:.4f}, b_0={base_bias:.4f}")

    cell_weights = None
    if args.balance:
        cell_weights = balance_table(z, y, args.balance_bins, args.balance_max_weight)
        print(f"Balancing cell weights: {[round(w, 3) for w in cell_weights.tolist()]}")

    fit_calibrator(
        calibrator,
        train_cached,
        held_cached,
        args.device,
        base_beta=base_beta,
        base_bias=base_bias,
        prior_weight=args.prior_weight,
        bias_prior_weight=args.bias_prior_weight,
        cell_weights=cell_weights,
        num_bins=args.balance_bins,
        num_epochs=args.num_epochs,
        batch_size=args.head_batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        seed=args.seed,
    )

    candidates = evaluate_head_candidates(calibrator, held_cached, args.device, base_beta, base_bias, args.n_bins)
    print("Candidate-level metrics on the held-out part of --data:")
    for name in ("uncalibrated", "global_scaling", "calibrated"):
        if name in candidates:
            print(format_metrics(name[:12], candidates[name]))

    report = {
        "model": args.model,
        "data": args.data,
        "method": "head",
        "num_train": len(train),
        "num_held_out": len(held_out),
        "base_beta": base_beta,
        "base_bias": base_bias,
        "candidate_metrics_held_out": candidates,
        "args": vars(args),
    }

    if args.span_eval:
        target = eval_data if eval_data else held_out
        which = "--eval-data" if eval_data else "held-out part of --data"
        print(f"Evaluating decoded spans on {which} ({len(target)} examples)...")
        results = {}
        with model.calibrator_disabled():
            results["uncalibrated"] = span_reliability_report(model, target, args)
            set_logit_calibration(model.config, temperature, base_bias)
            results["global_scaling"] = span_reliability_report(model, target, args)
            set_logit_calibration(model.config, 1.0, 0.0)
        results["calibrated"] = span_reliability_report(model, target, args)
        report["span_reliability"] = results
        print_span_reliability(results)

    if args.no_save:
        return 0

    model = model.to(dtype=orig_dtype)
    model.calibrator.float()  # keep the small head in full precision
    output_dir = Path(args.output_dir)
    model.save_pretrained(output_dir)
    with open(output_dir / "calibration_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Saved model with calibration head and calibration_report.json to {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
