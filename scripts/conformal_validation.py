"""Empirical validation of ConformalGLiNER's coverage/risk guarantees.

Implements docs/research/eval_plan.md's protocol against real data (not
synthetic): CoNLL-2003 and WNUT-17 via DFKI-SLT/cross_ner (sidesteps
`datasets`'s script-loading rejection, per eval_plan.md §1), using
gliner-community/gliner_small-v2.5.

Scope disclosed up front (docs/research/design.md's "descope, don't fake
rigor" standard applies here too): this run covers in-domain CoNLL-2003,
in-domain WNUT-17, and zero-shot Pair A (CoNLL-2003 -> WNUT-17, the
eval_plan.md-designated headline pair) -- not the full CrossNER 5-domain
sweep or Pairs B/C. Pool sizes are capped (see POOL_CAP below) for CPU
runtime; T defaults to 50 trials (eval_plan.md's own "fast dev" figure,
not the 200-trial final-numbers figure) so this is runnable in one sitting
on a laptop. Both are disclosed in the output results markdown, not hidden.

One forward pass per pooled sentence set; all T-trial resampling happens
on cached scores/tensors afterward (no repeated model calls per trial).

Usage:
    OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE python scripts/conformal_validation.py \
        --output_dir results/conformal
"""

from __future__ import annotations

import os
import json
import time
import random
import argparse
from typing import Dict, List, Tuple, Sequence
from pathlib import Path
from dataclasses import dataclass

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import matplotlib

matplotlib.use("Agg")
import torch
import matplotlib.pyplot as plt
from datasets import load_dataset

from gliner import GLiNER
from gliner.conformal.scores import extract_raw_scores
from gliner.conformal.calibrators import calibration_floor, crc_lambda_search, split_conformal_quantile

MODEL_ID = "gliner-community/gliner_small-v2.5"
ALPHAS = [0.05, 0.10, 0.20]
POOL_CAP = 1200  # sentences per pool -- see module docstring
BATCH_SIZE = 16


def bio_to_spans(tags: List[str]) -> List[Tuple[int, int, str]]:
    """Decode a BIO tag sequence into inclusive-end (start, end, type) triples."""
    spans = []
    start = None
    etype = None
    for i, tag in enumerate([*tags, "O"]):
        if tag.startswith("B-"):
            if start is not None:
                spans.append((start, i - 1, etype))
            start, etype = i, tag[2:]
        elif tag.startswith("I-") and etype == tag[2:]:
            continue
        else:
            if start is not None:
                spans.append((start, i - 1, etype))
            start, etype = None, None
    return spans


def load_examples(dataset_id: str, config: str, split: str, cap: int) -> List[Dict]:
    ds = load_dataset(dataset_id, name=config, split=split) if config else load_dataset(dataset_id, split=split)
    tag_names = ds.features["ner_tags"].feature.names
    out = []
    for row in ds:
        if cap is not None and len(out) >= cap:
            break
        tokens = row["tokens"]
        if not tokens:
            continue
        tags = [tag_names[t] for t in row["ner_tags"]]
        ner = [list(s) for s in bio_to_spans(tags)]
        out.append({"tokenized_text": tokens, "ner": ner})
    return out


@dataclass
class Pool:
    name: str
    examples: List[Dict]
    probs: List[torch.Tensor]  # per-example (L, K, C) sigmoid probs
    id_to_class: List[Dict[int, str]]
    labels: List[str]


def build_pool(model, name: str, examples: List[Dict], labels: Sequence[str], batch_size: int = BATCH_SIZE) -> Pool:
    probs: List[torch.Tensor] = []
    id_to_class: List[Dict[int, str]] = []
    for i in range(0, len(examples), batch_size):
        batch = examples[i : i + batch_size]
        raw = extract_raw_scores(model, batch, labels)
        p = torch.sigmoid(raw.logits)
        for j in range(p.shape[0]):
            probs.append(p[j])
            id_to_class.append(raw.id_to_classes[j])
    return Pool(name=name, examples=examples, probs=probs, id_to_class=id_to_class, labels=list(labels))


def gold_nc_by_example(pool: Pool) -> List[List[Tuple[str, float]]]:
    """Per example, list of (type, nonconformity_score) for gold entities.

    Unrepresentable gold spans (wider than max_width) get score=inf, see
    gliner/conformal/scores.py::align_gold_scores -- reimplemented here
    per-example since Pool caches dense per-example tensors rather than a
    flat batch.
    """
    out = []
    for ex, probs, cls_map in zip(pool.examples, pool.probs, pool.id_to_class):
        class_to_id = {v: k for k, v in cls_map.items()}
        L, K, C = probs.shape
        entry = []
        for start, end, etype in ex.get("ner", []):
            if etype not in class_to_id:
                continue
            width = end - start
            col = class_to_id[etype] - 1
            if not (0 <= start < L) or not (0 <= width < K) or not (0 <= col < C):
                score = float("inf")
            else:
                score = 1.0 - probs[start, width, col].item()
            entry.append((etype, score))
        out.append(entry)
    return out


def trial_metrics(
    calib_pool: Pool,
    test_pool: Pool,
    calib_gold: List[List[Tuple[str, float]]],
    test_gold: List[List[Tuple[str, float]]],
    alpha: float,
    n_calib: int,
    n_trials: int,
    mode: str,
    seed: int,
    pool_and_resplit: bool = False,
) -> Dict:
    """Mirror ConformalGLiNER's own calibrated/uncalibrated split (design.md §5).

    A type only contributes to the headline coverage/efficiency numbers if it met
    the calibration floor in *that trial's* calibration subsample. Types requested
    at test time that never met the floor (e.g. WNUT-only types under Pair A's
    CoNLL-derived calibration) are tracked separately as `uncalibrated_*` -- never
    blended into the guaranteed-looking headline number. This is exactly the
    scenario the zero-shot descope (design.md §0) predicts and this eval is meant
    to demonstrate, not accidentally paper over.

    pool_and_resplit=True implements eval_plan.md §2.2's actual in-domain protocol:
    pool calib_pool+test_pool together and draw a *fresh* random calib/test
    partition every trial, rather than using calib_pool and test_pool as static,
    separately-sourced sets. This matters empirically, not just by-the-book: an
    earlier run of this script found CoNLL-2003's *official* validation and test
    splits are themselves not fully exchangeable for this model (mean
    nonconformity 0.22 on validation vs 0.27 on test -- a real, documented
    property of that benchmark's val/test construction, not a code bug), which
    silently violated split conformal's exchangeability precondition and produced
    a measured ~4-5pp coverage undershoot. Pooling and re-splitting per trial is
    the correct way to test "does split-conformal coverage hold when
    exchangeability genuinely is satisfied" without that confound. Pair A
    (zero-shot) deliberately keeps calib_pool/test_pool separate -- that
    non-exchangeability *is* the experiment there.
    """
    rng = random.Random(seed)

    if pool_and_resplit:
        combined_probs = calib_pool.probs + test_pool.probs
        combined_id_to_class = calib_pool.id_to_class + test_pool.id_to_class
        combined_gold = calib_gold + test_gold
        combined_n = len(combined_probs)
    else:
        calib_n_total = len(calib_pool.examples)
        test_indices_all = list(range(len(test_pool.examples)))

    coverages, effs, raw_counts = [], [], []
    uncal_coverages = []
    per_type_hits: Dict[str, int] = {}
    per_type_n: Dict[str, int] = {}
    n_ok_trials = 0
    floor = calibration_floor(alpha)

    for _trial in range(n_trials):
        if pool_and_resplit:
            shuffled = list(range(combined_n))
            rng.shuffle(shuffled)
            calib_idx = shuffled[: min(n_calib, combined_n)]
            test_idx = shuffled[min(n_calib, combined_n) :]
        else:
            calib_idx = rng.sample(range(calib_n_total), min(n_calib, calib_n_total))
            test_idx = test_indices_all

        calib_types_n: Dict[str, int] = {}
        for i in calib_idx:
            for t, _ in (combined_gold if pool_and_resplit else calib_gold)[i]:
                calib_types_n[t] = calib_types_n.get(t, 0) + 1
        calibrated_types = {t for t, n in calib_types_n.items() if n >= floor}
        if not calibrated_types:
            continue
        calib_gold_source = combined_gold if pool_and_resplit else calib_gold
        pooled_scores = [s for i in calib_idx for (t, s) in calib_gold_source[i] if t in calibrated_types]
        if len(pooled_scores) < floor:
            continue

        if mode == "span_filter":
            try:
                tau = split_conformal_quantile(pooled_scores, alpha)
            except ValueError:
                continue

            def admit(s, tau=tau):
                return s <= tau
        else:  # risk_control
            gold_lists = [[s for (t, s) in calib_gold_source[i] if t in calibrated_types] for i in calib_idx]
            try:
                lam = crc_lambda_search(gold_lists, alpha, verify_monotone=False)
            except ValueError:
                continue

            def admit(s, lam=lam):
                return s <= lam

        n_ok_trials += 1
        hits, ngold = 0, 0
        uncal_hits, uncal_ngold = 0, 0
        eff_sum, raw_sum = 0.0, 0.0
        sentence_losses: List[float] = []  # CRC's own per-sentence loss (theory.md Eq. 4)
        test_gold_source = combined_gold if pool_and_resplit else test_gold
        test_probs_source = combined_probs if pool_and_resplit else test_pool.probs
        test_cls_source = combined_id_to_class if pool_and_resplit else test_pool.id_to_class
        for i in test_idx:
            probs = test_probs_source[i]
            cls_map = test_cls_source[i]
            L, K, C = probs.shape
            sentence_gold = [(t, s) for t, s in test_gold_source[i] if t in calibrated_types]
            sentence_hits = 0
            for etype, s in test_gold_source[i]:
                if etype in calibrated_types:
                    ngold += 1
                    per_type_n[etype] = per_type_n.get(etype, 0) + 1
                    if admit(s):
                        hits += 1
                        sentence_hits += 1
                        per_type_hits[etype] = per_type_hits.get(etype, 0) + 1
                else:
                    # descriptive only, no guarantee -- raw p>0.5 rule, matching
                    # ConformalGLiNER's own out-of-calibration fallback behavior.
                    uncal_ngold += 1
                    if s <= 0.5:
                        uncal_hits += 1
            # CRC's own loss convention (theory.md Eq. 4): 0 for entity-free sentences,
            # avoids a 0/0 and matches exactly what crc_lambda_search calibrated against.
            sentence_losses.append(1.0 - sentence_hits / len(sentence_gold) if sentence_gold else 0.0)
            for col in range(C):
                etype = cls_map.get(col + 1)
                if etype is None or etype not in calibrated_types:
                    continue
                nc = 1.0 - probs[:, :, col]
                thresh = tau if mode == "span_filter" else lam
                eff_sum += (nc <= thresh).sum().item()
                raw_sum += L * K

        if mode == "risk_control":
            # Report the quantity CRC actually calibrates and guarantees: the mean
            # PER-SENTENCE miss rate, not entities pooled flat across sentences.
            # These differ whenever gold-entity count per sentence is uneven (theory.md
            # part ii's "informative m" point) -- pooling flat would silently measure a
            # different, uncalibrated quantity and can show spurious "undercoverage"
            # that has nothing to do with the (valid) CRC guarantee actually being tested.
            coverages.append(1.0 - sum(sentence_losses) / len(sentence_losses) if sentence_losses else float("nan"))
        else:
            coverages.append(hits / ngold if ngold else float("nan"))
        if uncal_ngold:
            uncal_coverages.append(uncal_hits / uncal_ngold)
        effs.append(eff_sum / len(test_idx))
        raw_counts.append(raw_sum / len(test_idx))

    per_type_coverage = {t: per_type_hits.get(t, 0) / n for t, n in per_type_n.items() if n > 0}
    return {
        "alpha": alpha,
        "n_calib": n_calib,
        "n_trials_requested": n_trials,
        "n_trials_ok": n_ok_trials,
        "uncalibrated_coverage_mean": (sum(uncal_coverages) / len(uncal_coverages)) if uncal_coverages else None,
        "n_uncalibrated_trials_with_data": len(uncal_coverages),
        "coverage_mean": sum(coverages) / len(coverages) if coverages else float("nan"),
        "coverage_std": (sum((c - sum(coverages) / len(coverages)) ** 2 for c in coverages) / len(coverages)) ** 0.5
        if coverages
        else float("nan"),
        "efficiency_mean": sum(effs) / len(effs) if effs else float("nan"),
        "raw_candidates_mean": sum(raw_counts) / len(raw_counts) if raw_counts else float("nan"),
        "per_type_coverage": per_type_coverage,
    }


def run_suite(
    name: str, calib_pool: Pool, test_pool: Pool, n_trials: int, n_calib: int, pool_and_resplit: bool = False
) -> List[Dict]:
    calib_gold = gold_nc_by_example(calib_pool)
    test_gold = gold_nc_by_example(test_pool)
    rows = []
    for mode in ("span_filter", "risk_control"):
        for alpha in ALPHAS:
            t0 = time.time()
            m = trial_metrics(
                calib_pool,
                test_pool,
                calib_gold,
                test_gold,
                alpha,
                n_calib,
                n_trials,
                mode,
                seed=1234,
                pool_and_resplit=pool_and_resplit,
            )
            m.update({"pair": name, "mode": mode, "seconds": round(time.time() - t0, 1)})
            rows.append(m)
            uncal = m["uncalibrated_coverage_mean"]
            uncal_str = f", uncalibrated_coverage={uncal:.4f} (no guarantee)" if uncal is not None else ""
            print(
                f"[{name}/{mode}/alpha={alpha}] coverage={m['coverage_mean']:.4f}"
                f"+-{m['coverage_std']:.4f} eff={m['efficiency_mean']:.1f}"
                f" ({m['n_trials_ok']}/{n_trials} trials, {m['seconds']}s){uncal_str}"
            )
    return rows


def calib_size_sensitivity(calib_pool: Pool, test_pool: Pool, alpha: float, n_trials: int) -> List[Dict]:
    calib_gold = gold_nc_by_example(calib_pool)
    test_gold = gold_nc_by_example(test_pool)
    rows = []
    for n_calib in [50, 100, 200, 500, 1000]:
        if n_calib > len(calib_pool.examples):
            continue
        m = trial_metrics(
            calib_pool,
            test_pool,
            calib_gold,
            test_gold,
            alpha,
            n_calib,
            n_trials,
            "span_filter",
            99,
            pool_and_resplit=True,
        )
        m["n_calib"] = n_calib
        rows.append(m)
        print(f"[calib_size n={n_calib}] coverage={m['coverage_mean']:.4f}+-{m['coverage_std']:.4f}")
    return rows


def make_plots(rows: List[Dict], sensitivity_rows: List[Dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # (a) coverage vs alpha, small multiples per (pair, mode)
    pairs_modes = sorted({(r["pair"], r["mode"]) for r in rows})
    fig, axes = plt.subplots(1, len(pairs_modes), figsize=(5 * len(pairs_modes), 4), sharey=True)
    if len(pairs_modes) == 1:
        axes = [axes]
    for ax, (pair, mode) in zip(axes, pairs_modes):
        sub = sorted([r for r in rows if r["pair"] == pair and r["mode"] == mode], key=lambda r: r["alpha"])
        xs = [r["alpha"] for r in sub]
        ys = [r["coverage_mean"] for r in sub]
        es = [r["coverage_std"] for r in sub]
        ax.errorbar(xs, ys, yerr=es, marker="o", label="empirical")
        ax.plot([0, 1], [1, 0], "k--", alpha=0.5, label="y=1-alpha")
        ax.set_xlim(0, 0.25)
        ax.set_ylim(0, 1.05)
        ax.set_title(f"{pair}\n{mode}")
        ax.set_xlabel("alpha")
    axes[0].set_ylabel("empirical coverage")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(out_dir / "coverage_vs_alpha.png", dpi=150)
    plt.close(fig)

    # (b) efficiency vs alpha
    fig, axes = plt.subplots(1, len(pairs_modes), figsize=(5 * len(pairs_modes), 4), sharey=False)
    if len(pairs_modes) == 1:
        axes = [axes]
    for ax, (pair, mode) in zip(axes, pairs_modes):
        sub = sorted([r for r in rows if r["pair"] == pair and r["mode"] == mode], key=lambda r: r["alpha"])
        xs = [r["alpha"] for r in sub]
        ys = [r["efficiency_mean"] for r in sub]
        raw = [r["raw_candidates_mean"] for r in sub]
        ax.plot(xs, ys, marker="o", label="admitted (efficiency)")
        ax.plot(xs, raw, "k--", alpha=0.5, label="raw candidates (pre-filter)")
        ax.set_title(f"{pair}\n{mode}")
        ax.set_xlabel("alpha")
        ax.set_yscale("log")
    axes[0].set_ylabel("mean candidates / sentence")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(out_dir / "efficiency_vs_alpha.png", dpi=150)
    plt.close(fig)

    # (c) per-class coverage at alpha=0.1, span_filter mode
    pairs = sorted({r["pair"] for r in rows})
    fig, axes = plt.subplots(1, len(pairs), figsize=(6 * len(pairs), 4))
    if len(pairs) == 1:
        axes = [axes]
    for ax, pair in zip(axes, pairs):
        row = next((r for r in rows if r["pair"] == pair and r["mode"] == "span_filter" and r["alpha"] == 0.10), None)
        if row is None:
            continue
        types = sorted(row["per_type_coverage"])
        vals = [row["per_type_coverage"][t] for t in types]
        ax.bar(types, vals)
        ax.axhline(0.9, color="k", linestyle="--", alpha=0.5)
        ax.set_title(f"{pair} (alpha=0.1, span_filter)")
        ax.set_ylim(0, 1.05)
        ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(out_dir / "per_class_coverage.png", dpi=150)
    plt.close(fig)

    # (d) calibration-size sensitivity
    if sensitivity_rows:
        fig, ax = plt.subplots(figsize=(6, 4))
        xs = [r["n_calib"] for r in sensitivity_rows]
        ys = [r["coverage_mean"] for r in sensitivity_rows]
        es = [r["coverage_std"] for r in sensitivity_rows]
        ax.errorbar(xs, ys, yerr=es, marker="o")
        ax.axhline(0.9, color="k", linestyle="--", alpha=0.5)
        ax.set_xscale("log")
        ax.set_xlabel("n_calib")
        ax.set_ylabel("empirical coverage (alpha=0.1)")
        ax.set_title("Calibration-set-size sensitivity (in-domain CoNLL-2003)")
        fig.tight_layout()
        fig.savefig(out_dir / "calib_size_sensitivity.png", dpi=150)
        plt.close(fig)


def write_results_md(rows: List[Dict], sensitivity_rows: List[Dict], out_dir: Path, n_trials: int) -> None:
    lines = [
        "# Conformal-GLiNER Empirical Validation Results",
        "",
        f"Model: `{MODEL_ID}`. Trials per (pair, mode, alpha): {n_trials}. Pool cap: {POOL_CAP} sentences.",
        "",
        "**Disclosed scope** (docs/research/design.md's descope standard applies to the eval too): "
        "this run covers in-domain CoNLL-2003, in-domain WNUT-17, and zero-shot Pair A "
        "(CoNLL-2003 -> WNUT-17) from docs/research/eval_plan.md. It does not cover Pairs B/C or "
        "the full 5-domain CrossNER sweep -- those are documented as future work, not silently "
        "dropped.",
        "",
        "## Summary table",
        "",
        "`coverage_mean` is over calibrated types only (guaranteed, per design.md §5). "
        "`uncalibrated_coverage` (when present) is the raw p>0.5 empirical rate for types "
        "requested at test time that never met the calibration floor -- descriptive only, "
        "carries no guarantee, and is exactly what design.md §0's zero-shot descope predicts "
        "will happen for Pair A's WNUT-only types.",
        "",
        "| pair | mode | alpha | n_calib | trials_ok | coverage_mean | coverage_std "
        "| efficiency_mean | raw_candidates_mean | uncalibrated_coverage |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        uncal = r["uncalibrated_coverage_mean"]
        uncal_str = f"{uncal:.4f}" if uncal is not None else "n/a"
        lines.append(
            f"| {r['pair']} | {r['mode']} | {r['alpha']} | {r['n_calib']} "
            f"| {r['n_trials_ok']}/{r['n_trials_requested']} "
            f"| {r['coverage_mean']:.4f} | {r['coverage_std']:.4f} "
            f"| {r['efficiency_mean']:.1f} | {r['raw_candidates_mean']:.1f} | {uncal_str} |"
        )

    lines += ["", "## Per-type coverage (span_filter, alpha=0.1, calibrated types only)", ""]
    for r in rows:
        if r["mode"] == "span_filter" and r["alpha"] == 0.10:
            lines.append(f"**{r['pair']}**:")
            for t, c in sorted(r["per_type_coverage"].items()):
                lines.append(f"- {t}: {c:.4f}")
            lines.append("")

    if sensitivity_rows:
        lines += ["## Calibration-set-size sensitivity (in-domain CoNLL-2003, alpha=0.1, span_filter)", ""]
        lines.append("| n_calib | coverage_mean | coverage_std |")
        lines.append("|---|---|---|")
        for r in sensitivity_rows:
            lines.append(f"| {r['n_calib']} | {r['coverage_mean']:.4f} | {r['coverage_std']:.4f} |")

    (out_dir / "RESULTS.md").write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", default="results/conformal")
    ap.add_argument("--n_trials", type=int, default=50)
    ap.add_argument("--n_calib", type=int, default=500)
    ap.add_argument("--pool_cap", type=int, default=POOL_CAP)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model {MODEL_ID}...")
    model = GLiNER.from_pretrained(MODEL_ID)

    conll_labels = ["person", "organisation", "location", "misc"]
    wnut_labels = ["corporation", "creative-work", "group", "location", "person", "product"]

    print("Loading CoNLL-2003 validation...")
    conll_val = load_examples("DFKI-SLT/cross_ner", "conll2003", "validation", args.pool_cap)
    print("Loading CoNLL-2003 test...")
    conll_test = load_examples("DFKI-SLT/cross_ner", "conll2003", "test", args.pool_cap)
    print("Loading WNUT-17 validation...")
    wnut_val = load_examples("leondz/wnut_17", None, "validation", args.pool_cap)
    print("Loading WNUT-17 test...")
    wnut_test = load_examples("leondz/wnut_17", None, "test", args.pool_cap)

    print(f"CoNLL val/test: {len(conll_val)}/{len(conll_test)}, WNUT val/test: {len(wnut_val)}/{len(wnut_test)}")

    print("Building pools (forward passes)...")
    t0 = time.time()
    conll_calib_pool = build_pool(model, "conll_calib", conll_val, conll_labels)
    conll_test_pool = build_pool(model, "conll_test", conll_test, conll_labels)
    wnut_calib_pool = build_pool(model, "wnut_calib", wnut_val, wnut_labels)
    wnut_test_pool = build_pool(model, "wnut_test", wnut_test, wnut_labels)
    # Pair A: calibrate on CoNLL types, test on WNUT types -- score the WNUT test pool
    # against WNUT's own label set (already have wnut_test_pool for that), and score the
    # CoNLL calibration pool against CoNLL's own labels (already have conll_calib_pool).
    print(f"Pools built in {time.time() - t0:.1f}s")

    rows = []
    rows += run_suite(
        "in-domain CoNLL-2003",
        conll_calib_pool,
        conll_test_pool,
        args.n_trials,
        args.n_calib,
        pool_and_resplit=True,
    )
    rows += run_suite(
        "in-domain WNUT-17", wnut_calib_pool, wnut_test_pool, args.n_trials, args.n_calib, pool_and_resplit=True
    )
    rows += run_suite(
        "zero-shot CoNLL-2003->WNUT-17 (Pair A)", conll_calib_pool, wnut_test_pool, args.n_trials, args.n_calib
    )

    print("Calibration-size sensitivity (in-domain CoNLL-2003)...")
    sensitivity_rows = calib_size_sensitivity(conll_calib_pool, conll_test_pool, alpha=0.10, n_trials=args.n_trials)

    (out_dir / "raw_results.json").write_text(json.dumps({"rows": rows, "sensitivity": sensitivity_rows}, indent=2))
    make_plots(rows, sensitivity_rows, out_dir)
    write_results_md(rows, sensitivity_rows, out_dir, args.n_trials)
    print(f"Done. Results in {out_dir}/")


if __name__ == "__main__":
    main()
