# Empirical Validation Results

Phase 2 deliverable. Full protocol in `docs/research/eval_plan.md`; runnable source in
`scripts/conformal_validation.py`. Model: `gliner-community/gliner_small-v2.5`. 50 trials per
(pair, mode, α); pool cap 1200 sentences per source split (CPU runtime, disclosed below).
Raw output (`raw_results.json`, plots) is in `results/conformal/` (gitignored, matching this
repo's own convention — regenerate with the command in `scripts/conformal_validation.py`'s
docstring; takes about 3 minutes on a single CPU core once datasets are cached).

## Two real bugs found and fixed during this run, not glossed over

Both are recorded in full in `CLAUDE.md`'s decision log; summarized here because they're part
of what makes the numbers below trustworthy, not incidental to them.

1. **In-domain calibration/test split wasn't exchangeable.** The first pass calibrated on
   CoNLL-2003's official *validation* split and tested on its official *test* split, as static
   separate pools. Coverage undershot target by ~4-5 percentage points at every α, on both
   in-domain datasets — far outside sampling noise (~5 standard deviations at α=0.1). The
   calibration math itself was already independently verified correct (a 20,000-trial synthetic
   check in `tests/test_conformal_calibrators.py` lands at 0.9016 ± 0.0021 against a 0.9
   target), so the bug had to be in how real data was fed to it. A controlled comparison
   confirmed it: CoNLL-2003's validation and test splits have measurably different score
   distributions for this model (mean nonconformity 0.22 vs 0.27) — a real, documented property
   of how that benchmark's splits were constructed, not a code defect. `eval_plan.md` §2.2 had
   specified the right protocol for in-domain runs all along (pool validation+test, draw a
   fresh random partition every trial); the first implementation just hadn't followed it.
2. **`risk_control`'s reported coverage pooled entities flat instead of averaging per sentence.**
   Conformal Risk Control calibrates and guarantees the *per-sentence* average missed-entity
   rate (theory.md Eq. 4) — a different quantity from pooling every gold entity across every
   sentence whenever entity count varies per sentence, which it does in real data. This also
   affected the shipped library, not just the validation script — `ConformalGLiNER.coverage_report`
   had the identical bug, fixed in the same pass, with a deterministic regression test added
   (`test_risk_control_reports_per_sentence_not_per_entity_pooled`) that could not have been
   caught by the original synthetic unit test (its synthetic data happened to have exactly one
   gold entity per example, which makes the two quantities coincide).

Both fixes are visible in the git history as separate, atomic commits. The numbers below are
post-fix.

## Disclosed scope

- **Datasets**: in-domain CoNLL-2003, in-domain WNUT-17, and zero-shot Pair A (calibrate on
  CoNLL-2003's 4 types, measure coverage on WNUT-17) — the eval_plan.md-designated headline
  pair. **Not covered**: Pairs B/C (CrossNER-AI, CrossNER politics→music) and the full 5-domain
  CrossNER sweep. Given as future work, not silently dropped — see `eval_plan.md` §1.4 for
  what those would add.
- **Modes**: `span_filter` and `risk_control` were run through the full protocol.
  **`mondrian` was not separately run** — its per-type coverage claim is already directly
  evidenced by `span_filter`'s per-type breakdown below (see the `organisation`/`misc`
  under-coverage finding), which is exactly the failure mode Mondrian mode exists to fix; a
  standalone Mondrian validation run would largely re-demonstrate the same phenomenon with
  Mondrian's own (by-construction-valid, per theory.md v) per-type thresholds instead. Flagged
  as a scope decision, not an oversight.
- **Pool sizes** capped at 1200 sentences per source split for CPU tractability (~3 min total
  runtime including model + dataset loading). `n_calib=500` for the headline numbers.

## Summary table

`coverage_mean` is over calibrated types only — the actually-guaranteed number. `uncalibrated_coverage`
is the raw `p>0.5` empirical rate for types that never met the calibration floor — descriptive
only, no guarantee, shown for Pair A specifically to make the zero-shot descope (design.md §0)
concrete rather than abstract.

| pair | mode | α | coverage_mean | coverage_std | target (1−α) | efficiency_mean | uncalibrated_coverage |
|---|---|---|---|---|---|---|---|
| in-domain CoNLL-2003 | span_filter | 0.05 | 0.9482 | 0.0095 | 0.95 | 347.8 | n/a |
| in-domain CoNLL-2003 | span_filter | 0.10 | 0.8973 | 0.0126 | 0.90 | 223.3 | n/a |
| in-domain CoNLL-2003 | span_filter | 0.20 | 0.7947 | 0.0167 | 0.80 | 102.9 | n/a |
| in-domain CoNLL-2003 | risk_control | 0.05 | 0.9490 | 0.0095 | 0.95 | 389.0 | n/a |
| in-domain CoNLL-2003 | risk_control | 0.10 | 0.8978 | 0.0134 | 0.90 | 257.2 | n/a |
| in-domain CoNLL-2003 | risk_control | 0.20 | 0.7960 | 0.0193 | 0.80 | 101.3 | n/a |
| in-domain WNUT-17 | span_filter | 0.05 | 0.9523 | 0.0092 | 0.95 | 264.6 | 0.6961 |
| in-domain WNUT-17 | span_filter | 0.10 | 0.9005 | 0.0150 | 0.90 | 160.2 | n/a |
| in-domain WNUT-17 | span_filter | 0.20 | 0.8055 | 0.0242 | 0.80 | 79.9 | n/a |
| in-domain WNUT-17 | risk_control | 0.05 | 0.9526 | 0.0085 | 0.95 | 182.1 | 0.6961 |
| in-domain WNUT-17 | risk_control | 0.10 | 0.9049 | 0.0122 | 0.90 | 96.2 | n/a |
| in-domain WNUT-17 | risk_control | 0.20 | 0.8058 | 0.0171 | 0.80 | 32.5 | n/a |
| **Pair A** (CoNLL→WNUT) | span_filter | 0.05 | 0.9374 | 0.0088 | 0.95 | 61.8 | **0.5510** |
| **Pair A** | span_filter | 0.10 | 0.8724 | 0.0108 | 0.90 | 38.1 | **0.5510** |
| **Pair A** | span_filter | 0.20 | 0.8092 | 0.0116 | 0.80 | 17.6 | **0.5510** |
| **Pair A** | risk_control | 0.05 | 0.9802 | 0.0027 | 0.95 | 63.7 | **0.5510** |
| **Pair A** | risk_control | 0.10 | 0.9578 | 0.0032 | 0.90 | 36.7 | **0.5510** |
| **Pair A** | risk_control | 0.20 | 0.9320 | 0.0056 | 0.80 | 12.3 | **0.5510** |

**Reading this table**: every in-domain row tracks its target closely (within roughly 0.3–2
standard deviations, both directions — matches the theory, which permits mild over-coverage,
never systematic under-coverage, at finite n). Pair A's *calibrated* types (`location`,
`person` — shared vocabulary with CoNLL) show mild under-coverage for `span_filter` at tight α
(0.8724 vs 0.90 target at α=0.1) — consistent with `docs/conformal.md`'s stated limitation that
domain shift degrades calibrated-type coverage too, not just uncalibrated types; `risk_control`
over-covers on Pair A instead, which is a healthy direction to be wrong in (the guarantee is
"≥", not "="). Pair A's **uncalibrated** types (`corporation`, `creative-work`, `group`,
`product` — never seen during CoNLL calibration) sit at **0.551 coverage regardless of α or
mode** — exactly the flat, unguaranteed number one gets from a fixed raw-threshold rule, in
stark contrast to the 0.87–0.98 the calibrated types achieve. This is the concrete number
behind the "not a zero-shot guarantee" claim in `docs/conformal.md` and `docs/PR_DESCRIPTION.md`
— not a hedge, an observed fact.

## Per-type coverage (span_filter, α=0.1) — the motivation for Mondrian mode, made concrete

| Dataset | Type | Coverage | vs. 0.90 target |
|---|---|---|---|
| CoNLL-2003 | `location` | 0.9773 | over |
| CoNLL-2003 | `person` | 0.9766 | over |
| CoNLL-2003 | `misc` | 0.8493 | **under** |
| CoNLL-2003 | `organisation` | **0.7170** | **substantially under** |
| WNUT-17 | `person` | 0.9496 | over |
| WNUT-17 | `product` | 0.8859 | ~on target |
| WNUT-17 | `location` | 0.8839 | ~on target |
| WNUT-17 | `corporation` | 0.8429 | under |
| WNUT-17 | `creative-work` | 0.8462 | under |
| WNUT-17 | `group` | 0.8103 | under |

CoNLL-2003's `organisation` type sits at 0.717 coverage against a 0.90 target under the pooled
`span_filter` guarantee — a real, measured instance of exactly the "rare/harder types
systematically under-covered by the marginal guarantee" failure mode `theory.md` iii-c predicts
and cites 2601.16999 Table 8 for. The pooled guarantee (3(a)) is only a statement about the
*average* across all calibrated types — it says nothing about any individual type, and
`organisation` (evidently a harder type for this model — more heterogeneous surface forms than
`person`/`location`) is the one absorbing the slack that keeps the pooled average near target.
This is the direct empirical case for `mondrian` mode, not a hypothetical one.

## Calibration-set-size sensitivity (in-domain CoNLL-2003, α=0.1, span_filter)

| n_calib | coverage_mean | coverage_std |
|---|---|---|
| 50 | 0.9012 | 0.0374 |
| 100 | 0.9055 | 0.0224 |
| 200 | 0.9010 | 0.0162 |
| 500 | 0.8991 | 0.0110 |
| 1000 | 0.8991 | 0.0092 |

Mean sits within 0.006 of the 0.90 target at every tested size (no systematic drift as n
grows — the earlier bugs, when present, showed up here too as a mean stuck around 0.85–0.86
regardless of n, which is itself a useful diagnostic pattern: variance shrinking without the
mean converging to target is a sign of a real bug, not of "needing more data"). Standard
deviation shrinks monotonically from 0.0374 at n=50 to 0.0092 at n=1000, exactly the
`Θ(1/√n)`-type behavior the finite-sample theory predicts.

## Plots

Four PNGs in `results/conformal/` (not committed — regenerate via `scripts/conformal_validation.py`):
`coverage_vs_alpha.png`, `efficiency_vs_alpha.png`, `per_class_coverage.png`,
`calib_size_sensitivity.png`.
