<!--
LOCAL ONLY — draft for the user to review before manually opening the upstream PR.
Not committed to the eventual PR itself; lives in this working tree for review.
Placeholders marked [FILL: ...] are pending the empirical validation run
(scripts/conformal_validation.py) finishing; see docs/research/validation_results.md
once written.
-->

# Add conformal prediction: calibrated coverage/risk guarantees for zero-shot NER

## Motivation

GLiNER scores every candidate `(span, type)` pair with an independent sigmoid and filters
with `threshold=0.5` by default. That threshold has no statistical meaning — it doesn't say
what fraction of true entities a user should expect to miss, and it isn't calibrated to any
particular deployment's data or entity types. Several open issues in this repo are symptoms
of exactly this gap: #69 (feature request just to expose confidence values at all), #192
(label ordering changing confidence scores — a miscalibration symptom), #324 (transformers
v5 causing uniformly low/meaningless scores).

This PR adds `gliner.conformal`, a small additive module that replaces the arbitrary
`threshold=0.5` cutoff with a threshold **calibrated on a held-out labeled set**, backed by
finite-sample, distribution-free guarantees from the conformal prediction literature
(Vovk et al.; Angelopoulos & Bates, arXiv:2107.07511; Angelopoulos et al., Conformal Risk
Control, arXiv:2208.02814). Two very recent papers (Singer, Sengupta & Pazdernik,
arXiv:2601.16999; Kotte, PASC, arXiv:2605.18812) establish conformal-prediction theory for
NER specifically, but neither ships code and neither addresses open-vocabulary/zero-shot
label sets — as far as we can find (see `docs/research/prior_art.md` for the full survey:
MAPIE, crepes, TorchCP, Fortuna, PUNCC, nonconformist all checked), **no released
implementation of conformal prediction for NER exists anywhere**, closed-set or otherwise.
This is, to our knowledge, the first one, and the first that works with GLiNER's
inference-time arbitrary label sets.

## What this is NOT claiming

Read `docs/conformal.md`'s Limitations section and `docs/research/theory.md` §vi in full
before reviewing the API — the short version: **this is not a rigorous zero-shot coverage
guarantee for entity types never seen in calibration.** Split-conformal validity requires
calibration/test exchangeability; a type with zero calibration occurrences has no
well-defined quantile (undefined, not merely wide) and no theorem in the literature we
surveyed licenses a coverage claim for it. `ConformalGLiNER` handles this honestly: types
below the calibration floor get a loud warning and GLiNER's original uncalibrated behavior,
flagged `"calibrated": False` on every affected entity — never silently blended into a
guaranteed-looking number. We think shipping this scoped-but-honest version is more useful,
and more credible, than a version that quietly overclaims.

## API

```python
from gliner import GLiNER
from gliner.conformal import ConformalGLiNER

model = GLiNER.from_pretrained("gliner-community/gliner_small-v2.5")
cg = ConformalGLiNER(model)                                  # wraps, never mutates, the model
cg.calibrate(calib_data, alpha=0.1, mode="risk_control")      # mode: span_filter | risk_control | mondrian
entities = cg.predict_entities(text, labels)                  # entities + "conformal" guarantee metadata
report = cg.coverage_report(test_data)                        # empirical validation, disjoint from calib_data
cg.save_calibration(path)                                     # JSON: thresholds, alpha, mode, calibrated types
ConformalGLiNER.load_calibration(path, model)
```

Three guarantee modes (full math in `docs/research/theory.md`, accessible explanation in
`docs/conformal.md`):
- **`span_filter`** — marginal per-entity coverage `P(gold span ∈ output) ≥ 1-α`.
- **`risk_control`** — Conformal Risk Control bounding expected missed-entity rate ≤ α (the
  flagship mode for compliance/PII use cases). We prove GLiNER's independent-sigmoid decode
  rule satisfies CRC's required monotonicity condition *by construction* (`theory.md` §iii-b),
  and identify one real implementation hazard doing that proof: greedy overlap resolution
  must be applied *after* the conformal threshold, not before, or the nesting property CRC
  needs breaks. `ConformalGLiNER` is built this way; there's a regression test
  (`test_monotonicity_precondition_is_checked_by_default`) tied directly to the proof.
- **`mondrian`** — per-type calibration so rare types aren't systematically under-covered by
  the marginal guarantee, with an explicit, enforced calibration-data floor per type.

## Design

No changes to any existing model code. Raw pre-sigmoid, pre-decode span scores are already
reachable via the public `model.run_batch()` (confirmed by reading every `forward()` in
`gliner/modeling/base.py` — see `docs/research/repo_map.md` §5); `gliner/conformal/` is a
pure additive package:

```
gliner/conformal/
├── scores.py        # raw score extraction, span-mode only (see Scope below)
├── calibrators.py   # pure-Python split-conformal quantile, CRC λ-search, Mondrian partitioning
└── wrapper.py        # ConformalGLiNER
```

`calibrators.py` has no GLiNER dependency and is independently unit-tested against synthetic
scores with analytically known coverage.

### Scope: span-mode models only

`UniEncoderSpanGLiNER` and `BiEncoderSpanGLiNER` (the default `span_mode="markerV0"`
architecture). Token-mode, generative-decoder, and relation-extraction variants apply their
confidence threshold *inside* the forward pass to prune candidates before returning scores
(`get_span_representations` → `extract_spans_from_tokens`, and the relex adjacency-selection
paths), so `run_batch()`'s output isn't the full candidate universe for those architectures.
Calibrating against it would silently understate true coverage rather than producing a valid
guarantee, so it's explicitly unsupported (`NotImplementedError`, not a silent wrong answer).

## Tests

- `tests/test_conformal_calibrators.py` — synthetic, no network. Includes a 20,000-trial
  empirical coverage check for `split_conformal_quantile` (measured 0.9016 ± 0.0021 against a
  0.9 target) and a 200-trial risk check for `crc_lambda_search`.
- `tests/test_conformal_gliner.py` — integration tests against
  `gliner-community/gliner_small-v2.5` (mirrors `test_models.py`'s existing network-touching
  test pattern). Covers all three modes, calibration-floor enforcement, the out-of-calibration
  warn+fallback path, empty predictions, save/load round-trip (including a model-mismatch
  warning), `coverage_report`'s shape, and rejection of non-span-mode models.
- **334 pre-existing tests pass unmodified** — confirms this is fully additive with zero
  regressions to existing functionality.

## Empirical validation

`scripts/conformal_validation.py` runs the protocol in `docs/research/eval_plan.md` against
real data (CoNLL-2003 and WNUT-17 via `DFKI-SLT/cross_ner`, `gliner-community/gliner_small-v2.5`):
in-domain calibration/coverage on both datasets, plus the zero-shot Pair A experiment
(calibrate on CoNLL-2003's 4 types, measure coverage on WNUT-17) that's designed to
*demonstrate*, not just claim, the exchangeability limitation above.

**Two real bugs surfaced and got fixed during this run, not after** — both documented in full
in `docs/research/validation_results.md` and `CLAUDE.md`'s decision log, summarized here
because "the empirical section validates the theory" is a claim worth being able to audit, not
take on faith:

1. The first pass showed a consistent ~4-5pp coverage undershoot at every α, on both in-domain
   datasets — a red flag, since the calibrator math is independently unit-tested to land within
   noise of target (20,000-trial synthetic check: 0.9016 ± 0.0021 against 0.9). Diagnosis:
   calibrating on CoNLL-2003's *official* validation split and testing on its *official* test
   split showed a real, measurable score-distribution gap between the two (mean nonconformity
   0.22 vs 0.27) — those two splits are not fully exchangeable for this model, a property of how
   the benchmark's splits were constructed, not a defect in the conformal machinery.
   `eval_plan.md` §2.2 already specified the correct protocol for in-domain runs (pool
   validation+test, draw a fresh random partition every trial); the first implementation had
   deviated from it.
2. `risk_control` calibrates and guarantees a *per-sentence* average missed-entity rate — a
   different quantity from pooling every gold entity flat across sentences whenever entity
   count varies per sentence. This bug was in the **shipped library**, not just the validation
   script: `ConformalGLiNER.coverage_report` had the same flaw, fixed in the same commit, with a
   deterministic regression test added that the original synthetic unit test structurally could
   not have caught (its synthetic data had exactly one entity per example).

Both fixed; numbers below are post-fix.

### Summary (α ∈ {0.05, 0.10, 0.20}, coverage_mean over calibrated types only)

| pair | mode | target | measured (α=0.05 / 0.10 / 0.20) |
|---|---|---|---|
| in-domain CoNLL-2003 | span_filter | 0.95 / 0.90 / 0.80 | 0.9482 / 0.8973 / 0.7947 |
| in-domain CoNLL-2003 | risk_control | 0.95 / 0.90 / 0.80 | 0.9490 / 0.8978 / 0.7960 |
| in-domain WNUT-17 | span_filter | 0.95 / 0.90 / 0.80 | 0.9523 / 0.9005 / 0.8055 |
| in-domain WNUT-17 | risk_control | 0.95 / 0.90 / 0.80 | 0.9526 / 0.9049 / 0.8058 |

Every in-domain row tracks its target within ~0.3-2 standard deviations, both directions —
matches theory, which permits mild finite-sample over-coverage but never systematic
under-coverage.

### Pair A: the zero-shot descope, measured, not just claimed

Calibrating on CoNLL-2003 and testing coverage on WNUT-17: the two **calibrated** types
(`location`, `person` — shared vocabulary with CoNLL, so genuinely represented in calibration)
reach 0.87–0.98 coverage across α. The four **uncalibrated** types (`corporation`,
`creative-work`, `group`, `product` — never seen during CoNLL calibration) sit at a **flat
0.551 coverage regardless of α or mode** — exactly the unguaranteed number you get from a raw
threshold with no calibration behind it. That gap (0.87-0.98 vs 0.551) is the concrete evidence
for this PR's central limitation claim, not a hedge.

### Calibration-set-size sensitivity (in-domain CoNLL-2003, α=0.1)

Coverage mean stays within 0.006 of the 0.90 target at every tested size (n ∈ {50, 100, 200,
500, 1000}); standard deviation shrinks monotonically from 0.0374 to 0.0092 — the expected
`Θ(1/√n)` behavior, and a useful diagnostic: when the two bugs above were still present, this
sweep showed a mean *stuck* around 0.85-0.86 regardless of n, which is itself the tell that
something was wrong (variance-without-convergence, not "just needs more data").

Full results, per-type coverage breakdown (including a real illustration of *why* `mondrian`
mode exists — CoNLL's `organisation` type measures 0.717 coverage against a 0.90 pooled target,
while `location`/`person` overshoot to compensate), plots, and this run's disclosed scope are
in `docs/research/validation_results.md`. Plots are regenerable via
`scripts/conformal_validation.py` (not committed as binaries — see that file for the exact
command, ~3 min on one CPU core) and can be attached directly to the GitHub PR.

## Documentation

- `docs/conformal.md` — practitioner guide: motivation, the three modes explained
  accessibly, a runnable example, and a Limitations section covering the exchangeability
  caveat, domain shift, Mondrian's calibration-data cost, and the span-mode-only scope.
- `docs/research/{repo_map,theory,prior_art,eval_plan,design}.md` — the full research and
  design trail behind every decision above, kept for anyone who wants to audit the reasoning
  (not typically part of a PR, offered here for transparency; can be trimmed from the actual
  PR diff if the maintainers prefer a leaner change).

## Additive, reviewed for scope creep

- No changes to `gliner/model.py`, `gliner/modeling/`, `gliner/decoding/`, or any other
  existing file.
- No new required dependencies for the shipped package — `gliner/conformal/` uses only
  NumPy/PyTorch (already required). `scripts/conformal_validation.py` (not part of the
  package; dev/validation tooling only, not imported by anything in `gliner/`) additionally
  uses `datasets` and `matplotlib` to fetch benchmark data and produce plots. **Deliberately
  left undeclared in `pyproject.toml`/`requirements.txt`**, matching this repo's own established
  convention: none of the existing benchmark/eval scripts under `scripts/` (e.g.
  `convert_to_onnx.py`, or the vocab-pruning branch's `baseline_eval.py`/`visualize_results.py`,
  which need the same `datasets`/`matplotlib` tooling) declare their dependencies either — users
  install them ad hoc to run a specific script. Confirmed this isn't a CI risk either:
  `.github/workflows/tests.yml` runs `pytest -q --tb=short` against only `requirements.txt` +
  `pytest`/`sentencepiece`/`onnxruntime`, and `scripts/` isn't collected by pytest
  (`testpaths = ["tests"]`), so `tests/test_conformal_*.py` (which need neither package) run
  clean either way. `ruff check gliner` (CI's exact lint invocation, which only covers
  `gliner/`, not `tests/`/`scripts/`) passes clean on this branch.
- `gliner/conformal` is not imported by `gliner/__init__.py` by default — it's an opt-in
  `from gliner.conformal import ConformalGLiNER`, so there's no import-time cost for users
  who don't use it.
