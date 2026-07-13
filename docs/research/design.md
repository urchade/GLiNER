# Conformal-GLiNER — Design Doc (Phase 1)

Synthesizes `repo_map.md` (Agent A), `theory.md` (Agent B), `prior_art.md` (Agent C),
`eval_plan.md` (Agent D). Resolves every open question the mission brief listed for Phase 1.
Math notation matches `theory.md` throughout — read that file first if any statement below
looks unmotivated; it isn't restated here in full, only cited by section.

---

## 0. The one decision that reshapes everything else

The mission brief's framing — "give me predictions such that, with probability ≥ 1−α, [a
guarantee] holds," calibrated once and implicitly expected to travel to arbitrary zero-shot
labels — is **not rigorously supportable**, per `theory.md` §(vi). Restating the argument in one
paragraph because it drives every design choice below:

Split conformal validity requires the calibration and test `(x, y)` pairs to be **exchangeable**
as an `(n+1)`-tuple. If calibration only ever sees entity type `t ∈ 𝒯_cal`, then a query for a
type `t* ∉ 𝒯_cal` has **zero calibration mass** — there is no sense in which it's exchangeable
with the calibration draw (Mondrian quantile at `n=0` is undefined, not wide; the pooled/marginal
guarantee doesn't transfer either, since it's a statement about the pooled calibration
*population*, and `t*` wasn't part of it). Neither `theory.md`'s two target papers nor the
covariate-shift-conformal literature they cite licenses a finite-sample claim here — covariate-
shift reweighting (Tibshirani et al. 2019) handles *rare-but-present* types, not *never-observed*
ones.

**Decision: `ConformalGLiNER` ships a rigorous guarantee scoped explicitly to the set of entity
types represented (with adequate calibration mass) in the calibration set. It does not, and will
not claim to, guarantee coverage for arbitrary user-supplied zero-shot types at inference time.**
This is a deliberate, documented descope per the mission brief's own §5 standard ("if a guarantee
mode turns out not to be validly implementable, document why and descope it rather than shipping
fake rigor"). The product is still genuinely useful and still novel (see `prior_art.md` §3 — no
existing conformal-NER work, closed-set or otherwise, has been released as code at all): it's
"calibrated coverage for the label set you calibrated on," which is exactly what CoNLL/WNUT/
CrossNER-style deployments actually do in practice (a fixed extraction schema, calibrated once).
What we do *not* get to say is "point this at a type nobody's ever calibrated and still get a
number with meaning" — for that case we fall back to a loudly-flagged, non-guaranteed heuristic
(§5 below), never a silent one.

This needs your explicit sign-off — see Open Questions, §8.

---

## 1. Guarantee modes shipped (three), and two modes explicitly NOT shipped

### 1.1 `"span_filter"` — marginal per-entity coverage (default mode)

**Statement** (theory.md iii-a, Eq. 3):
```
P( gold span ∈ C_t(x_new) | (span, t) is a true entity of type t in x_new ) ≥ 1 − α
```
Nonconformity score `s(x, (span,t)) = 1 − p_θ(span, t | x)`. Threshold `τ_t` = the
`⌈(n+1)(1−α)⌉/n`-quantile (theory.md i) of calibration scores for gold spans of type `t`.
`C_t(x) = {span : s(x,(span,t)) ≤ τ_t}`.

**Exchangeability unit**: the pool of *(sentence, gold-span)* pairs whose true type is `t`,
across the calibration corpus (theory.md ii, unit 2c) — not sentences, not all-spans-pooled.
`1−α` bounds a frequency over **entity occurrences of type t**, not over sentences. A sentence
with 10 type-`t` entities contributes 10 trials; this is a known, documented property (not a
bug) — see `coverage_report()`'s per-class breakdown, §5.

**Why "span-filter" and not "sentence-filter"**: GLiNER has no joint sequence model (no CRF) —
it emits independent per-`(span,type)` sigmoids (repo_map.md §3, confirmed for every one of the
6 forward-pass variants). The full-sequence framing from 2601.16999 requires ranking whole-
sentence labelings by a joint probability GLiNER structurally does not compute (theory.md iv).
**Full-sequence mode is not shipped** — see §1.4.

### 1.2 `"risk_control"` — Conformal Risk Control on missed-entity rate (flagship mode)

**Loss** (theory.md iii-b, Eq. 4), threshold `λ ∈ [0,1]`, `Cλ(x) = {(span,t) : p_θ ≥ 1−λ}`:
```
ℓ(Cλ(x), y(x)) = 1 − |y(x) ∩ Cλ(x)| / |y(x)|     if y(x) ≠ ∅,  else 0
```
`λ̂ = inf{λ : R̂ₙ(λ) + (1−α)/n ≤ α}` (CRC's finite-sample-conservative formula, `B=1` since the
loss is bounded in `[0,1]`). Then `E[ℓ(C_λ̂(X_new), Y_new)] ≤ α` — "we provably miss < α of
entities on average," exactly the compliance/PII framing the mission brief wants as the
flagship.

**Monotonicity proof** (theory.md iii-b, Claims 1–2, done in full, not hand-waved): GLiNER's
independent-sigmoid, single-shared-threshold decode rule is *nested* by construction
(`λ₁≤λ₂ ⟹ Cλ₁⊆Cλ₂`), which makes the miss-rate loss monotone non-increasing in `λ` — CRC's
required condition holds **by construction**, not by assumption. This is the one place the
brief asked us to "show," and it's shown: `theory.md` iii-b Claims 1/2 + right-continuity +
boundary argument, all proved from GLiNER's actual decode semantics, not asserted.

**The one implementation hazard this proof exposes** (theory.md iii-b, final paragraph):
GLiNER's *deployed* decoder applies greedy overlap resolution (`greedy_search`,
`gliner/decoding/decoder.py:92-137`, repo_map.md §4) **after** thresholding, and that step can
break nesting — a span present at a looser `λ` could get suppressed by a newly-admitted
higher-priority overlapping span that wouldn't have existed at a tighter `λ`. **Design fix**:
`Cλ(x)` for calibration/risk-control purposes is always defined on the **pre-overlap-resolution
candidate set** (raw thresholded pairs, straight off `run_batch()`'s output — repo_map.md §5).
Overlap resolution (flat-NER collapsing) is applied as a **separate, threshold-independent
post-processing step** for the user-facing `predict_entities()` output, reusing
`gliner.decoding.utils.has_overlapping`/`has_overlapping_nested` unchanged, but it never
participates in the `λ` calibration/nesting argument. Coverage/risk numbers in
`coverage_report()` are computed against the pre-resolution set; the returned `predict_entities`
spans are post-resolution for usability. This is documented explicitly (and unit-tested
explicitly — a nesting-violation regression test) precisely because it's the one spot the
theory doesn't automatically protect us.

### 1.3 `"mondrian"` — class-conditional span-filter

Per-type version of §1.1: independent threshold `τ_t` calibrated only on type-`t` calibration
occurrences, for **every type with `n^(t) ≥ ⌈1/α⌉ − 1`** (theory.md v's hard floor — below this
the quantile is undefined/degenerate, not just wide). Guarantee (theory.md iii-c, Eq. 7) holds
*simultaneously* for every qualifying type — strictly stronger than §1.1's pooled-average bound,
and the direct fix for the "rare types systematically under-covered by the marginal guarantee"
failure mode the mission brief names (empirically confirmed as a real failure by 2601.16999
Table 8, not hypothetical — theory.md iii-c).

Types below the floor: **not given a Mondrian threshold**. `calibrate(mode="mondrian")` records
which types qualified and which didn't; `predict_entities()` on a sub-floor type falls back to
`"span_filter"`'s pooled threshold with the same loud non-guarantee warning as §5.

### 1.4 Explicitly NOT shipped in v1, with reasons

- **Full-sequence / sentence-level conformal sets** (2601.16999's headline method). Requires a
  joint sequence probability model; GLiNER doesn't have one (§1.1). Building one would be a new
  architecture component, violating the mission brief's "no architecture changes" constraint.
  Descoped, not attempted.
- **PASC-style pipeline-joint coverage** (2605.18812). Per theory.md iv's judgment call: PASC's
  own paper states it collapses to standard split conformal at `K=1` (single stage) — plain
  GLiNER NER *is* `K=1`, so PASC adds nothing here. **Kept in the back pocket**: if
  GLiNER-Robust's scope later grows to include the repo's existing relation-extraction wrapper
  (`predict_relations`, per repo cartography — chaining NER→RE), PASC's max-nonconformity
  reduction (their Prop. 4, verified correct in theory.md) becomes directly relevant. Not now.
- **Rigorous guarantees for never-calibrated types** — see §0. This is the load-bearing descope.

---

## 2. Nonconformity score

**Default and only score for v1: `s(x, (span,t)) = 1 − p_θ(span, t | x)`** where `p_θ` is
GLiNER's own sigmoid output — this is literally the quantity GLiNER already computes (no extra
forward pass, no architecture change; repo_map.md §5 confirms `run_batch()` returns exactly this
pre-sigmoid logit, one `torch.sigmoid` call away from `p_θ`). This is the direct GLiNER analogue
of 2601.16999's subsequence-mode nonconformity scores (theory.md iv) — no top-K-beam
approximation needed, since GLiNER's per-pair sigmoid already *is* the marginal probability the
CRF paper has to approximate via beam search.

Rank-based and length-normalized alternatives (mission brief §3) are **not implemented in v1** —
noted as a documented extension point in `calibrators.py` (score computation is isolated in one
function so swapping it later doesn't touch the calibration engine), not built now. No evidence
from any of the four reports that they're needed for a correct v1; adding them without an
empirical reason would be scope creep.

---

## 3. API design

```python
from gliner.conformal import ConformalGLiNER

model = GLiNER.from_pretrained("gliner-community/gliner_small-v2.5")
cg = ConformalGLiNER(model)                      # wraps, never mutates, the model

# calib_data: List[Dict] — same {"tokenized_text": [...], "ner": [[start,end,"type"],...]}
# shape GLiNER's own evaluate()/training pipeline already uses (data_processing/processor.py,
# to be confirmed exactly against this checkout in Phase 2 — repo_map.md didn't fully verify
# the training-JSON schema, only the inference-time predict_entities signature)
cg.calibrate(calib_data, alpha=0.1, mode="risk_control")   # mode ∈ {span_filter, risk_control, mondrian}

preds = cg.predict_entities(text, labels)        # entities + guarantee metadata (see below)
report = cg.coverage_report(test_data)           # empirical validation, disjoint from calib_data

cg.save_calibration(path)                        # JSON: scores/quantiles, alpha, mode, calibrated-type set + counts, model id/hash
ConformalGLiNER.load_calibration(path, model)     # classmethod; re-wraps a (possibly different-process) model
```

`predict_entities()` return shape extends the normal GLiNER entity dict with a guarantee-status
field per entity, e.g. `{"text": ..., "label": ..., "start": ..., "end": ..., "score": ...,
"conformal": {"mode": "risk_control", "alpha": 0.1, "calibrated": true}}` — `"calibrated":
false` is set (never silently omitted) whenever the entity's type falls outside the calibrated
type set, per §5.

`ConformalGLiNER` never mutates `model` — it holds a reference and calls `model.run_batch(...)`
(public, repo_map.md §5) directly, applying its own sigmoid + conformal threshold + (for
`predict_entities`, not `coverage_report`) the existing `greedy_search`/`has_overlapping`
post-processing from `gliner.decoding.utils`. **Zero core-model changes required** — repo_map.md
§5 confirms `run_batch` already exposes exactly the raw tensor needed; this was the mission
brief's "at most, expose raw span scores if not already accessible" contingency, and it turns out
not to be needed at all.

---

## 4. Package placement

```
gliner/conformal/
├── __init__.py         # exports ConformalGLiNER, calibrate/quantile helpers
├── scores.py            # extract_span_scores(model, texts, labels) -> raw (B,L,K,C)-or-(B,W,C,3)
                          #   tensor + aligned gold-span index, per repo_map.md §5's interception point
├── calibrators.py       # pure NumPy/PyTorch, model-agnostic, independently synthetic-testable:
                          #   split_conformal_quantile(scores, alpha) — the ⌈(n+1)(1-α)⌉/n order stat
                          #   crc_lambda_search(losses, alpha) — CRC's inf{...} search, monotone-loss-checked
                          #   mondrian_partition(scores, types, alpha) — per-type calibration + floor check
└── wrapper.py            # ConformalGLiNER: calibrate/predict_entities/coverage_report/save/load
```

Matches repo_map.md §1's existing layout convention (`gliner/decoding/`, `gliner/evaluation/` as
siblings of `gliner/modeling/`) — `gliner/conformal/` sits at the same level, purely additive,
imports from `gliner.decoding.utils` and `gliner.model` but nothing imports it back. Test files:
`tests/test_conformal_calibrators.py` (synthetic, no network — mirrors `test_decoder.py`'s
fixture pattern, repo_map.md §9) and `tests/test_conformal_gliner.py` (integration, downloads
`gliner-community/gliner_small-v2.5` once — mirrors `test_models.py::test_span_model`'s only
network-touching pattern).

---

## 5. Out-of-calibration-type semantics (the "guarantee void" warning, made precise)

Per §0's decision, this is not a generic "label sets differ" warning — it's specific and
mechanical:

1. At `calibrate()` time, record `𝒯_cal` = every type with `n^(t) ≥ ⌈1/α⌉ − 1` calibration
   occurrences (the theory.md v floor), plus, separately, every type seen at all (even below
   floor) for diagnostic purposes.
2. At `predict_entities(text, labels)` time, for each requested label `∉ 𝒯_cal`:
   - Emit a `UserWarning` (once per call, listing the offending types, not once per span) —
     "type(s) {…} were not adequately represented in calibration (need ≥N occurrences, saw M);
     the ≥1−α guarantee does NOT apply to these types."
   - Still return predictions for that type (don't silently drop user-requested labels), but
     with raw uncalibrated `p_θ > 0.5` filtering (GLiNER's original behavior) and
     `"conformal": {"calibrated": false}` on every entity of that type.
3. `coverage_report()` on a test set containing out-of-calibration types **must** report their
   coverage separately from calibrated types, never blend them into one aggregate number — a
   blended number would silently launder an unguaranteed result into a guaranteed-looking one.

This is the concrete mechanism that turns §0's descope from a documentation note into an
enforced, testable behavior (edge-case test: "unseen labels" from the mission brief's Phase 2
test list, §4 checklist below).

---

## 6. Calibration-set-size floor enforcement

Per eval_plan.md §2.1: `calibrate()` **raises**, does not silently degrade, when
`n_calib < ⌈1/α⌉` for the mode's relevant pool (whole calibration set for `span_filter`/
`risk_control`; per-type pool for `mondrian` — where sub-floor types are excluded per §1.3
rather than raising, since other types may still be fine). Error message states the exact
floor and the observed `n`. This matches the mission brief's own worked example (n=10, α=0.05
needs rank 11 > 10) almost exactly — eval_plan.md §2.1 independently derives the same table.

---

## 7. Empirical validation plan (adopted from eval_plan.md verbatim, summarized)

- Checkpoint: `gliner-community/gliner_small-v2.5` (Apache-2.0, ≈166M params, CPU-feasible).
- Datasets: CoNLL-2003 and WNUT-17 and CrossNER (5 domains), all via `DFKI-SLT/cross_ner`
  configs to sidestep `datasets`'s script-loading rejection (eval_plan.md §1 — verified live).
- Zero-shot transfer pairs (used to *demonstrate* §0's descope empirically, not to claim it
  doesn't apply): (A) CoNLL-2003→WNUT-17, (B) CoNLL-2003→CrossNER-AI, (C) CrossNER-politics→
  CrossNER-music. Pair A is expected to show visibly degraded/undefined coverage for WNUT-17's
  `corporation`/`creative-work`/`group`/`product` types — that's not a bug to fix, it's the
  planned empirical demonstration of why §0's scoping decision is necessary, and it becomes a
  figure in the eventual PR/paper, not a swept-under-the-rug failure.
- Metrics/plots: exactly eval_plan.md §3–4 (coverage-vs-α with T=100 seeded trials, efficiency,
  per-class bars, calibration-size sensitivity) — adopted without modification, it's already
  concrete and directly implementable.

---

## 8. Open questions for HARD STOP #1 (need your explicit answers)

1. **§0's descope** — ship "coverage guaranteed for calibration-represented types only,"
   explicitly not a zero-shot guarantee for arbitrary novel types. This is the single biggest
   deviation from the mission brief's literal framing. Approve, or want a different treatment
   (e.g. descope further to *only* closed-set mode and drop the "zero-shot" framing from
   marketing entirely; or, invest in the speculative embedding-distance/Lipschitz argument
   theory.md iii-vi flags as a currently-unestablished alternative foundation — explicitly out
   of scope for this project as scoped)?
2. **Default mode** — propose `risk_control` as the flagship default (matches the mission
   brief's own "flagship for compliance/PII users" framing) but `span_filter` as the
   conceptually simpler one. Which should `calibrate()`'s default `mode=` be, or require it
   explicit with no default?
3. **Out-of-calibration-type behavior (§5)** — propose "warn loudly + return raw-threshold
   predictions flagged `calibrated: false}`" rather than refuse outright. Confirm, or prefer a
   hard refusal (raise instead of warn-and-degrade)?
4. **Full-sequence and PASC modes** — confirmed out of scope for v1 (§1.4). Any objection?
5. **Calibration data format** — assumed to reuse GLiNER's existing training/eval JSON schema
   (`tokenized_text` + `ner` triples); Phase 2's first task will verify this exactly against
   `gliner/data_processing/processor.py` before writing `scores.py`. Flagging now since
   repo_map.md didn't fully pin this down (it focused on the inference path, not training-data
   ingestion) — not a blocker, just noting it's the first thing Phase 2 confirms.
6. Anything from the four research reports you want re-litigated before implementation starts —
   in particular Agent B's retracted-fabrication note (theory.md §vi) is worth your own read if
   you want to sanity-check the most load-bearing claim in this document yourself.

No implementation code has been written. Everything above is docs only
(`docs/research/{repo_map,theory,prior_art,eval_plan,design}.md`), `CLAUDE.md`, and
`.gitignore`/housekeeping commits. Awaiting your answers before Phase 2 starts.
