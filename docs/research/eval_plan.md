# Empirical Evaluation Protocol — Conformal-GLiNER

**Phase 0, Agent D deliverable.** Defines *how Phase 2 will empirically prove* (not just
assert) that `ConformalGLiNER`'s coverage/risk guarantees actually hold, at what cost in
prediction-set size, and how they behave under genuine zero-shot label transfer. This is a
protocol document — it fixes dataset IDs, split sizes, formulas, and plots so Phase 2 can be
implemented without further research. It does not choose the nonconformity score or guarantee
mode (marginal vs. Mondrian, split-conformal vs. CRC) — that is Agent B / `design.md`'s call.
Every metric below is defined generically against "the conformalized prediction set `C(x)`",
so it slots in unchanged whichever score function Phase 1 settles on.

---

## 1. Dataset survey

All four datasets below were checked live against the current HuggingFace Hub state
(2026-07-13). HF's `datasets` library has been tightening script-based loading (`datasets`
≥ 3.0 rejects loading scripts by default — `DatasetWithScriptNotSupportedError`), which bit
several classic NER datasets, CoNLL-2003 and WNUT-17 included. Concrete workarounds below.

### 1.1 CoNLL-2003 (4-class: PER, ORG, LOC, MISC)

The canonical repo `eriktks/conll2003` (formerly bare `conll2003`) still ships a Python loading
script, so plain `load_dataset("eriktks/conll2003")` on a recent `datasets` version raises
`DatasetWithScriptNotSupportedError` ("Dataset scripts are no longer supported, but found
conll2003.py"). It is **not gated**, just broken under strict script rejection. Three fixes,
in order of preference:

```python
# Preferred: pull the auto-converted parquet revision, no script execution at all.
from datasets import load_dataset
conll = load_dataset("eriktks/conll2003", revision="convert/parquet")

# Fallback if that revision is stale/unavailable: explicitly trust the script.
conll = load_dataset("eriktks/conll2003", trust_remote_code=True)
```

**Recommended for this project instead of either workaround**: load CoNLL-2003 through
`DFKI-SLT/cross_ner`'s `conll2003` config (see §1.2) — it is the *same* CoNLL-2003 data
(train 14,987 / validation 3,466 / test 3,684 sentences, matches the canonical split sizes
exactly), hosted as a plain Arrow/parquet dataset with no loading script, so it sidesteps the
gating issue entirely and lets us use one dataset repo for both the source (CoNLL) and target
(CrossNER domains) sides of the zero-shot transfer experiments in §1.4:

```python
from datasets import load_dataset
conll = load_dataset("DFKI-SLT/cross_ner", name="conll2003")
# splits: conll["train"] (14987), conll["validation"] (3466), conll["test"] (3684)
# fields: tokens (List[str]), ner_tags (List[int], BIO scheme over a shared 79-tag vocabulary)
```

Labels for this config, mapped down from BIO to the flat set used for calibration/eval:
`person`, `organisation`, `location`, `misc` (CrossNER's shared tag vocabulary spells ORG as
`organisation`, everything else matches standard CoNLL-03 PER/LOC/MISC semantics).

### 1.2 CrossNER (5 domains: AI, literature, music, politics, science)

```python
from datasets import load_dataset
ai         = load_dataset("DFKI-SLT/cross_ner", name="ai")
literature = load_dataset("DFKI-SLT/cross_ner", name="literature")
music      = load_dataset("DFKI-SLT/cross_ner", name="music")
politics   = load_dataset("DFKI-SLT/cross_ner", name="politics")
science    = load_dataset("DFKI-SLT/cross_ner", name="science")
```

No script, no gating, loads directly. Split sizes (sentences):

| domain     | train | validation | test |
|------------|------:|-----------:|-----:|
| conll2003  | 14987 |       3466 | 3684 |
| politics   |   200 |        541 |  651 |
| science    |   200 |        450 |  543 |
| music      |   100 |        380 |  456 |
| literature |   100 |        400 |  416 |
| ai         |   100 |        350 |  431 |

All six configs share one 39-entity-type tag vocabulary (`academicjournal`, `algorithm`,
`award`, `band`, `book`, `country`, `event`, `field`, `location`, `organisation`, `person`,
`product`, `programlang`, `researcher`, `task`, `university`, `misc`, … — full list in the
CrossNER paper/README), but each domain only realizes its own relevant subset in the actual
annotations, e.g.:
- **ai**: `field`, `task`, `product`, `algorithm`, `researcher`, `metrics`, `programlang`,
  `university`, `conference`, `country`, `location`, `organisation`, `person`, `misc`
- **music**: `musicalartist`, `musicgenre`, `song`, `band`, `album`, `musicalinstrument`,
  `award`, `event`, `country`, `location`, `organisation`, `person`, `misc`
- **politics**: `politician`, `politicalparty`, `election`, `country`, `organisation`,
  `person`, `event`, `location`, `misc`

Note: this repo's own `gliner/evaluation/evaluate_ner.py::get_for_all_path` already treats
`CrossNER_AI/literature/music/politics/science` as the canonical zero-shot benchmark group
(kept out of the training-average table) — consistent with using CrossNER here as our
zero-shot stress test too.

### 1.3 WNUT-17 (emerging/rare entities — the "hard zero-shot" set)

```python
from datasets import load_dataset
wnut = load_dataset("leondz/wnut_17")
# or, if the loading-script issue below bites: load_dataset("leondz/wnut_17", trust_remote_code=True)
```

`leondz/wnut_17` also ships a legacy loading script and can hit the same
`DatasetWithScriptNotSupportedError` depending on installed `datasets` version — same two
fixes as §1.1 (`trust_remote_code=True`, or pin an older `datasets`/use the parquet-converted
revision if present). Splits: train 3,394 / validation 1,009 / test 1,287 sentences. Labels
(6 classes, IOB2): `corporation`, `creative-work`, `group`, `location`, `person`, `product`.
These are exactly the "genuinely novel" categories relevant to §1.4 — `corporation`,
`creative-work`, `group`, `product` have no clean analogue in CoNLL-03's 4-class scheme.

### 1.4 Zero-shot label-transfer pairs (concrete)

Calibration and test *must* come from different label spaces to actually exercise the
zero-shot claim — calibrating and testing on the same 4 CoNLL classes only proves ordinary
split-conformal coverage, not that it survives GLiNER's genuine zero-shot setting. Three
pairs, in priority order for Phase 2:

| # | Calibrate on (source, seen types) | Test on (target, unseen types) | Why |
|---|---|---|---|
| **A (primary)** | CoNLL-2003 val split via `cross_ner/conll2003` — `person, organisation, location, misc` | WNUT-17 test split — `corporation, creative-work, group, location, person, product` | Newswire → noisy/social text; 4/6 target types have no CoNLL analogue (`corporation`, `creative-work`, `group`, `product`); `location`/`person` partially overlap, giving a built-in "easy vs. hard subset" contrast inside one run. |
| **B** | CoNLL-2003 val split — 4 classes | CrossNER **AI** test split — `field, task, product, algorithm, researcher, metrics, programlang, university, conference, country, location, organisation, person, misc` | Newswire → technical domain; almost fully disjoint type vocabulary. |
| **C (bonus, domain-only)** | CrossNER **politics** (train+validation pooled, ~741 sentences) | CrossNER **music** test split (456 sentences) | Isolates domain-transfer effect on its own, without CoNLL's comparatively "easy" newswire text as a confound. |

Report all three; A is the headline number for `design.md` / any eventual PR writeup because
WNUT-17 is the community's standard "hard zero-shot" set and the type mismatch is largest.

---

## 2. Split strategy

GLiNER is used **frozen** — no fine-tuning anywhere in this evaluation. That collapses the
usual conformal trio (proper-train / calibration / test) to two roles:

- **Score function** = the frozen pretrained checkpoint itself. There is no "proper training
  set" step at all; whatever GLiNER learned during its own pretraining is fixed input, not
  something this evaluation touches or re-splits.
- **Calibration set** = held-out *labeled* sentences (gold spans + types) used **only** to
  compute the conformal quantile/threshold.
- **Test set** = a separate held-out labeled set used **only** to measure whether the
  resulting coverage actually holds. Never used to pick the threshold.

**Recommended checkpoint**: `gliner-community/gliner_small-v2.5`
(https://huggingface.co/gliner-community/gliner_small-v2.5) — verified live: Apache-2.0,
`pytorch_model.bin` is 664,140,326 bytes (fp32, ≈166M params, matches the published GLiNER
"small" spec — DeBERTa-v3-small backbone), and it also ships `model.fp16.safetensors` /
`model.bf16.safetensors` (~332MB) for a lighter CPU download. It's the actively-maintained
community successor to `urchade/gliner_small-v2.1` (same architecture/size, 610,652,234-byte
fp32 checkpoint, last updated 2024) and is small enough for CPU-only test runs of the size
this protocol needs (hundreds to low-thousands of sentences per eval pass). Do not use
`gliner_medium`/`gliner_large`/`gliner_xxl` for Phase 2's default CI-style runs — reserve
those for an optional final "does the guarantee still hold on a bigger backbone" sanity check.

```python
from gliner import GLiNER
model = GLiNER.from_pretrained("gliner-community/gliner_small-v2.5")
```

### 2.1 Calibration-set-size floor (the degenerate-n problem)

Split conformal's quantile is the `⌈(n+1)(1-α)⌉`-th order statistic of `n` calibration
nonconformity scores (Angelopoulos & Bates correction). This is only defined when that rank is
`≤ n`. Solving `⌈(n+1)(1-α)⌉ ≤ n` gives the minimum usable `n` per α:

| α | minimum n (rank first ≤ n) | what happens below it |
|---|---:|---|
| 0.20 | 4 | rank `> n` → quantile undefined; must fall back to a trivial/`∞`-augmented set |
| 0.10 | 9 | same failure mode |
| 0.05 | 19 | matches the brief's own example: n=10, α=0.05 needs rank 11 > 10 available points |

At exactly the minimum n, the quantile equals the single largest observed calibration score —
mathematically valid but maximally conservative (huge/degenerate prediction sets, near-zero
efficiency). Phase 2's calibrator **must** raise a clear error (not silently degrade) if
`n_calib < ⌈1/α⌉`, and the calibration-set-size sensitivity sweep (§3.4) exists precisely to
show where, above that hard floor, coverage/efficiency actually stabilize in practice —
expect that to be well above the mathematical minimum (rule of thumb from the literature:
`n ≥ 100–200` for stable variance at α=0.1, more for smaller α).

### 2.2 Concrete split procedure

For **in-domain** runs (calibrate and test on the same dataset's label space, e.g. CoNLL val
→ CoNLL test, or WNUT val → WNUT test): use the dataset's native `validation` split entirely
for calibration and native `test` split entirely for test — they are already disjoint by
construction, no extra shuffling needed for the headline numbers. For the seeded-trial
re-splits used to build confidence intervals (§3.1, §3.4), pool `validation + test`, then for
each of the `T` trials draw a fresh random partition of that pool into a calibration subset of
size `n_calib` and a test subset of size `n_test` (fixed, e.g. all remaining pooled examples),
**with a different seed per trial**, sampling without replacement within a trial.

For **zero-shot transfer** runs (§1.4): calibration is drawn *only* from the source dataset
(e.g. CoNLL val), test is drawn *only* from the target dataset (e.g. WNUT test). These pools
never mix — there is no sense in which they could be exchangeable with each other (different
label spaces), which is exactly the point: the experiment measures how much coverage degrades
when the calibration/test exchangeability assumption is deliberately violated by a domain/type
shift, not whether it holds under a fair split.

**Hard invariant, everywhere**: calibration and test sets must be disjoint, and coverage/
efficiency numbers must be computed only on the test set. Reusing calibration examples to also
report "coverage" is a biased, overfit estimate — the calibration set is exactly the set the
threshold was tuned to satisfy by construction, so its empirical coverage will trivially sit
at or above `1-α` regardless of whether the method generalizes. Phase 2's test suite should
include an explicit regression test that asserts calibration-set and test-set indices are
disjoint before any metric is computed, and a "canary" test that recomputing coverage on the
calibration set itself produces an implausibly high number (>> 1-α) to catch anyone
accidentally wiring the same split into both roles.

---

## 3. Metrics protocol

Notation: test set has `M` sentences `x_1..x_M`, sentence `x_j` has gold entity set
`E_j = {(span_k, type_k)}`. `C(x_j)` is `ConformalGLiNER`'s output prediction set for `x_j` at
level `α` — whatever nonconformity score/guarantee mode Phase 1 chooses, it must expose a
per-sentence set of surviving `(span, type)` pairs; everything below is defined against that
interface only. `N = Σ_j |E_j|` is the total number of gold entities in the test set.

### 3.1 Empirical coverage vs. target `1-α`

```
Cov(α) = (1/N) * Σ_j Σ_{(span,type) ∈ E_j} 1[(span,type) ∈ C(x_j)]
```

i.e. the fraction of gold entities whose true `(span, type)` survived the conformal filter.
Compute for **α ∈ {0.05, 0.10, 0.20}** (fixed grid for all headline plots/tables — this range
covers the loose-to-strict guarantees practitioners actually ask for; narrower α needs bigger
`n_calib`, see §2.1, so 0.05 is the practical floor given realistic dataset sizes here).

**Confidence interval via seeded trials**: split-conformal coverage is a random variable over
the draw of the calibration set (for finite `n_calib`, `Cov(α)` marginalized over calibration
draws follows approximately `Beta(n_calib + 1 - l, l)` where `l = n_calib + 1 - ⌈(n_calib+1)(1-α)⌉`,
with a standard deviation on the order of `sqrt(α(1-α)/n_calib)`). Recompute `Cov(α)` for
**T = 100 trials** by default (re-splitting calibration/test per §2.2 with a new seed each
time), report mean ± std (and/or a percentile band) across trials. Use `T = 50` for fast local/
CI-smoke-test runs during development, and bump to `T = 200` for the final numbers that go into
`design.md`'s validation section or any PR writeup — 100 trials keeps the standard error of the
*mean* coverage estimate at roughly `1/10` of the single-trial std (e.g. at α=0.1, n_calib=200,
single-trial std ≈ 2.1%, so SE of the 100-trial mean ≈ 0.21%), which is precise enough to
visually distinguish "theory holds" from "off-by-one bug in the quantile rank" on the plot in
§4(a) without needing thousands of trials (compute budget matters here — this all needs to run
on CPU with a 166M-param model across multiple datasets × 3 α values × several `n_calib`
points).

### 3.2 Efficiency (average prediction-set size)

```
Eff(α) = (1/M) * Σ_j |C(x_j)|
```

Mean number of predicted `(span, type)` pairs per sentence surviving the filter — the "cost"
of the coverage guarantee. Report alongside the mean number of raw candidate `(span, type)`
pairs GLiNER scores *before* filtering (i.e. all span/type combinations above whatever floor
score the model assigns, pre-conformal-threshold) so Phase 2 can show the filter is doing real
work: if `Eff(α)` sits close to the raw candidate count, the conformal filter is trivially
passing nearly everything through and the coverage number is meaningless (degenerate — coverage
looks great because nothing was filtered, not because calibration worked). Plot efficiency
against α (§4b) to show the expected monotonic tradeoff: efficiency should shrink as α grows
(looser guarantee → smaller, more confident sets).

### 3.3 Per-class coverage breakdown

Same formula as §3.1, restricted to gold entities of one type `t`:

```
Cov(α, t) = (1/N_t) * Σ_j Σ_{(span,type) ∈ E_j, type=t} 1[(span,type) ∈ C(x_j)]
```

`N_t` = count of gold entities of type `t` in the test set. Compute for every type present in
the test set's label space, at a fixed α (see §4c). This is the diagnostic for whether
*marginal* (pooled, span-filter-only) conformal mode systematically under-covers rare/hard
types while over-covering common/easy ones (the marginal guarantee only promises coverage
averaged over the whole test set — it says nothing about any individual class) — the concrete
empirical motivation for building the Mondrian (per-class-calibrated) mode. Rare classes with
very small `N_t` (e.g. `corporation` in WNUT-17, which has few hundred instances) will have
high-variance per-class coverage estimates on any single split; report these bars with the
same 100-trial seeded-resampling approach as §3.1 so the per-class bars also carry an error bar,
not a point estimate that could just be noise.

### 3.4 Calibration-set-size sensitivity

Sweep `n_calib ∈ {50, 100, 200, 500, 1000}`, holding α fixed (run this sweep at α = 0.1 as the
default; optionally repeat at 0.05/0.2 if compute allows), each point averaged over the same
`T`-trial reseeding as §3.1. Report mean ± std of `Cov(α)` per `n_calib` (§4d).

**Dataset sizing note** (grounds this in what's actually available, per §1's split-size
table): use **CoNLL-2003** (`cross_ner/conll2003`, validation split alone has 3,466 sentences
≈ several thousand gold entities) or **WNUT-17** (train+validation pooled ≈ 4,400 sentences)
for this sweep — both comfortably support `n_calib` up to 1000 gold entities with room left
over for a same-size-or-larger disjoint test pool. Do **not** run the full `{50...1000}` grid
on the CrossNER domain splits (AI/literature/music/politics/science) — their train+validation
pools are only ~450–900 sentences each, so `n_calib=1000` entities is infeasible or would
leave a near-empty test set; cap the CrossNER-domain version of this sweep at `n_calib ∈
{50, 100, 200}` and note the cap explicitly in any resulting plot/table rather than silently
truncating the grid. Expected result if the implementation is correct: variance shrinks
monotonically with `n_calib`, and the mean converges toward `1-α` from above (split conformal
is marginally *conservative* at finite `n`, so slight over-coverage at small `n_calib` is
expected and not itself a bug — under-coverage that doesn't shrink toward `1-α` as `n_calib`
grows would be the actual red flag).

---

## 4. Required plots (Phase 2 deliverables)

Exactly four plots, each tied to a metric above:

**(a) Coverage vs. α curve.** X-axis: α ∈ {0.05, 0.10, 0.20}. Y-axis: empirical `Cov(α)` from
§3.1, mean over `T` trials with a shaded confidence band (±1 std or a percentile band across
trials). Overlay a reference line `y = 1 - α` (i.e. the diagonal from (0.05, 0.95) to
(0.20, 0.80)). One such curve per dataset/pair from §1 (in-domain CoNLL, in-domain WNUT,
zero-shot pairs A/B/C) — either as small multiples or overlaid with a legend; small multiples
preferred once zero-shot pairs are included, since their curves are expected to sag visibly
below the reference line and that needs to be visually unambiguous, not hidden by overlap.

**(b) Prediction-set size / efficiency vs. α.** X-axis: same α grid. Y-axis: `Eff(α)` from
§3.2, same dataset/pair breakdown as (a), plotted alongside (or annotated with) the mean raw
pre-filter candidate count as a dashed reference line, so the reader can see the filter isn't
degenerate (§3.2's warning).

**(c) Per-class coverage bar chart at fixed α.** Fixed α = 0.10 (the middle of the grid).
X-axis: gold entity type (one bar per type present in the test set). Y-axis: `Cov(α=0.1, t)`
from §3.3, with error bars from the trial resampling, and the `y = 0.9` reference line. Run
once per dataset that has a meaningful multi-class breakdown (CoNLL 4-class, WNUT-17 6-class,
each CrossNER domain's ~10–14 realized types) — this is the plot that motivates Mondrian mode,
so it should be produced for at least one in-domain case and one zero-shot-transfer case
(pair A) to show both "does marginal mode under-cover rare in-domain classes" and "does it get
worse under label shift."

**(d) Calibration-set-size sensitivity curve.** X-axis: `n_calib` ∈ {50, 100, 200, 500, 1000}
(capped at {50, 100, 200} for CrossNER domains per §3.4). Y-axis: mean `Cov(α=0.1)` with a
std/error-bar band, plus the `y = 0.9` reference line. This is the plot that validates finite-
sample theory isn't being silently broken by an implementation bug (e.g., an off-by-one in the
`⌈(n+1)(1-α)⌉` rank, or accidentally calibrating and testing on overlapping indices) — variance
should visibly shrink left-to-right and the mean should hug 0.9 (with the always-conservative
slight-over-coverage caveat from §3.4) rather than drift further from it as `n_calib` grows.

---

## 5. Summary table Phase 2 should produce

One row per (dataset/pair, α) combination, minimum columns: `dataset_pair`, `alpha`, `n_calib`,
`n_test`, `n_trials`, `coverage_mean`, `coverage_std`, `efficiency_mean`, `raw_candidates_mean`.
This is the flat table that both feeds plots (a)/(b) directly (group by `dataset_pair`, plot
vs. `alpha`) and gives a quick pass/fail read (`coverage_mean >= alpha_target - 2*coverage_std`
as a sanity check) before spending time on the more detailed per-class/per-size breakdowns.
