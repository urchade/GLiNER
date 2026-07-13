# Conformal Prediction for GLiNER

## Stop using `threshold=0.5`

GLiNER scores every candidate `(span, type)` pair with an independent sigmoid and, by
default, keeps anything above `threshold=0.5`. That number is a convenient default, not
a statistical guarantee — nothing about it tells you what fraction of true entities you're
actually going to miss, and nothing calibrates it to your data, your entity types, or your
risk tolerance.

`gliner.conformal.ConformalGLiNER` replaces that arbitrary cutoff with a threshold
**calibrated on a held-out labeled set**, backed by finite-sample, distribution-free
guarantees from the conformal prediction literature. Instead of "keep anything above 0.5,"
you get to ask for something like:

> "Calibrate a threshold such that, on average, I miss fewer than 10% of true entities."

...and get a number back that is provably true (under the assumptions below), not tuned by
eyeballing a validation set.

This is a new, additive module — it wraps a `GLiNER` model without modifying it, and has
zero effect on the standard `predict_entities`/`inference` API unless you opt in.

## Quickstart

```python
from gliner import GLiNER
from gliner.conformal import ConformalGLiNER

model = GLiNER.from_pretrained("gliner-community/gliner_small-v2.5")
cg = ConformalGLiNER(model)

# calib_data: held-out labeled sentences, same format as GLiNER's own training/eval data
calib_data = [
    {"tokenized_text": ["Apple", "was", "founded", "by", "Steve", "Jobs", "."],
     "ner": [[0, 0, "organization"], [4, 5, "person"]]},
    # ... more labeled examples, ideally 100+ per entity type
]

cg.calibrate(calib_data, alpha=0.1, mode="risk_control")

entities = cg.predict_entities(
    "Netflix was founded by Reed Hastings.", ["organization", "person"]
)
# each entity carries a "conformal" field:
# {"text": "Netflix", "label": "organization", "score": 0.99,
#  "conformal": {"mode": "risk_control", "alpha": 0.1, "calibrated": True}}
```

## The three guarantee modes

All three are calibrated from the *same* raw span scores GLiNER already computes — no
extra forward pass, no architecture change.

### `"risk_control"` (the default, and the one to reach for first)

Bounds the **expected fraction of true entities you miss**, on average across sentences:

```python
cg.calibrate(calib_data, alpha=0.1, mode="risk_control")
```

With `alpha=0.1`: *"on average, ConformalGLiNER misses fewer than 10% of the true entities
in a sentence."* This is the mode to use for compliance/PII-style requirements ("we provably
miss under 5% of PII entities on average") — it directly controls the thing you usually
actually care about (missed entities), rather than an indirect proxy.

### `"span_filter"`

Bounds coverage **per entity occurrence**: *"for a randomly drawn true entity of a given
type, it's included in the output with probability at least 90%."* This is the more classical
conformal-prediction framing (closer to "prediction sets" in the broader literature) and is
a good default if you want the simplest possible mental model, or if per-entity behavior
matters more to you than the sentence-level missed-entity rate.

### `"mondrian"`

Same as `span_filter`, but calibrated **separately per entity type**, so a common type
(e.g. `person`) can't "subsidize" a rare type (e.g. `chemical_compound`) — each type gets
its own guarantee, at the cost of needing enough calibration examples of *every* type you
care about (see Limitations).

```python
cg.calibrate(calib_data, alpha=0.1, mode="mondrian")
```

## Validating and saving a calibration

```python
report = cg.coverage_report(test_data)  # test_data must be disjoint from calib_data
print(report["overall_coverage"], report["per_type_coverage"])

cg.save_calibration("calibration.json")
cg2 = ConformalGLiNER.load_calibration("calibration.json", model)
```

`test_data` must not overlap with `calib_data` — reusing calibration examples to also
report coverage produces an inflated, meaningless number, since the threshold was tuned
to fit exactly that data.

## Limitations — read this before you trust a number

This section exists because a calibrated-looking number is more dangerous than an
obviously-arbitrary one if the calibration doesn't actually apply.

**The guarantee only covers entity types you actually calibrated on, with enough data.**
Every mode requires roughly `⌈1/alpha⌉` calibration occurrences of a type before it gets a
real threshold (concretely: ~19 for `alpha=0.05`, ~9 for `alpha=0.1`, ~4 for `alpha=0.2`).
If you ask `predict_entities` for a type that wasn't adequately represented in calibration,
`ConformalGLiNER` will:
- warn you loudly,
- fall back to GLiNER's original uncalibrated `p > 0.5` behavior for that type only,
- flag every entity of that type `"conformal": {"calibrated": False}` in the output.

It will never silently blend an unguaranteed number into a guaranteed-looking one.

**This is *not* a zero-shot guarantee for arbitrary novel entity types.** This is the most
important limitation and the reason for the point above. Conformal prediction's guarantee
relies on *exchangeability* between your calibration data and what you query at inference
time. If you calibrate on `{person, organization, location}` and then ask for
`chemical_compound` — a type with **zero** calibration occurrences — there is no
mathematical sense in which that query is exchangeable with your calibration set, and no
theorem (here or in the broader conformal-prediction literature) licenses a coverage claim
for it. This isn't a corner case we haven't gotten around to handling; it's a structural
fact about what conformal prediction can prove: split-conformal validity requires the
calibration and test points to be exchangeable, and a type with zero calibration
occurrences was never part of that exchangeable draw at all — there is no rank statistic to
compute a quantile from. GLiNER's flagship feature is arbitrary inference-time label
sets — this module deliberately does *not* pretend to extend a statistical guarantee to
labels outside what you actually calibrated on. If your workflow requires open-vocabulary
guarantees, this isn't (yet) the tool for that; treat the raw sigmoid score as the
heuristic it always was for those types.

**Domain shift still degrades things, even for calibrated types.** Calibrating on newswire
text and deploying on social media text, for a type name that's nominally the same
(`location` means the same thing in both), is a milder violation of exchangeability than a
genuinely novel type — but it's still a violation. Expect coverage to visibly sag if your
deployment distribution meaningfully differs from your calibration distribution. A concrete
measurement: calibrating on CoNLL-2003 and measuring coverage on WNUT-17, the shared-vocabulary
types (`location`, `person`) still reach 0.87–0.98 coverage across α ∈ {0.05, 0.1, 0.2} — good,
but visibly softer than the ~0.90–0.95 in-domain numbers — while WNUT-17's own types with zero
CoNLL-2003 analogue (`corporation`, `creative-work`, `group`, `product`) sit at a flat ~0.55
regardless of α, exactly the unguaranteed number you'd expect from a raw uncalibrated cutoff.
Reproduce via `scripts/conformal_validation.py`.

**`mondrian` mode costs calibration data linearly in the number of types.** Every type
needs its own `~1/alpha`-sized calibration pool; with a fixed calibration budget, more
types means either fewer types getting a real (non-degenerate) threshold, or a looser
`alpha`.

**Nested-span decoding is applied after the conformal filter, not before.** The coverage/
risk guarantee is computed against the pre-overlap-resolution candidate set; the final
`predict_entities` output additionally applies GLiNER's usual flat/nested-NER overlap
resolution as a threshold-independent post-processing step. This is a deliberate design
choice needed to keep the Conformal Risk Control guarantee mathematically valid —
applying overlap resolution *before* defining the
calibrated set would break the nesting property the risk-control proof depends on.

**Scope: span-mode models only.** `ConformalGLiNER` currently supports GLiNER's span-mode
architectures (`UniEncoderSpanGLiNER`, `BiEncoderSpanGLiNER` — the default
`span_mode="markerV0"` configuration, and what most published GLiNER checkpoints use).
Token-mode, generative-decoder, and relation-extraction variants apply their confidence
threshold *inside* the forward pass to prune candidates, so the raw-score interception
this module relies on doesn't give the full candidate universe for those architectures;
using it there would silently understate the true, uncalibrated candidate pool rather than
producing a valid guarantee, so it's explicitly unsupported (raises `NotImplementedError`)
rather than quietly wrong.
