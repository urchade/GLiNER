# Prior Art & Competitive Landscape: Conformal Prediction for NER / GLiNER

Phase 0, Agent C deliverable. Web research only, no code written. Compiled 2026-07-13.

Scope: (1) survey general-purpose conformal-prediction Python libraries for
license and NER/structured-prediction support, (2) search specifically for
conformal-prediction-for-NER prior art, including two named arXiv papers,
(3) test the novelty claim that no mainstream NER library/framework ships
built-in conformal coverage guarantees.

---

## 1. Conformal prediction libraries

### MAPIE (scikit-learn-contrib)
- **URL:** https://github.com/scikit-learn-contrib/MAPIE
- **License:** BSD-3-Clause (permissive). Confirmed via `pyproject.toml` and README.
- **What it provides:** Split conformal prediction, conformalized quantile
  regression, jackknife+/CV+, and **Conformal Risk Control (CRC)** for
  classification and regression. scikit-learn-compatible API. Companion
  paper: arXiv:2207.12274.
- **NER / structured-prediction support:** None. Assumes a fixed set of
  scalar/single-label classification or regression outputs. No sequence
  labeling, span, or variable-cardinality-set primitives.
- **Verdict:** Safe to learn from and adapt algorithmic patterns or even
  snippets of code from, with attribution, given the permissive BSD-3-Clause
  license. Its CRC (conformal risk control) machinery is the most relevant
  building block for a future "control expected miss-rate over extracted
  entities" mode, even though it isn't structured-prediction-aware out of
  the box.

### crepes (Henrik Boström)
- **URL:** https://github.com/henrikbostrom/crepes (extension:
  https://github.com/predict-idlab/crepes-weighted)
- **License:** BSD-3-Clause (permissive).
- **What it provides:** Standard **and Mondrian** conformal classifiers;
  standard, normalized, and Mondrian conformal regressors and conformal
  predictive systems. `crepes-weighted` extends this to weighted CP for
  covariate shift. Companion papers at COPA 2022/2024.
- **NER / structured-prediction support:** None directly — single-label
  classification and scalar regression only.
- **Verdict:** Safe to reuse ideas/code from (BSD-3-Clause). Its **Mondrian
  category mechanism** (partitioning calibration by a discrete attribute) is
  the closest conceptual match to what a per-entity-type calibrated
  GLiNER would need (one Mondrian category per candidate label), and is
  worth studying closely even though no code will transfer directly to a
  span/sequence setting.

### TorchCP (ml-stat-Sustech)
- **URL:** https://github.com/ml-stat-Sustech/TorchCP
- **License:** **LGPL-3.0** (repo LICENSE file confirmed verbatim: "GNU
  LESSER GENERAL PUBLIC LICENSE, Version 3"). This is copyleft, not
  permissive. **Flag: reuse-risky.** Not GPL/AGPL-level viral, but LGPL
  still requires that modifications to LGPL-covered source remain
  LGPL-licensed and (if distributed) source-available. Do not copy TorchCP
  source into the Apache-2.0-licensed GLiNER-Robust codebase. Reading it for
  algorithmic ideas (the math/algorithms themselves are not copyrightable)
  and citing it is fine; copying code is not.
- **What it provides:** PyTorch-native CP toolbox with GPU acceleration,
  score functions (LAC, APS, SAPS, RAPS), CP-aware training, and support for
  classification, regression, GNNs, and an LLM/"conformal language modeling"
  module (JMLR paper, arXiv:2402.12683).
- **NER / structured-prediction support:** None found. The "LLM" module
  targets generation/selection tasks (conformal language modeling à la
  Quach et al.), not token classification or sequence labeling.
- **Verdict:** Useful reference for score-function design and GPU-efficient
  calibration patterns, but any code reuse must respect LGPL-3.0 — safest
  path is independent reimplementation citing the ideas, not copy-paste.

### Fortuna (AWS Labs)
- **URL:** https://github.com/awslabs/fortuna
- **License:** Apache-2.0 (permissive, and license-compatible with
  GLiNER's own Apache-2.0 licensing).
- **What it provides:** Unified interface for uncertainty quantification —
  conformal methods plus Bayesian inference — for classification and
  regression, with three usage modes (from uncertainty estimates, from
  model outputs, from Flax models). Companion paper: arXiv:2302.04019.
- **NER / structured-prediction support:** None found; text-classification
  benchmarks are mentioned but not token-level/NER tasks.
- **Status:** **Archived by AWS on 2025-04-23 — no longer maintained.**
- **Verdict:** Freely reusable license-wise, but low practical value given
  it's archived and has no structured-prediction support.

### nonconformist (donlnz)
- **URL:** https://github.com/donlnz/nonconformist
- **License:** MIT (permissive), per GitHub's license badge.
- **What it provides:** One of the earliest general Python CP
  implementations — inductive conformal prediction, ACP, exchangeability
  testing, interpolated p-values, Venn/Venn-ABERS predictors, built as a
  scikit-learn extension.
- **NER / structured-prediction support:** None; classification/regression
  only. Project is effectively unmaintained (docs "severely deprecated").
- **Verdict:** Freely reusable license-wise; mostly of historical/reference
  interest at this point.

### PUNCC (deel-ai / Thales-affiliated)
- **URL:** https://github.com/deel-ai/puncc
- **License:** MIT (permissive).
- **What it provides:** Regression (split CP, CQR, CV+, EnbPI...),
  classification (LAC, classwise LAC, APS, RAPS), **object detection
  (box-wise split conformal object detection)**, and anomaly detection
  (SplitCAD). Compatible with scikit-learn/PyTorch/TensorFlow. Companion
  paper: Mendil et al., PMLR v204.
- **NER / structured-prediction support:** No NER/sequence-labeling module,
  but its **object-detection module already handles variable-cardinality,
  variable-size structured outputs (bounding boxes)** — conceptually the
  nearest existing analog to "conformalize a variable-size set of extracted
  spans," even though the modality (boxes vs. text spans) differs.
- **Verdict:** Most architecturally relevant of the surveyed libraries for
  designing a GLiNER conformal wrapper. MIT license makes code-level
  borrowing (with attribution) low-risk. Its box-wise CP design is worth
  studying as a template for span-wise CP.

### Other libraries encountered (lower relevance, noted for completeness)
- **aangelopoulos/conformal-prediction** — educational/reference
  implementations (classification, regression) by a leading CP researcher;
  no NER content. https://github.com/aangelopoulos/conformal-prediction
- **gchers/cpy** — small classic Python CP implementation, unmaintained.
  https://github.com/gchers/cpy
- **team-daniel/Conformal_Prediction_Algs** — tutorial-style catalogue of CP
  algorithms across classification/regression/time-series/risk-aware tasks;
  no NER/sequence-labeling content.
- **valeman/awesome-conformal-prediction** — the standard curated
  bibliography for the field
  (https://github.com/valeman/awesome-conformal-prediction). Checked
  directly for NLP entries: it lists an "Uncertainty estimation in NLP"
  tutorial (Schuster & Fisch), a paraphrase-detection CP talk, and a
  Venn-ABERS calibration-for-NLU paper — but **has no dedicated entry for
  NER, sequence labeling, or token classification** as of this check. This
  independently corroborates that the field-standard bibliography does not
  consider CP-for-NER a populated sub-area yet.

**Cross-library confirmation:** every general-purpose CP library surveyed
(MAPIE, crepes, TorchCP, Fortuna, nonconformist, PUNCC) assumes
single-label classification or scalar/interval regression as its unit of
prediction. The closest thing to "structured" support anywhere is PUNCC's
object-detection module (bounding boxes) and TorchCP's GNN/LLM modules —
neither is sequence labeling or NER. **The premise that mainstream CP
libraries don't handle NER/sequence labeling is confirmed.**

---

## 2. Conformal-NER-specific prior art (Task 2)

### arXiv:2601.16999 — "Uncertainty Quantification for Named Entity
Recognition via Full-Sequence and Subsequence Conformal Prediction"
- **Authors:** Matthew Singer, Srijan Sengupta, Karl Pazdernik.
  Submitted 2026-01-13. Subjects: cs.CL, cs.LG, stat.ML.
- **URL:** https://arxiv.org/abs/2601.16999 (HTML:
  https://arxiv.org/html/2601.16999)
- **License of the paper itself:** CC BY-NC-ND 4.0 — non-commercial,
  no-derivatives. This restricts reusing the paper's *text/figures*
  verbatim or distributing derivative works of the paper, but does **not**
  restrict independently reimplementing the underlying algorithms/formulas
  (ideas and math are not copyrightable) as long as it's an independent
  implementation, properly cited, not a copy of their expression.
- **Code:** No GitHub link, code release, or supplementary repository found
  anywhere — not in the abstract page, not in the HTML full text, not on
  paperswithcode-style search. **No implementation appears to exist
  publicly.**
- **What it actually does:** Extends split conformal prediction to
  CRF-based sequence-labeling NER, producing prediction sets of
  *full-sentence label sequences* (and subsequence/entity-level variants)
  guaranteed to contain the true labeling at a chosen confidence level.
  Proposes three base non-conformity scores (probability-deviation,
  cumulative-probability, rank-based) plus entity-level variants and two
  combination strategies (Naive Intersection, Conditional/nested), and
  compares against a RAPS-style penalized baseline.
- **Models evaluated:** Babelscape (multilingual BERT/WikiNEuRal), dslim
  BERT-base, Jean-Baptiste RoBERTa, TNER RoBERTa-large — all **CRF-based,
  closed/fixed label-set, supervised** NER models. **No zero-shot or
  open-label-set model (GLiNER or otherwise) is evaluated.**
- **Datasets:** CoNLL++, CoNLL-Reduced, WikiNEuRal.
- **Label-set assumption:** Fixed, closed label set with IOB2 tagging,
  known at calibration time. Explicitly does not address open-set/zero-shot
  labeling.

### arXiv:2605.18812 — "PASC: Pipeline-Aware Conformal Prediction with
Joint Coverage Guarantees for Multi-Stage NLP and LLM Pipelines"
- **Author:** Varun Kotte. Submitted 2026-05-12.
- **URL:** https://arxiv.org/abs/2605.18812 (HTML:
  https://arxiv.org/html/2605.18812v1)
- **License of the paper itself:** CC BY 4.0 — permissive, reuse with
  attribution is fine including for commercial purposes.
- **Code:** No GitHub link or code release found on the abstract page,
  HTML full text, or reproducibility section (the paper's "Reproducibility
  Notes" describe experimental protocol — five calibration/test seeds —
  but link to no repository). **No implementation appears to exist
  publicly.**
- **What it actually does:** Reduces *joint* coverage across a multi-stage
  pipeline (e.g., NER → entity disambiguation → entity typing, or
  retriever → reader) to a single scalar CP problem on the max
  nonconformity score across stages, giving a finite-sample joint-coverage
  guarantee tighter than a Bonferroni union bound. On a 3-stage
  NER→NED→typing pipeline over CoNLL-2003 it reports 96.4% end-to-end
  coverage vs. 93.4% (Bonferroni) and 86.5% (independent per-stage CP) at
  equal set sizes.
- **NER component used:** `dslim/bert-base-NER`, a standard supervised
  closed-label-set BERT model. The "zero-shot" component in this pipeline
  is a downstream *entity-typing* classifier (RoBERTa-large zero-shot
  classifier) applied to spans already found by the supervised NER stage —
  **not** a zero-shot entity-recognition/extraction model. GLiNER-style
  zero-shot span extraction is not addressed.

### Conformal Structured Prediction (ICLR 2025)
- **Authors:** Botong Zhang, Shuo Li, Osbert Bastani. arXiv:2410.06296.
  https://arxiv.org/abs/2410.06296 ; OpenReview:
  https://openreview.net/forum?id=mKfmLQXP6J
- **What it does:** First general framework for CP over structured label
  spaces representable as a DAG (e.g., hierarchical coarse-to-fine image
  labels), using integer programming to build structured prediction sets
  that implicitly encode large label sets compactly.
- **Relevance to NER:** Theoretically adjacent (structured outputs,
  variable-size prediction sets) but not built for or evaluated on
  sequence labeling / span extraction, and does not target NER. No
  GitHub/code link found in the abstract, comments, or search.

### Other adjacent work found
- **CONFIDE** (conformal prediction for fine-tuned encoder LMs, applies CP
  to BERT/RoBERTa [CLS]/hidden-state embeddings) — targets sentence-level
  text classification, not token-level NER.
- **arXiv:2604.08885**, "Uncertainty-Aware Transformers: Conformal
  Prediction for Language Models" — general LM uncertainty, not
  NER-specific.
- General search across GitHub topics (`named-entity-recognition`, `ner`,
  `entity-recognition`, `nested-named-entity-recognition`) and Hugging Face
  Spaces turned up **zero repositories or Spaces at the intersection of
  "conformal" and "NER"** — only unrelated NER repos (NeuroNER, deep_ner,
  DeepPavlov NER, BERT-NER, etc.) and unrelated CP repos. No toy, partial,
  or abandoned "conformal NER" implementation was found anywhere on GitHub,
  Hugging Face, or the general web.

**Summary of Task 2:** Two theory papers (both 2026, both extremely
recent) establish that CP-for-NER is an active research question, but
**neither has released code**, and **neither addresses zero-shot /
open-label-set NER** — both assume a closed, fixed label set with a
supervised, task-specific NER model (CRF-tagger or fine-tuned BERT). No
implementation targeting GLiNER, or any zero-shot/generalist NER model,
was found anywhere.

---

## 3. Novelty verdict

**Claim under test:** "No mainstream NER library or framework (spaCy,
GLiNER itself, Flair, HuggingFace transformers token-classification
pipelines, AllenNLP, etc.) currently ships built-in conformal prediction /
calibrated coverage guarantees for entity extraction."

**Verdict: holds up, with one caveat about very recent academic (not
library) prior art.**

Evidence for the claim:
- No evidence found of conformal prediction in spaCy, Flair, or the
  HuggingFace `transformers` token-classification pipeline
  (`src/transformers/pipelines/token_classification.py` and the associated
  docs describe only raw softmax/argmax outputs, no CP machinery).
  AllenNLP was checked via the same search sweep with no hits either (and
  is itself a largely inactive project at this point).
- **GLiNER itself has no calibrated uncertainty**, only a raw sigmoid
  confidence score per (span, label) pair with a manually-tuned decision
  threshold (default 0.5, commonly retuned to 0.3–0.5 per community
  guidance). This is explicitly *not* a coverage guarantee — it's an
  uncalibrated heuristic cutoff. Corroborating evidence from GLiNER's own
  issue tracker: issue #324 (transformers v5.0.0 causes uniformly low/
  meaningless scores), issue #192 (label ordering changes confidence
  scores — a symptom of exactly the kind of miscalibration conformal
  prediction is designed to correct), and issue #69 (a 2023 feature request
  just to *expose* confidence values at all). This is strong, concrete
  evidence GLiNER's current scoring is uncalibrated and unstable — the
  opposite of a coverage guarantee.
- No general-purpose CP library (Section 1) provides NER/sequence-labeling
  support out of the box.
- No GitHub/Hugging Face implementation combining "conformal" and "NER"
  was found anywhere (Section 2).

Caveat / what would undermine a *stronger* version of the novelty claim:
- Two 2026 arXiv papers (2601.16999 and 2605.18812) already establish the
  *theory* of conformal prediction for sequence-labeling NER and for
  multi-stage NLP pipelines including NER, with worked non-conformity
  scores and empirical coverage results. If the project's claim were "no
  one has ever formulated conformal prediction for NER," that claim would
  be **false** — this ground has been broken academically, twice, within
  the last several months. The accurate framing is narrower: **no shipped,
  usable, open-source implementation exists**, and **no zero-shot/
  open-label-set model has been addressed** by any of this prior art.

How a GLiNER-specific `ConformalGLiNER` would still differ / add value
even given this prior art:
1. **Zero-shot / open label-set generality.** Every piece of NER-CP prior
   art found (2601.16999, 2605.18812, and all surveyed libraries) assumes
   a fixed, closed label set fitted at training time. GLiNER's defining
   feature is inference-time arbitrary label sets; a conformal wrapper that
   preserves finite-sample coverage guarantees *across arbitrary
   user-supplied label sets*, without per-label-set retraining or
   recalibration, is not something any prior art attempts.
2. **No public code exists for CP-on-NER at all**, closed-set or
   otherwise — an actual working, released implementation is itself a
   contribution regardless of the theoretical novelty question.
3. **Integration depth.** A `ConformalGLiNER` wrapper integrated directly
   into an actively-maintained, widely-used zero-shot NER library (GLiNER,
   Apache-2.0, active GitHub community) is a materially different
   contribution from a standalone research-code artifact evaluated only on
   CoNLL-style closed-set benchmarks.
4. **Guarantee modes.** MAPIE-style Conformal Risk Control (expected
   miss-rate control) and PUNCC-style variable-cardinality set conformal
   (object-detection-style box-wise CP, here adapted to spans) are both
   architecturally closer to what a production span-extraction system
   needs than the full-sequence-labeling-set framing in 2601.16999; a
   GLiNER wrapper could combine ideas from both traditions (Mondrian
   per-label-type calibration from `crepes`, box/span-wise CP from
   `PUNCC`, CRC from `MAPIE`) in a way none of the individual pieces of
   prior art do on their own.

**Bottom line:** the strict "no library ships this" claim is well
supported and should be stated as-is. The stronger "no one has thought of
conformal prediction for NER" framing does **not** hold — cite
2601.16999 and 2605.18812 as related work — but neither paper ships code,
neither addresses zero-shot/GLiNER-style open label sets, and no
implementation of any kind was found publicly. The project's actual white
space is: a released, zero-shot-capable, GLiNER-integrated conformal
wrapper — not "the first conformal NER method" (that claim would not
survive scrutiny) but plausibly "the first released one, and the first
that works with open/zero-shot label sets."

---

## License risk summary (quick reference)

| Library | License | Risk | Notes |
|---|---|---|---|
| MAPIE | BSD-3-Clause | Safe | permissive |
| crepes | BSD-3-Clause | Safe | permissive; Mondrian CP relevant |
| PUNCC | MIT | Safe | permissive; box-wise CP relevant |
| Fortuna | Apache-2.0 | Safe | permissive; archived/unmaintained |
| nonconformist | MIT | Safe | permissive; unmaintained |
| TorchCP | **LGPL-3.0** | **Reuse-risky** | copyleft; do not copy source into Apache-2.0 codebase, ideas/citation only |
| arXiv 2601.16999 (paper) | CC BY-NC-ND 4.0 | Caution on text/figures | cite and reimplement independently; do not reproduce paper text/figures; no code exists to "reuse" anyway |
| arXiv 2605.18812 (paper) | CC BY 4.0 | Safe | permissive paper license; no code exists to reuse anyway |
