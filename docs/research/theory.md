# Conformal Guarantees for GLiNER: Theory Foundations

Phase 0, Agent B deliverable. Pure theory/literature research; no code written, no other file
modified. Compiled 2026-07-13.

**Provenance note (read this first).** Sections marked **[FULL]** below are built from the
actual PDF text of the source (extracted with `pdftotext -layout` after downloading, then read
directly, equation-by-equation — not from an abstract or a lossy summary). Sections marked
**[ABSTRACT]** are reconstructed from the abstract plus secondary material only, because full-text
extraction was not attempted or not needed for that point. Every theorem, algorithm, and proof
quoted below was read from primary-source PDF text; where a first-pass automated fetch produced a
claim that could not be verified against the primary text on a second pass (this happened once,
noted explicitly in §6), that claim is flagged and discarded rather than silently kept.

| # | Paper | Status |
|---|---|---|
| 1 | Singer, Sengupta & Pazdernik, *Uncertainty Quantification for NER via Full-Sequence and Subsequence Conformal Prediction*, arXiv:2601.16999 (Jan 2026) | **[FULL]** — full text extracted from `arxiv.org/pdf/2601.16999`, all of Sections 1–8 plus proof appendix S1 read directly |
| 2 | Kotte, *PASC: Pipeline-Aware Conformal Prediction with Joint Coverage Guarantees for Multi-Stage NLP and LLM Pipelines*, arXiv:2605.18812 (May 2026) | **[FULL]** — full text extracted and read, including appendices A–G |
| 3 | Angelopoulos & Bates, *A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification*, arXiv:2107.07511 | **[FULL]** for Theorem 1 (marginal coverage), Appendix D proof, §4.1–4.3 (group-balanced, class-conditional, risk control) |
| 4 | Angelopoulos, Bates, Fisch, Lei & Schuster, *Conformal Risk Control*, arXiv:2208.02814 | **[FULL]** — Theorem 1, Theorem 2, Proposition 1, and their proofs read directly from the extracted PDF text |
| 5 | Vovk, Gammerman & Shafer, *Algorithmic Learning in a Random World* | **[ABSTRACT/secondary]** — not read directly; used only as the citation target that #3 and #4 both point to for the exchangeability-based coverage theorem and the Mondrian/class-conditional constructions (Vovk's original results, per Angelopoulos & Bates §4.1–4.2: "as first documented by Vovk in [14]"). This is sufficient for the stated purpose ("confirm the exchangeability framework") since #3/#4 restate and prove the relevant results with full rigor. |
| 6 | Zaratiana et al., GLiNER, arXiv:2311.08526 | **[ABSTRACT]** — skimmed for architecture grounding only, as instructed; cross-checked against this repo's own code (`gliner/decoding/decoder.py`, `gliner/modeling/span_rep.py`) via project memory, which confirms independent sigmoid scoring per (span, type) pair |
| 7 | GLiNER bi-encoder "Million-Label NER" paper, arXiv:2602.18487 | **[ABSTRACT]** — skimmed for architecture grounding only |

---

## 0. Notation and setup

Fix a joint sample space of (input, label) pairs. In the classical conformal literature this is
$(X,Y) \in \mathcal X \times \mathcal Y$. We will overload this once we get to NER, where a single
"input" is a sentence and the corresponding "label" is a *variable-size set of typed spans*, not a
scalar.

A **nonconformity score** is any measurable function $s: \mathcal X \times \mathcal Y \to \mathbb R$
with the convention that *larger* $s$ means *worse* agreement between $x$ and $y$ under the
trained model. Given a calibration set $\{(X_i,Y_i)\}_{i=1}^n$ and a miscoverage level
$\alpha \in (0,1)$, split conformal prediction outputs

$$
C(x) = \{y : s(x,y) \le \hat q\}, \qquad
\hat q = \mathrm{Quantile}\Big(\{s(X_i,Y_i)\}_{i=1}^n;\ \frac{\lceil (n+1)(1-\alpha)\rceil}{n}\Big).
$$

Everything below is either a special case or a direct generalization of this template.

---

## (i) The split-conformal marginal coverage guarantee, and why the $\lceil(n+1)(1-\alpha)\rceil/n$ correction is not optional

### Statement

**Theorem (split conformal marginal coverage; Vovk et al., restated as Theorem 1 in Angelopoulos &
Bates 2107.07511, and independently re-derived as Proposition 1 in 2601.16999, §S1.1, following
Gupta–Kuchibhotla–Ramdas).**
Let $(X_1,Y_1),\dots,(X_n,Y_n),(X_{n+1},Y_{n+1})$ be **exchangeable** random variables (in
particular this holds if they are i.i.d., which is the weaker practical assumption both source
papers state their result under, but exchangeability is all that is actually used in the proof).
Let $s$ be any fixed nonconformity score computed from a model trained on data *independent of*
(or, in the transductive case, symmetric in) the calibration and test indices, define

$$
\hat q = \mathrm{Quantile}\Big(\{s(X_i,Y_i)\}_{i=1}^n;\ \frac{\lceil (n+1)(1-\alpha)\rceil}{n}\Big)
$$

(the $\lceil(n+1)(1-\alpha)\rceil$-th smallest of the $n$ calibration scores), and
$C(x) = \{y : s(x,y) \le \hat q\}$. Then

$$
\mathbb P\big(Y_{n+1} \in C(X_{n+1})\big) \ \ge\ 1-\alpha. \tag{1}
$$

If additionally the scores $s(X_i,Y_i)$ have a continuous joint distribution (no ties, a.s.), the
guarantee is two-sided:

$$
1-\alpha \ \le\ \mathbb P\big(Y_{n+1}\in C(X_{n+1})\big) \ \le\ 1-\alpha+\frac{1}{n+1}. \tag{2}
$$

**What the probability is over.** This is the crux point to be precise about, since it is the
single most commonly misstated fact about conformal prediction. The probability in (1)/(2) is
**over the joint randomness of the calibration set and the test point together** — i.e. over the
draw of $(X_1,Y_1,\dots,X_n,Y_n,X_{n+1},Y_{n+1})$ as an exchangeable $(n{+}1)$-tuple. It is *not*
conditional on the realized calibration set. In particular:

- The guarantee is **marginal**, not **conditional**: it does not say
  $\mathbb P(Y_{n+1}\in C(X_{n+1}) \mid X_{n+1}=x) \ge 1-\alpha$ for a fixed $x$, nor does it say
  $\mathbb P(Y_{n+1}\in C(X_{n+1}) \mid \mathcal D_{\mathrm{cal}}) \ge 1-\alpha$ for a fixed
  realized calibration set $\mathcal D_{\mathrm{cal}}$ (that conditional statement is true only in
  expectation over re-draws of $\mathcal D_{\mathrm{cal}}$; for any *particular* calibration draw
  the conditional coverage is itself a random variable, distributed — for i.i.d. data and
  continuous scores — as $\mathrm{Beta}(n+1-l,\, l)$ where $l=\lceil (n+1)\alpha\rceil$; this
  detail is standard but not required by the task, flagged here only so "1$-\alpha$" is not
  over-interpreted).
- Exchangeability is what is *actually* used, not i.i.d. — this matters directly for us because
  training-set draws, model-fitting randomness, and calibration-set draws all need not be i.i.d.
  in the usual sense; they only need to be *exchangeable*, which is a weaker, permutation-symmetry
  condition. This is why the guarantee survives things like *stratified mixtures* of exchangeable
  populations (2601.16999 Theorem 1, see part (v) below) — "mixtures of exchangeable sequences
  remain exchangeable" (2601.16999, §4.3, verbatim) — but it does **not** survive genuine
  distribution shift between calibration and test (see part (vi)).

### Proof sketch (quantile lemma)

The proof given in both source papers is the standard "rank argument," and it is worth writing
out in full because every "does exchangeability hold here?" question we ask later reduces to
whether *this specific step* is licensed.

**Step 1 (reduce set-membership to a scalar-quantile event).** By construction,
$Y_{n+1} \in C(X_{n+1}) \iff s(X_{n+1},Y_{n+1}) \le \hat q$. So

$$
\mathbb P(Y_{n+1}\in C(X_{n+1})) = \mathbb P\big(s(X_{n+1},Y_{n+1}) \le \hat q\big).
$$

**Step 2 (exchangeability of the labeled pairs implies exchangeability of the scores).** Since
$s$ is a fixed measurable function (fixed *before* looking at the calibration/test split — this
is exactly what "split" conformal buys you: the score function itself was frozen on a disjoint
training fold, so it is not a function of the calibration/test indices), any permutation-symmetry
of $\{(X_i,Y_i)\}_{i=1}^{n+1}$ pushes forward to permutation-symmetry of
$\{s(X_i,Y_i)\}_{i=1}^{n+1} =: \{s_1,\dots,s_{n+1}\}$. So $s_1,\dots,s_{n+1}$ are exchangeable
scalar random variables.

**Step 3 (quantile lemma).** For exchangeable scalars $s_1,\dots,s_{n+1}$, the rank of $s_{n+1}$
among all $n{+}1$ values is, marginally, uniform on $\{1,\dots,n+1\}$ (this is the defining
symmetry property of exchangeability applied to the rank statistic, which is itself a symmetric,
hence exchangeable-invariant, function of the tuple — ties handled by an a.s.-continuity
assumption or by random tie-breaking). Consequently

$$
\mathbb P\Big(s_{n+1} \le \big(\text{the } \lceil(n+1)(1-\alpha)\rceil\text{-th smallest of }
s_1,\dots,s_n\big)\Big) \ \ge\ \frac{\lceil (n+1)(1-\alpha)\rceil}{n+1} \ \ge\ 1-\alpha,
$$

where the last inequality is just $\lceil z \rceil \ge z$ applied to $z = (n+1)(1-\alpha)$. The
left-hand quantity is exactly $\mathbb P(s_{n+1}\le \hat q)$, which by Step 1 equals
$\mathbb P(Y_{n+1}\in C(X_{n+1}))$. $\blacksquare$

(2601.16999's own Proposition 1 proof, §S1.1, is a verbatim instance of this argument dressed in
NER notation, citing "Lemma 2 of Romano, Patterson and Candes (2019)" for the quantile step; the
structure is identical to the one above.)

### Why the $\lceil(n+1)(1-\alpha)\rceil/n$ correction, not $(1-\alpha)$

This is not a cosmetic finite-sample nicety — it is *necessary* for the inequality to hold at all
for finite $n$, and the reason is visible directly in Step 3. If you instead used the naive
$(1-\alpha)$-quantile of the $n$ calibration scores (i.e. rank $\lfloor n(1-\alpha)\rfloor$ or
$n(1-\alpha)$ without any adjustment), the achieved rank-probability would be
$\lfloor n(1-\alpha)\rfloor / (n+1) < 1-\alpha$ for essentially every finite $n$ — you are
comparing the test score against only $n$ calibration draws while implicitly needing to place it
among $n+1$ exchangeable draws (itself included). Concretely: to guarantee the test point's rank
is $\le k$ out of $n+1$ with probability $\ge 1-\alpha$, you need $k/(n+1)\ge 1-\alpha$, i.e.
$k \ge (n+1)(1-\alpha)$, and since $k$ must be an integer you need $k=\lceil(n+1)(1-\alpha)\rceil$.
Two consequences that matter directly for GLiNER-scale calibration sets:

- **The correction is an $O(1/n)$ effect that vanishes asymptotically** (by (2), the two-sided gap
  is exactly $1/(n+1)$), so for large calibration sets ($n$ in the thousands, e.g. CoNLL-scale)
  the difference between $\lceil(n+1)(1-\alpha)\rceil/n$ and $(1-\alpha)$ is negligible in
  practice — but it is *not* negligible for small per-class calibration pools, which is exactly
  the regime we will hit under Mondrian/class-conditional calibration for rare GLiNER entity
  types (part (v)).
- **When $\lceil(n+1)(1-\alpha)\rceil > n$** (i.e. $n$ is too small relative to $\alpha$, concretely
  whenever $n < \alpha^{-1} - 1$), the quantile is undefined/returns $+\infty$ by convention and the
  prediction set degenerates to "include everything" — this is the formal reason a Mondrian mode
  needs $n^{(w)} \gtrsim 1/\alpha$ calibration points *per class* $w$ just for the correction term
  to be well-defined, independent of any statistical-efficiency argument (quantified further in
  part (v)).

---

## (ii) Why NER is a variable-cardinality SET prediction problem, and what breaks under naive porting

Standard conformal classification treats $Y$ as a single categorical draw from a fixed label space
$\mathcal Y = \{1,\dots,K\}$: one input, one true label, one nonconformity score $s(x,y)$ per
candidate label, one prediction set $C(x)\subseteq \mathcal Y$. NER breaks every one of these
assumptions simultaneously:

1. **The "label" is a set of typed spans, not a scalar.** For a sentence $x$ with $t$ tokens, the
   ground truth is $y = \{(a_1,b_1,c_1),\dots,(a_m,b_m,c_m)\}$ — a set of $m$ (start, end,
   type) triples, where $m$ itself is a **random variable with no fixed upper bound** (bounded
   only by $O(t^2\cdot|\mathcal T|)$ candidate spans $\times$ types in the enumeration sense, not
   by any statistical assumption). Classification conformal theory has no native object for "the
   true answer is itself a random-size collection."

2. **The unit of exchangeability question becomes genuinely ambiguous, and the three candidate
   choices are not interchangeable:**
   - **(a) The sentence** $(x_i, y_i)$ where $y_i$ is the *entire* label sequence/entity set. This
     is what 2601.16999's full-sequence method uses (§4.2, Proposition 1: exchangeability
     assumed over $\{(x_i,y_i)\}$ where $y_i$ ranges over full labelings in $\mathcal L^{t_i}$).
     This is the *only* one of the three choices for which the Section (i) proof goes through
     **without modification**, because sentences genuinely are i.i.d./exchangeable draws from the
     data-generating process (that is the natural sampling unit of a labeled corpus).
   - **(b) The individual span** (or every candidate (span, type) pair), treated as its own
     exchangeable draw, à la ordinary multi-class classification applied span-by-span. This is
     **not licensed by the same proof** without extra machinery, for two independent reasons.
     First, spans within the same sentence are **not exchangeable with spans from other
     sentences**: they share the same context vector $x$, the same encoder pass, and are
     dependent on each other through the model's contextualization — permuting *spans* (as
     opposed to permuting *sentences*) does not correspond to any symmetry of the actual
     data-generating process, so there is no exchangeability theorem to invoke at that level of
     granularity. Second, and more subtly, **the number of "trials" contributed by each sentence
     is itself informative** — a sentence with many gold entities is not a random, context-free
     draw of "many i.i.d. span trials"; $m$ is correlated with sentence content, and that content
     is exactly what also drives the nonconformity score. Pooling spans across sentences into one
     flat i.i.d.-looking calibration set silently reweights the implicit sampling distribution
     toward sentences with more entities, which is not obviously the population the marginal
     guarantee is supposed to describe.
   - **(c) The (sentence, gold-span) pair, conditional on the sentence having $\ge 1$ candidate of
     the class in question.** This is what 2601.16999's subsequence/entity-level method actually
     does (§5): it defines the guarantee (their Eq. 23) as
     $\mathbb P(w\in C_{w,\mathrm{ent}}(x_{\mathrm{new}},\tau_w,a,b) \mid y_{a:a+b}=w) \ge 1-\alpha$
     — i.e. **conditional on the event that this particular subsequence is truly of class $w$**.
     This sidesteps issue (b)'s "informative $m$" problem by *conditioning it away*: the
     calibration pool for class $w$ is literally "every gold span of class $w$ across the corpus,"
     and the guarantee is stated *given* that a span of class $w$ occurs at that location — it
     says nothing directly about "coverage per sentence" or about spans that are *not* of class
     $w$. This is a **weaker and different object** than (a): it is a per-occurrence guarantee
     about the conditional distribution of scores given class membership, not a per-sentence
     guarantee about the whole label sequence.

3. **What concretely breaks if you naively flatten spans into an i.i.d. classification pool and
   apply vanilla conformal classification per span, ignoring the joint-labeling structure:** you
   get *marginal, per-span* coverage in the weak "conditional-on-class" sense of (c) above — this
   part is not broken, 2601.16999 proves it (their Eq. 26). What breaks is the **translation back
   to a sentence-level or "did I recover this entity correctly" guarantee**, for two compounding
   reasons documented explicitly in the paper:
   - **Family-wise error from combining $m$ per-span guarantees into one sentence-level claim.**
     If a sentence contains $s$ true entities and you naively AND together $s$ independent
     $1-\alpha$-level per-span events, the probability all $s$ hold jointly has *lower bound*
     $(1-\alpha)^s$, not $1-\alpha$ — this is worse than a Bonferroni problem, it is the same
     phenomenon PASC frames abstractly (part (iv) below: "the probability that all stages are
     simultaneously covered is at most $(1-\alpha)^K$"). 2601.16999 §6 confirms this empirically:
     their "Integrated without Šidák" method, which does exactly this naive per-span-then-AND
     combination, **fails to maintain valid coverage for multi-entity inputs** — Table 5 shows
     empirical coverage dropping from 97.7% (1 entity) to 86.8% (5 entities) against a 95% target,
     a real, measured, structural failure — and requires an explicit Šidák correction
     $1-\alpha_{\text{Šidák}} = (1-\alpha)^{1/\hat s}$ (where $\hat s$ is the *predicted*, not
     true, number of entities — itself an approximation) to restore validity.
   - **The reference class problem for "false" candidates.** Classification conformal sets are
     defined over $\mathcal Y=\{1,\dots,K\}$, a space that contains *every possible label*
     including the true one by definition of the label space. In NER, the overwhelming majority
     of candidate spans are non-entities (label "O" / not-a-span), and the "true" object being
     predicted is a sparse subset of an $O(t^2)$-sized candidate universe. A prediction *set* in
     the classification sense (⊆ label space) is not the natural object; the natural per-instance
     object is closer to a *risk-controlled selection rule* (part (iii), risk-control mode) than
     to a coverage set, precisely because $|y|$ is unbounded and most of the "label space" is
     structurally negative.

**Bottom line for GLiNER.** The clean, provably valid statement is the *sentence-as-exchangeable-
unit, full-label-sequence* one (2(a)) — but GLiNER does not produce a single joint labeling
distribution the way a CRF does (see part (iv)); it produces $O(L\cdot K)$ **independent** per-
(span,type) sigmoids (this repo's own `gliner/decoding/decoder.py` implements exactly
"sigmoid + threshold + greedy overlap resolution," confirming there is no joint sequence model to
put a full-sequence conformal set over). This pushes us structurally toward either (c) — per-
entity/per-class conditional coverage, with the family-wise caveats above made explicit rather
than hidden — or toward risk control over the whole extracted set (part iii), which sidesteps the
"set of labelings" formalism entirely by controlling an *expectation* instead of a coverage
*event*.

---

## (iii) Formal definitions of the three candidate guarantees

Notation: $x$ = a sentence, $y(x) = \{(a_i,b_i,c_i)\}_{i=1}^{m(x)}$ = gold typed spans, $\mathcal
T$ = set of entity types under consideration, $p_\theta(\text{span},t\mid x)\in(0,1)$ = GLiNER's
sigmoid score for span $\text{span}$ and type $t$.

### (a) Span-filter mode: marginal per-entity coverage

**Definition.** For a nonconformity score $s(x,(\text{span},t))$ (e.g. $1-p_\theta$), a
per-class threshold $\tau_t$, and prediction set
$C_t(x) = \{\text{span} : s(x,(\text{span},t)) \le \tau_t\}$, the guarantee we can rigorously make
is:

$$
\mathbb P\big(\text{gold span}\in C_t(x_{\mathrm{new}}) \;\big|\; (\text{span},t)\text{ is a true
entity of type } t \text{ in } x_{\mathrm{new}}\big) \ \ge\ 1-\alpha. \tag{3}
$$

**The subtlety the task flags is real, and here is the precise resolution.** (3) is **not** the
same statement as "$\mathbb P(\text{sentence } x_{\mathrm{new}} \text{ has all its entities
covered}) \ge 1-\alpha$" and it is **not** the same as an unconditional statement over sentences.
The exchangeability unit that licenses (3), following 2601.16999 Eq. 23–26 exactly, is: *the pool
of (sentence, gold-span) pairs restricted to spans whose true type is $t$, across the corpus, is
exchangeable* — which follows from exchangeability of sentences plus a fixed, score-independent
rule for enumerating gold spans within a sentence. What "$1-\alpha$" bounds under this framing is
a **frequency over entity occurrences of type $t$**, not a frequency over sentences and not a
frequency over all entity types pooled together. Two sentences that each contain 10 type-$t$
entities contribute 10 "trials" each to this guarantee, so a marginal miscoverage event
concentrated in a few entity-dense sentences is fully consistent with (3) holding — this is
exactly analogous to the "Group A / Group B" marginal-vs-conditional trap in Angelopoulos & Bates
§3.2 (their Figure 10), just instantiated at the (sentence, span) granularity instead of a
demographic-group granularity. If a *per-sentence* ("does this sentence's full entity set validate
end-to-end") guarantee is wanted, that requires either the full-sequence route (part ii, unit 2a)
or the risk-control route below, not this one.

### (b) Risk-control mode: bounding expected miss rate

**Loss.** Define the per-sentence miss rate at threshold $\lambda\in[0,1]$,

$$
\ell(C_\lambda(x), y(x)) \;=\;
\begin{cases}
1 - \dfrac{|\,y(x) \cap C_\lambda(x)\,|}{|y(x)|}, & y(x)\neq\varnothing \\[4pt]
0, & y(x)=\varnothing
\end{cases}
\qquad
C_\lambda(x) = \{(\text{span},t) : p_\theta(\text{span},t\mid x) \ge 1-\lambda\}. \tag{4}
$$

(The $y(x)=\varnothing$ convention avoids a $0/0$; it is the standard convention used for the
false-negative-rate example in Angelopoulos & Bates §4.3 and in Conformal Risk Control §1, and it
matches GLiNER's own decoding convention of simply emitting no spans for an entity-free sentence.)

This is a direct instance of the worked multilabel-classification example in both source papers
(Gentle Intro §4.3: $C_\lambda(x)=\{k: f(X)_k\ge 1-\lambda\}$; CRC §1.1, same form) — GLiNER's
independent per-(span,type) sigmoid architecture is *literally* the multilabel setting these
papers use as their canonical CRC example, with "class $k$" replaced by "candidate (span,type)
pair." This correspondence is not a coincidence to be argued for; it is a syntactic match.

**Guarantee.** Choose

$$
\hat\lambda \;=\; \inf\Big\{\lambda\in[0,1] : \widehat R_n(\lambda) + \frac{B-\alpha}{n} \le \alpha\Big\},
\qquad \widehat R_n(\lambda) = \frac1n\sum_{i=1}^n \ell(C_\lambda(x_i), y(x_i)), \tag{5}
$$

with $B=1$ here (the loss (4) is bounded in $[0,1]$; this is the finite-sample-conservative
$\hat\lambda$ formula from CRC Theorem 1's proof / Gentle Intro Eq. 12, not the naive
$\inf\{\lambda:\widehat R_n(\lambda)\le\alpha\}$ — the extra $(B-\alpha)/n$ margin is exactly what
makes the finite-sample proof (below) go through, analogous in spirit to the $\lceil\cdot\rceil$
correction in part (i)). Then, **provided the monotonicity condition below holds**,

$$
\mathbb E\big[\ell(C_{\hat\lambda}(X_{\mathrm{new}}), Y_{\mathrm{new}})\big] \ \le\ \alpha. \tag{6}
$$

**Monotonicity/nesting condition (CRC Theorem 1, verbatim requirement): $\ell(C_\lambda(x),y)$
must be non-increasing and right-continuous in $\lambda$, and $\ell(C_{\lambda_{\max}}(x),y)\le\alpha$
almost surely.** This is not automatic for an arbitrary loss — Conformal Risk Control's own
Proposition 1 explicitly exhibits a non-monotone loss for which the guarantee (6) **fails by an
arbitrary amount** (their bound: $\mathbb E[\ell(C_{\hat\lambda},Y)] \ge B-\epsilon$ for any
$\epsilon$). So this has to be checked, not assumed, for GLiNER.

**Proof that GLiNER's sigmoid-threshold miss rate (4) *is* monotone non-increasing in $\lambda$ as
$\lambda$ decreases from 1 (⟺ threshold $1-\lambda$ increases toward 1, more conservative) — i.e.
that it satisfies the CRC condition.**

*Claim 1 (nesting).* For $\lambda_1\le\lambda_2$, $C_{\lambda_1}(x)\subseteq C_{\lambda_2}(x)$.

*Proof.* $\lambda_1\le\lambda_2 \implies 1-\lambda_1 \ge 1-\lambda_2$. If
$(\text{span},t)\in C_{\lambda_1}(x)$ then $p_\theta(\text{span},t\mid x)\ge 1-\lambda_1 \ge
1-\lambda_2$, so $(\text{span},t)\in C_{\lambda_2}(x)$. $\square$

*Claim 2 (monotone loss).* $\lambda_1\le\lambda_2 \implies \ell(C_{\lambda_1}(x),y(x)) \ge
\ell(C_{\lambda_2}(x),y(x))$.

*Proof.* If $y(x)=\varnothing$ both sides are $0$, trivial. Otherwise, by Claim 1,
$y(x)\cap C_{\lambda_1}(x) \subseteq y(x)\cap C_{\lambda_2}(x)$, so
$|y(x)\cap C_{\lambda_1}(x)| \le |y(x)\cap C_{\lambda_2}(x)|$, hence
$1 - \frac{|y(x)\cap C_{\lambda_1}(x)|}{|y(x)|} \ \ge\ 1 - \frac{|y(x)\cap C_{\lambda_2}(x)|}{|y(x)|}$,
which is exactly $\ell(C_{\lambda_1},y) \ge \ell(C_{\lambda_2},y)$. $\square$

*Right-continuity.* For fixed $x,y$, the candidate set of $(\text{span},t)$ pairs is finite (at
most $O(L\cdot |\mathcal T|)$ where $L$ is the number of enumerated spans), so
$\lambda \mapsto \ell(C_\lambda(x),y)$ is a finite step function with jumps exactly at
$\lambda = 1-p_\theta(\text{span},t\mid x)$ for each candidate. Because inclusion in $C_\lambda$
uses "$\ge$" (a closed/weak inequality against the threshold $1-\lambda$), at each jump point the
pair *enters* the set at the jump value itself, i.e. $\ell$ takes its *lower* (post-jump) value at
the jump point — this is precisely the right-continuous convention CRC requires.

*Boundary condition $\ell(C_{\lambda_{\max}},y)\le\alpha$ a.s.* At $\lambda_{\max}=1$, threshold
$=1-\lambda_{\max}=0$, and since $p_\theta\in(0,1)$ strictly (sigmoid output), **every** enumerated
candidate satisfies $p_\theta\ge 0$, so $C_1(x)$ = the full candidate universe $\supseteq y(x)$,
giving $\ell(C_1(x),y(x))=0\le\alpha$ for every $\alpha>0$, a.s. $\blacksquare$

So: **yes**, GLiNER's independent-sigmoid architecture satisfies the CRC monotonicity requirement
*by construction*, because the miss-rate loss is a monotone functional of a *nested* family of
sets, and nestedness under a single shared scalar threshold on independent per-item scores is
essentially automatic (this is the same reason the multilabel classification worked example in
both source papers is monotone — GLiNER's decode rule is that example). The one place this could
fail in practice is if GLiNER's actual deployed decoder does *not* use a monotone family — e.g. if
greedy overlap/nested-span resolution (also implemented in `gliner/decoding/decoder.py` per this
repo's own code, on top of the raw sigmoid threshold) discards some spans *based on their
overlap with other spans* rather than purely on score. If the overlap-resolution step is applied
*before* defining $C_\lambda$ as "everything that survives decoding at threshold $\lambda$," it can
break Claim 1's nesting (a span present at a looser threshold could be suppressed by a
newly-admitted higher-priority overlapping span that would not have existed at a tighter
threshold) — this is an implementation-level caveat that Phase 1 must design around explicitly
(e.g. by defining $C_\lambda$ on the *pre-overlap-resolution* candidate set, then applying
overlap resolution as a fixed, threshold-independent post-processing step, so that the object
being calibrated is still provably nested), not a caveat about the theory above.

### (c) Mondrian / class-conditional mode

**Definition.** Calibrate a separate threshold $\tau_t$ per entity type $t\in\mathcal T$ using
only calibration spans of that type (exactly the construction in part (v) below and in
2601.16999 §5, Eq. 24). The guarantee is the *conjunction* of $|\mathcal T|$ separate instances
of (3):

$$
\forall t\in\mathcal T:\quad \mathbb P\big(\text{gold span}\in C_t(x_{\mathrm{new}})\;\big|\;
\text{true type}=t\big)\ \ge\ 1-\alpha. \tag{7}
$$

This is strictly stronger than (a) marginalized over types, because (a) only bounds the
*pooled* average across types (a marginal average can hide a rare type at 40% coverage offset by a
common type at 99.9% coverage), whereas (7) bounds each type separately — this is precisely the
"rare types systematically under-covered by the marginal guarantee" failure mode the task names,
and it is a real, not hypothetical, failure mode: 2601.16999 Table 8 measures exactly this and
finds full-sequence sets "fail to meet class-conditional coverage for the Miscellaneous class"
under marginal calibration, motivating their §5. The calibration-data cost of (7) is quantified in
part (v).

---

## (iv) Nonconformity scores in 2601.16999, and PASC's relevance to GLiNER

### Full-sequence vs. subsequence scores in 2601.16999

2601.16999 trains a **CRF** (not a GLiNER-style span classifier) and defines three baseline
nonconformity scores over *entire label sequences* $y^{(r)}$, ranked by the CRF's joint
probability $\hat P(y\mid x)$ (their Eq. 11, exact text):

$$
\mathrm{nc}_1(y\mid x) = 1-\hat P(y\mid x), \qquad
\mathrm{nc}_2(y^{(r)}\mid x) = \sum_{k=1}^r \hat P(y^{(k)}\mid x), \qquad
\mathrm{nc}_3(y^{(r)}\mid x) = r,
$$

where $y^{(r)}$ is the $r$-th most probable *full sentence labeling* under beam search (they use
beam width $K=100$; this caps the maximum achievable coverage at ≈99% at the sentence level, an
explicit engineering tradeoff they report). The **full-sequence** prediction set (their §4.2.1) is
literally a set of candidate full-sentence labelings $\{y^{(1)},y^{(2)},\dots\}$ — i.e. the
conformal object lives in the space of entire label sequences, and validity is w.r.t. "the whole
sentence's labeling is exactly right."

The **subsequence** variant (their §5) redefines the unit of prediction to a single entity span.
They first define the marginal probability that a specific subsequence $y_{a:a+b}$ equals a given
entity class $w$ by summing over the top-$K$ decoded sequences that contain it (Eq. 19):
$\hat P_{\mathrm{ent}}(y_{a:a+b}=w) = \sum_{r=1}^K \hat P(y^{(r)}) \cdot
\mathbf 1[y^{(r)}_{a:a+b}=w]$, then define per-class analogues of $\mathrm{nc}_1,\mathrm{nc}_2,
\mathrm{nc}_3$ over this marginal (Eqs. 20–22), and calibrate a **separate threshold $\tau_w$ per
class** using only calibration entities of that class. This is the direct precursor of our part
(iii)(c) Mondrian mode. **The key structural difference from full-sequence:** subsequence scores
throw away the sentence-level joint dependency (the paper's own §5 admits this: "these sets do not
capture contextual dependencies across different entities within a full sentence") in exchange for
class-conditional validity per entity, whereas full-sequence scores keep the joint dependency but
can only make a whole-sentence-level coverage claim. Their §6 "integrated" method is an explicit
attempt to recombine the two (intersect a full-sequence-derived set with a union of per-class
subsequence sets), which is exactly why it needs the Šidák correction discussed in part (ii).

None of this transfers to GLiNER as-is, because **GLiNER has no CRF / no joint sequence
probability $\hat P(y\mid x)$ to rank candidate full labelings with** — it produces independent
per-(span,type) sigmoids. The *subsequence* framing (per-entity nonconformity, per-class
calibration) is architecturally compatible with GLiNER pretty much unchanged (replace
$\hat P_{\mathrm{ent}}(y_{a:a+b}=w)$ with $p_\theta(\text{span},w\mid x)$ directly — GLiNER already
computes exactly this quantity, with no need for the top-$K$-beam approximation the CRF paper
needs, since GLiNER's per-pair sigmoid *is* already the marginal). The *full-sequence* framing does
not transfer without inventing a joint-sequence probability model GLiNER does not have.

### PASC's relevance to GLiNER: judgment call

**PASC (2605.18812)** solves a different problem: given $K$ *sequentially composed* models
$x \xrightarrow{f_1} z_1 \xrightarrow{f_2}\cdots\xrightarrow{f_K} z_K$ (their own worked example is
literally NER→NED→EntityTyping), it reduces the joint event "all $K$ stages are individually
correct" to a single scalar conformal problem via
$\bigcap_{k=1}^K\{s_k\le q\} = \{\max_k s_k \le q\}$ (their Proposition 4, proved by definition of
max — trivial once stated, but structurally the entire contribution of the paper), then applies
ordinary split conformal (part (i)) to the scalar $s_{\max}$ (their Theorem 6, which is *literally*
Theorem 3 in the same paper — i.e. the same Angelopoulos/Bates-style split conformal theorem —
applied to the derived scalar $s_{\max}$, with an explicit near-tightness bound
$1-\alpha \le \mathbb P(\cdot)\le 1-\alpha+1/(n+1)$ that is exactly our Eq. (2) again).

**Judgment: PASC is *not* directly relevant to the core GLiNER-Robust deliverable, and should not
be adopted, for a specific, arguable reason — not merely "GLiNER is single-stage."** GLiNER *is*
architecturally single-stage for the pure-NER use case (one forward pass, one score per
(span,type) pair — no sequential composition of independently-trained sub-models). PASC's own
paper says outright: "For $K=1$ (single stage): PASC reduces to standard conformal prediction ...
no multi-stage composition effects arise." So invoking PASC machinery for plain GLiNER NER would
be invoking a $K=1$ special case that collapses to exactly the part (i) theorem with no addition —
there is nothing PASC adds in that regime. Where PASC *would* become relevant is if
GLiNER-Robust's scope grows to include a **downstream stage consuming GLiNER's output** — e.g. an
entity-linking/disambiguation step, a relation-extraction step chained after span extraction (this
repo's `predict_relations` / `gliner/multitask/` machinery, per project memory, already contains a
relation-extraction wrapper on top of GLiNER) — in which case the "does the whole pipeline's
output jointly validate" question becomes exactly PASC's setting, and the maximum-nonconformity
reduction (their Definition 5, Eq. 8) would be the right tool, essentially for free (one shared
quantile, no Bonferroni tax; their Table 1: 96.4% vs. 93.4% Bonferroni vs. 86.5% independent CP at
identical set size). **Recommendation for Phase 1: treat GLiNER-Robust's core NER conformal module
as $K=1$/PASC-irrelevant now; keep PASC's max-nonconformity reduction in the back pocket
specifically for the day a relation-extraction or entity-linking stage gets bolted onto GLiNER's
output and needs a joint guarantee.**

**One caveat on PASC's own credibility, noted for calibration of how much weight to place on it:**
this is a single-author preprint (independent researcher, no institutional affiliation given) with
several self-citations to other 2026 preprints by the same author (Kotte 2026a, 2026b, and a
pending US patent application by the same author), it is not obviously peer-reviewed, and its
central theoretical claim (Proposition 4 / Theorem 6) — while mathematically correct as stated,
verified above — is a comparatively small step beyond a well-known identity
($\bigcap_k\{s_k\le q\}=\{\max_k s_k\le q\}$) dressed up in pipeline language. This does not affect
the correctness of the theorem, but it does mean it should be cited as "a straightforward and
mathematically valid instance of split conformal prediction applied to a max-aggregated score,"
not leaned on as a load-bearing external validation for anything beyond that.

---

## (v) Mondrian conformal prediction: general theory and calibration-data cost

**General method (Vovk's original construction, as restated with full proofs in Angelopoulos &
Bates §4.1–4.2, Propositions 1–2; independently re-derived for the NER setting as Theorem 1 in
2601.16999).** Given any partition of the joint sample space $\mathcal E = \bigsqcup_{j=1}^m E_j$
into $m$ measurable, mutually exclusive, exhaustive categories (a "Mondrian taxonomy" — group
membership, class label, or any other partition, so long as it is determined without looking at
the nonconformity score itself), calibrate **separately within each cell**:

$$
\tau_j = \mathrm{Quantile}\Big(\{s(X_i,Y_i)\}_{(X_i,Y_i)\in E_j};\ \frac{\lceil(n^{(j)}+1)(1-\alpha)\rceil}{n^{(j)}}\Big),
\qquad C_j(x) = \{y : s(x,y)\le\tau_j\},
$$

where $n^{(j)}=|\{i: (X_i,Y_i)\in E_j\}|$. **Guarantee:**
$\mathbb P\big(Y_{n+1}\in C_j(X_{n+1}) \mid (X_{n+1},Y_{n+1})\in E_j\big)\ge 1-\alpha$ for *every*
cell $j$ simultaneously. **Why this is valid and not just a heuristic:** exchangeability is
preserved under partitioning — 2601.16999's Theorem 1 proof (their §S1.4, quoted in full above in
part (ii)) is exactly: partitioning an exchangeable calibration set by a score-independent
criterion leaves each partition's sub-collection exchangeable, so Theorem/Proposition 1 (part i)
applies *verbatim within each cell, with $n$ replaced by $n^{(j)}$*. Nothing new is needed
theoretically; the entire content of "Mondrian conformal prediction" is the observation that
exchangeability is a property closed under this kind of conditioning.

**The calibration-data cost, quantified.** The cost is not a vague "you need more data for more
classes" hand-wave — it decomposes into two genuinely distinct effects, both visible directly from
part (i)'s machinery:

1. **A hard threshold-existence floor, per class, from the $\lceil\cdot\rceil$ correction itself.**
   As shown in part (i), the correction term is only well-defined and non-degenerate once
   $n^{(j)} \gtrsim \alpha^{-1}$ — more precisely, $\lceil(n^{(j)}+1)(1-\alpha)\rceil \le n^{(j)}$
   requires $n^{(j)} \ge \lceil(n^{(j)}+1)(1-\alpha)\rceil$, which rearranges to
   $n^{(j)} \ge \frac{1-\alpha}{\alpha} = \frac1\alpha - 1$. Below this, the per-class quantile
   saturates at "include everything" (infinite/maximal threshold) — the guarantee (7) is then
   technically still *true* but *vacuous* (the "prediction set" for that class is the whole
   candidate universe). At $\alpha=0.1$ this is $n^{(j)}\ge 9$; at $\alpha=0.01$,
   $n^{(j)}\ge 99$. For a Mondrian split over $m$ classes with a fixed total calibration budget
   $n$, uniform allocation gives $n^{(j)}\approx n/m$, so the hard requirement becomes
   $n \gtrsim m/\alpha$ — **linear in both the number of classes and $1/\alpha$**. This is the
   direct, provable, non-asymptotic version of "$\Theta(1/\alpha)$ calibration points per class."
2. **A statistical-efficiency / variance cost on top of the floor**, which is where "per-class
   coverage variance" enters and where the two source papers stop short of giving a closed-form
   because it depends on the (unknown, model- and score-dependent) shape of the per-class
   nonconformity score distribution near its $(1-\alpha)$-quantile — Ding, Angelopoulos, Bates,
   Jordan & Tibshirani, *Class-conditional conformal prediction with many classes* (NeurIPS 2023),
   cited by 2601.16999 explicitly as the source for handling exactly this many-classes regime, is
   the paper that develops the finer-grained (quantile-estimation-variance) analysis; it was not
   itself fetched in this research pass (out of scope of the 7 sources assigned), so its precise
   rate is not restated here as a "read" result — but its *existence and citation context* confirm
   that the floor above is the necessary-but-not-sufficient condition, and that achieving *low
   variance* around the target $1-\alpha$ (as opposed to merely a well-defined, non-degenerate
   threshold) requires materially more than $\Theta(1/\alpha)$ points per class in practice,
   scaling further with the desired tightness of per-class coverage.

**Direct consequence for GLiNER-Robust's Mondrian mode.** GLiNER's zero-shot entity-type space is
effectively unbounded/open-vocabulary (any user-supplied string is a valid "type" at inference).
Mondrian calibration is only rigorously definable over the **finite set of types actually observed
in the calibration corpus**, and per the floor above, each such type needs $\gtrsim 1/\alpha$
calibration occurrences (e.g. $\ge9$ at $\alpha=0.1$, realistically far more for a non-vacuous,
low-variance threshold) before its Mondrian threshold is meaningful. Long-tail types with fewer
than a handful of calibration occurrences (a near-certainty for any broad-coverage zero-shot
calibration corpus, by Zipf's law over entity type frequency) will structurally get vacuous or
high-variance thresholds under this scheme — this is a real, foreseeable engineering constraint
for Phase 1, not a hypothetical.

---

## (vi) Does exchangeability hold for "calibrate on training-domain labels, evaluate zero-shot on unseen entity types"? — the crux question

**Short answer: no, not in the form needed for a rigorous marginal-coverage claim about performance
on a genuinely novel entity type, and — critically — neither of the two target papers actually
engages with this question, because neither one operates in a genuinely open-vocabulary label
setting. This is worth stating plainly rather than papered over, since it is exactly what the task
asked to check.**

**What the source papers actually assume (verified directly, not inferred).** 2601.16999's entire
framework is built on a **fixed, closed label space** $\mathcal L = \{l_0,l_1,\dots,l_c,l_{\rm
start},l_{\rm stop}\}$ (their §3, verbatim), with $c$ named entity types fixed *before* any
calibration or test data is seen — all four of their evaluated models (Babelscape, Dslim,
Jean-Baptiste, TNER) are standard closed-set sequence taggers. Their "TNER trained on OntoNotes,
evaluated/fine-tuned on CoNLL" experiment, which is the closest thing in the paper to a
distribution-shift stress test, is explicitly a **domain shift** (different corpus, same *general
sense* of what an entity is, and TNER is fine-tuned before conformal calibration, so the label
space at calibration time matches the label space at test time) — it is not a **label-space shift**
(a genuinely new type unseen anywhere in training or calibration). *A first-pass automated fetch of
this paper's contents produced the claim that "the authors acknowledge exchangeability may not
hold for unseen entity types" — on full-text verification (grep + direct read of the extracted
PDF text) this sentence does not appear anywhere in the paper. This is flagged here explicitly as a
claim that was generated by an intermediate summarization step and did not survive verification
against the primary source; it is retracted and should not be treated as coming from 2601.16999.*
The paper simply does not discuss open-vocabulary or zero-shot entity types at all — the entire
apparatus (full-sequence labelings over $\mathcal L^{t}$, per-class Mondrian calibration indexed
by $w \in W$ for a fixed, enumerable $W$) is only defined relative to a closed $\mathcal L$/$W$.

Similarly, **PASC** explicitly limits its own exchangeability claim to distribution/covariate
shift, and says so directly in its own Discussion section (§7, quoted verbatim above in the
research log): *"Like all split CP methods, PASC requires exchangeability of calibration and test
data. Under covariate shift (e.g., WNUT-17), PASC still achieves $\ge 1-\alpha$ coverage
empirically ... however, the theoretical guarantee strictly requires exchangeability."* Their own
WNUT-17/WikiNEuRal shift experiments are, again, **domain shift within a fixed label space**
(CoNLL's 4 types), not label-space expansion.

**So the honest position is: this is a genuine, unaddressed gap in the literature Agent B was
asked to survey, not a solved problem we can cite our way out of.** Here is the precise argument
for *why* it cannot hold cleanly, stated at the same level of rigor as part (i)'s proof, followed
by what weaker claim actually survives.

**Why standard exchangeability fails for genuine zero-shot label-space extrapolation.** The proof
in part (i) requires that the calibration scores $\{s(X_i,Y_i)\}$ and the test score
$s(X_{n+1},Y_{n+1})$ be exchangeable **as a joint $(n{+}1)$-tuple**, which in particular requires
that $(X_{n+1},Y_{n+1})$ be drawn from *the same underlying distribution* (up to permutation
symmetry) as the calibration pairs. If calibration is performed using gold spans of types
$\mathcal T_{\rm cal} = \{{\rm PER, ORG, LOC,\dots}\}$ and the deployed/test query asks GLiNER to
score a type $t^\ast \notin \mathcal T_{\rm cal}$ (e.g. "chemical compound," "software license," a
type that appears zero times, or even a semantically unrelated distribution of types, in
calibration), then **there is no sense in which $(X_{n+1}, Y_{n+1})$ for that query is exchangeable
with the calibration tuples**: the marginal distribution of "true label given this is a
$t^\ast$-typed span" was never represented in the calibration draw at all — exchangeability
requires that every element of the augmented $(n{+}1)$-tuple be, marginally, drawn from a common
underlying law up to permutation, and a type with **zero calibration mass** trivially cannot
satisfy this (you cannot permute a data point that was never sampled into existence). This is not
a subtle failure of a technical regularity condition — it is a structural absence of the object the
theorem quantifies over. Formally: the Mondrian per-class guarantee (part iii-c, part v) is
*undefined*, not merely "wide," for $t^\ast\notin\mathcal T_{\rm cal}$, since $n^{(t^\ast)}=0$
makes the quantile computation in part (i) vacuous by construction (there is no calibration score
to rank against).

Even the **marginal** (pooled-over-all-types) guarantee (part iii-a, or unconditional split
conformal over the union of all calibration types) does not transfer cleanly to $t^\ast$: the
marginal guarantee (3) is a statement about the *pooled population of (sentence, gold-span) pairs
that occurred in calibration*, and a query at $t^\ast$ is, by the zero-shot premise, drawn from a
part of $(x,y)$-space with **structurally different nonconformity-score behavior** (GLiNER's
sigmoid confidence calibration is itself known to depend on how semantically close the queried
type's text prompt is to types seen during *training*, which is a separate but related
distribution-shift channel on top of the calibration/test split). There is no theorem in any of
the seven surveyed sources — nor, as far as this survey went, in the general conformal literature
these sources cite (Tibshirani et al.'s covariate-shift conformal prediction, cited by both
2107.07511 and PASC, is the standard tool for *known, reweightable* covariate shift, not for
*support-set expansion where the new region has zero calibration density*) — that licenses a
finite-sample marginal coverage claim at $t^\ast$ under these conditions. **"Zero-shot conformal
NER," read as "a rigorous $1-\alpha$ coverage claim about performance on an entity type never
observed in calibration," is not a coherent claim under the standard exchangeability framework, and
Phase 1 should not present it as one.**

**What weaker, still-rigorous claim survives, and what a Phase 1 design should actually promise.**
Three options, in decreasing order of how much they resemble the original ambition:

1. **Restrict the rigorous guarantee to the closed set $\mathcal T_{\rm cal}$ of types actually
   represented (with adequate mass, per part v's floor) in calibration, and be explicit that the
   guarantee does not extend beyond it.** This is honest and immediately implementable: ship
   Mondrian per-type thresholds for every type with $n^{(t)}\gtrsim 1/\alpha$ calibration
   occurrences, and for any type outside that set, either (a) refuse to issue a calibrated
   threshold and fall back to the raw uncalibrated sigmoid (clearly flagged as such to the
   downstream consumer), or (b) issue the pooled/marginal threshold (part iii-a) with an explicit
   caveat that it is *not* proven valid off-support — this converts an implicit, false claim into
   an explicit, true one about a smaller domain.
2. **Covariate-shift-weighted conformal prediction** (Tibshirani, Barber, Candès & Ramdas 2019,
   cited in both source papers' related work, not independently fetched in this pass) replaces the
   uniform exchangeability assumption with a *known likelihood-ratio reweighting* between the
   calibration and test covariate distributions, and recovers a valid guarantee *provided the
   likelihood ratio $w(x) = d\mathbb P_{\rm test}(x)/d\mathbb P_{\rm cal}(x)$ is known or
   estimable and has bounded support overlap*. This is mathematically real, but it requires
   $\mathbb P_{\rm cal}$ to place **positive density** on the region containing $t^\ast$-typed
   queries — i.e. it can rigorously handle "$t^\ast$ is rare but not absent" (reweight toward it),
   it **cannot** handle "$t^\ast$ has literally zero calibration density," which is exactly the
   genuinely-novel-type case. So this is a real tool for the *long-tail-but-observed* problem
   flagged in part (v), not for the *never-observed* problem.
3. **Drop the marginal-coverage framing entirely for novel types and report only an empirical,
   non-guaranteed calibration diagnostic** (e.g. measured coverage on a held-out set of *known*
   types, reported as a *transfer-quality proxy*, explicitly labeled as not carrying a
   distribution-free guarantee) — this is honest about being a heuristic, not a rebranded
   guarantee, and is the same posture the CRC paper itself takes toward non-monotone losses
   (part iii-b): when the formal condition fails, *say so and fall back to a clearly-labeled
   heuristic* rather than silently keeping the "$1-\alpha$" language.

**Recommendation for Phase 1 (design-relevant, stated plainly since the task says this decision
depends on the answer being honest):** ship the rigorous guarantee (span-filter or risk-control,
Mondrian where data supports it) scoped explicitly to the calibration corpus's observed type
distribution, market it as "coverage/risk guarantees for entity types represented in calibration,"
and do **not** market a coverage guarantee for arbitrary user-supplied zero-shot types — that
specific claim is not supportable by the conformal-prediction machinery surveyed here, full stop.
If genuine open-vocabulary guarantees are a hard project requirement, the honest next research
step is investigating whether a *structural* (not statistical) argument is available — e.g.
whether GLiNER's dual-encoder similarity-based scoring (arXiv:2602.18487) admits some kind of
Lipschitz/metric-embedding argument bounding score miscalibration as a function of embedding-space
distance from the nearest calibration type — but that would be a materially different, and
currently unestablished, theoretical foundation than anything in the seven sources reviewed here,
and is out of scope for this Phase 0 literature pass.
