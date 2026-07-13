# GLiNER-Robust — Master Engineering Roadmap

> This document is the single source of truth for all planned improvements.
> Update status fields as work progresses. Never delete a completed item — mark it ✅.

> **Branch note (2026-07-13):** this document describes the `feature/vocab-pruning-engine`
> branch (upstream-PR track). There is a second, unmerged branch,
> `feat/focal-dice-loss-openvino`, carrying a separate research-paper project (Dice loss,
> span-width weighting, OpenVINO INT8) — see `CLAUDE.md` for that track. The two have not been
> reconciled; features below do not include anything from the paper track.

---

## Current Branch State

| Branch | Feature | Status |
|---|---|---|
| `feature/vocab-pruning-engine` | Vocabulary Pruning Engine | ✅ COMPLETE |
| `feature/flash-deberta` | FlashDeBERTa Integration | ✅ COMPLETE (Python 3.10+ required for install) |
| `feature/entity-descriptions` | Entity Type Description Conditioning | ✅ COMPLETE |
| `feature/sliding-window` | Long-Document Sliding Window Inference | ✅ COMPLETE |
| `feature/hard-negatives` | Hard Negative Sampling | ✅ COMPLETE |
| `feature/contrastive-loss` | Label-Aware Contrastive Loss | ✅ COMPLETE |
| `feature/modernbert` | ModernBERT Backbone | ✅ COMPLETE |
| `feature/joint-ner-re` | Joint NER + Relation Extraction | ✅ COMPLETE |
| `feature/curriculum-learning` | Curriculum Learning Sampler | ✅ COMPLETE |

---

---

# FEATURE 1 — FlashDeBERTa Integration

**Branch:** `feature/flash-deberta`
**Motivation:** DeBERTa v2/v3's disentangled relative attention computes a full (L×L) position-bias matrix, making memory quadratic in sequence length. This is what causes the 384-token practical limit and makes the model slow at longer inputs. FlashDeBERTa (Knowledgator) rewrites this kernel with Flash Attention-style tiling, cutting memory to near-linear and achieving 50% speedup at 512 tokens, 5× at 4k tokens.

**Current state (discovered in code):** Already skeleton-integrated via env var:
```python
# gliner/modeling/encoder.py:117
if os.environ.get("USE_FLASHDEBERTA", "") and IS_FLASHDEBERTA:
    ModelClass = FlashDebertaV2Model
```
But NOT exposed as a proper API parameter, not documented, and the config has no field for it. This feature promotes it to a first-class citizen.

---

## Step 1.1 — Add `use_flash_attention` to `BaseGLiNERConfig`

**File:** `gliner/config.py`

Add to `BaseGLiNERConfig.__init__()`:
```python
use_flash_attention: bool = False
```
Add to the `__init__` signature and `self.use_flash_attention = use_flash_attention`.

**Why:** The config is serialised to `gliner_config.json`. Storing `use_flash_attention=True` there means a saved FlashDeBERTa model auto-reloads with the same attention backend — no env var needed.

---

## Step 1.2 — Thread `use_flash_attention` through `Transformer.__init__`

**File:** `gliner/modeling/encoder.py`, `Transformer.__init__`

Replace the env-var check:
```python
# BEFORE (line ~117):
if os.environ.get("USE_FLASHDEBERTA", "") and IS_FLASHDEBERTA:
    ModelClass = FlashDebertaV2Model
else:
    ModelClass = DebertaV2Model

# AFTER:
use_flash = getattr(config, "use_flash_attention", False) or os.environ.get("USE_FLASHDEBERTA", "")
if use_flash and IS_FLASHDEBERTA:
    ModelClass = FlashDebertaV2Model
elif use_flash and not IS_FLASHDEBERTA:
    warnings.warn(
        "use_flash_attention=True requested but 'flashdeberta' is not installed. "
        "Falling back to standard DeBERTa. Install with: pip install flashdeberta",
        UserWarning, stacklevel=2,
    )
    ModelClass = DebertaV2Model
else:
    ModelClass = DebertaV2Model
```

---

## Step 1.3 — Expose `flash_attention` parameter in `from_pretrained` and `load_from_config`

**File:** `gliner/model.py`

Add `flash_attention: bool = False` parameter. Before config loading, inject into config_overrides:
```python
# In from_pretrained():
config = cls._load_config(config_file, ..., use_flash_attention=flash_attention or None)

# In load_from_config():
if flash_attention:
    config_dict["use_flash_attention"] = True
```

---

## Step 1.4 — Extend `max_len` default when FlashDeBERTa is active

**File:** `gliner/config.py`

FlashDeBERTa makes long sequences practical. When `use_flash_attention=True`, the model should default to `max_len=1024` instead of 384. Add a post-init check:
```python
def __post_init__(self):
    if self.use_flash_attention and self.max_len == 384:
        self.max_len = 1024  # safe default for Flash attention
```

---

## Step 1.5 — Benchmark script

**File:** `scripts/benchmark_flash_attention.py`

Measure on `urchade/gliner_multi-v2.1`:
- Token lengths: [128, 256, 384, 512, 768, 1024, 2048]
- Backends: standard DeBERTa vs FlashDeBERTa
- Metrics: mean latency (20 runs), peak memory (MB), first-token-failure rate
- Output: `results/flash_attention_benchmark.csv` + plot

---

## Step 1.6 — Documentation

**File:** `docs/flash_attention.md`

Sections: Overview, Installation, Usage (one-liner), Benchmark table, Supported architectures, Limitations (FlashDeBERTa only for DebertaV2Config).

**File:** `docs/index.md` — add `flash_attention` entry.

---

## Step 1.7 — Validation

Run `scripts/validate_pruned_model.py`-style check: load original model and FlashDeBERTa model, assert identical predictions on 10 diverse sentences.

**Acceptance criteria:**
- All predictions identical (PASS ✓)
- At least 40% latency improvement at 512 tokens
- At least 200% improvement at 1024 tokens
- No OOM at 2048 tokens on 16GB machine

---

## Commit sequence for Feature 1

```
feat(encoder): add use_flash_attention config field
feat(encoder): route FlashDebertaV2Model via config instead of env var
feat(model): expose flash_attention=True in from_pretrained + load_from_config
feat(config): auto-extend max_len to 1024 when flash_attention is active
feat(scripts): add benchmark_flash_attention.py
docs: add flash_attention.md + update index
```

---

---

# FEATURE 2 — Entity Type Description Conditioning

**Branch:** `feature/entity-descriptions`
**Motivation:** GLiNER currently passes short type labels: `["person", "organization"]`. Research (IBM ZeroNER ACL 2025, OpenBioNER NAACL 2025) shows that passing full natural-language definitions instead — `["a named human individual", "a legally incorporated company, firm, or institution"]` — yields **+10–16% F1** on rare and novel entity types. The GLiNER architecture already tokenises label strings arbitrarily; this is an API + training-data change, not an architecture change.

**Key insight from codebase:** The label encoding path in `UniEncoderSpanProcessor` already tokenises full strings passed as entity types. Longer descriptions just produce more tokens in the prompt sequence — the attention mechanism handles them naturally. The only real constraints are `max_types` (25 types per pass) and prompt sequence length.

---

## Step 2.1 — Add `DescriptionDict` type alias and validation helper

**File:** `gliner/utils.py` (or new `gliner/description_utils.py`)

```python
# Support two calling conventions:
# 1. list of strings:   ["person", "organization"]   (existing)
# 2. list of dicts:     [{"label": "person", "description": "a named human individual"}, ...]  (new)
# 3. dict mapping:      {"person": "a named human individual", ...}  (new)

def normalise_labels(
    labels: Union[List[str], List[Dict[str, str]], Dict[str, str]]
) -> Tuple[List[str], List[str]]:
    """
    Returns (display_names, prompt_strings).
    display_names: what appears in entity["label"] in the output
    prompt_strings: what is tokenised and encoded as the entity type token sequence
    """
```

When a description is provided, `prompt_string = f"{label}: {description}"` (colon-space separator, validated by ZeroNER paper to be optimal for DeBERTa-family models).

---

## Step 2.2 — Thread `normalise_labels` into all inference entry points

**File:** `gliner/model.py`

All `predict_entities` / `batch_predict_entities` / `inference` calls that accept an `entity_types` or `labels` argument need to call `normalise_labels` at the top, then:
- Pass `prompt_strings` to the model for encoding
- Decode predictions back using `display_names` (so `entity["label"]` is still `"person"`, not `"person: a named human individual"`)

**Files to modify:** Every `inference()` and `predict_entities()` method across `UniEncoderGLiNER`, `BiEncoderGLiNER`, `UniEncoderSpanDecoderGLiNER`, `UniEncoderSpanRelexGLiNER`, `UniEncoderTokenRelexGLiNER`.

---

## Step 2.3 — Training data format extension

**File:** `gliner/data_processing/processor.py`

The training data JSON format currently uses `"ner": [[start, end, "type"]]`. Extend to support:
```json
{
  "tokenized_text": ["Apple", "was", "founded", ...],
  "ner": [[0, 0, "organization"]],
  "entity_descriptions": {
    "organization": "a legally incorporated company, firm, or institution"
  }
}
```

In `batch_generate_class_mappings`, if `entity_descriptions` is present in a batch item, replace the raw label string with `f"{label}: {description}"` before tokenisation.

---

## Step 2.4 — Add `max_description_length` to config

**File:** `gliner/config.py`

```python
max_description_length: Optional[int] = None  # None = unlimited, int = truncate
```

Truncation applied in `normalise_labels` before tokenisation. Warn if truncation occurs.

---

## Step 2.5 — Built-in description library (optional quality-of-life)

**File:** `gliner/descriptions.py`

A curated dict of high-quality descriptions for the 50 most common NER types (CoNLL, OntoNotes, WNUT-17 label sets), sourced from the ZeroNER paper's appendix. Users can do:
```python
from gliner.descriptions import ONTONOTES_DESCRIPTIONS
entities = model.predict_entities(text, ONTONOTES_DESCRIPTIONS)
```

---

## Step 2.6 — Evaluation script

**File:** `scripts/eval_descriptions.py`

Compare on WNUT-17 zero-shot:
- Baseline: short labels (`["emerging entity", "person", ...]`)
- With descriptions: ZeroNER-style definitions
- Report per-type F1 delta (heatmap)

Expected: +10–16% F1 on rare types (`creative-work`, `group`, `product`).

---

## Step 2.7 — Documentation

**File:** `docs/entity_descriptions.md`

Sections: Motivation, API (3 calling conventions), Training data format, Built-in description library, Benchmark results.

---

## Commit sequence for Feature 2

```
feat(utils): add normalise_labels() supporting description dicts and string lists
feat(model): thread description-aware label encoding through all inference methods
feat(processor): support entity_descriptions field in training JSON
feat(config): add max_description_length config field
feat: add gliner/descriptions.py with curated OntoNotes/WNUT/CoNLL description library
feat(scripts): add eval_descriptions.py benchmarking script
docs: add entity_descriptions.md + update index
```

---

---

# FEATURE 3 — Sliding-Window Long-Document Inference

**Branch:** `feature/sliding-window`
**Motivation:** GitHub Issue #95 (long context) and Discussion #113 (max_length) are the most-discussed limitations in the upstream repo. The 384-token limit is hardcoded in the config and causes severe F1 drops on documents longer than a few sentences. Users are rolling their own broken chunking logic. This feature adds a proper, built-in implementation.

**Architecture decision:** Implemented as a new method `predict_entities_long()` on `BaseEncoderGLiNER`, not a replacement for `predict_entities()`. This preserves backward compatibility and lets users explicitly opt in.

---

## Step 3.1 — Core chunking utility

**File:** `gliner/long_doc.py` (new file)

```python
def chunk_text_tokens(
    tokens: List[str],
    max_tokens: int,
    stride: int,
    min_chunk_size: int = 1,
) -> List[Tuple[int, int]]:
    """
    Yield (start_idx, end_idx) token ranges.
    stride < max_tokens creates overlapping chunks.
    """

def merge_entities(
    chunk_entities: List[List[Dict]],
    chunk_offsets: List[int],
    dedup_strategy: str = "max_score",  # or "first", "last"
) -> List[Dict]:
    """
    Merge entity lists from overlapping chunks.
    
    Deduplication: spans with identical (start, end, label) across chunks
    keep the one with the highest score (dedup_strategy="max_score").
    
    Boundary handling: entities whose span crosses a chunk boundary
    (start in one chunk, end in the next) are only surfaced if they
    appear in both the current chunk and the overlapping next chunk.
    """
```

---

## Step 3.2 — `predict_entities_long()` on `BaseEncoderGLiNER`

**File:** `gliner/model.py`

```python
def predict_entities_long(
    self,
    text: str,
    labels: List[str],
    threshold: float = 0.5,
    max_tokens: int = 384,
    stride: int = 128,
    flat_ner: bool = True,
    multi_label: bool = False,
    dedup_strategy: str = "max_score",
) -> List[Dict]:
    """
    Run entity extraction on texts longer than max_len using a sliding window.

    Args:
        text: Input text of arbitrary length.
        labels: Entity type labels (or description dicts — Feature 2 compatible).
        threshold: Confidence threshold.
        max_tokens: Tokens per window. Defaults to model's max_len.
        stride: Step size between windows. stride < max_tokens creates overlap.
                Recommended: stride = max_tokens // 3.
        flat_ner: If True, resolve overlapping entities by score.
        multi_label: If True, allow the same span to have multiple labels.
        dedup_strategy: How to handle spans predicted in multiple overlapping windows.
                        "max_score" keeps the highest-confidence prediction.

    Returns:
        List of entity dicts with char-level start/end positions, label, and score.
    """
```

Algorithm:
1. Tokenise `text` with the model's word splitter
2. Generate non-overlapping or overlapping token windows via `chunk_text_tokens`
3. For each chunk: call `predict_entities(chunk_text, labels, ...)` with the standard pipeline
4. Remap char offsets back to the full document
5. Call `merge_entities` to deduplicate

---

## Step 3.3 — `batch_predict_entities_long()` variant

**File:** `gliner/model.py`

Same as above but accepts `List[str]` and processes chunks in batches for GPU efficiency. Chunks from different documents are packed into the same batch.

---

## Step 3.4 — Config integration

**File:** `gliner/config.py`

```python
default_stride_ratio: float = 0.33  # stride = max_len * stride_ratio
```

---

## Step 3.5 — Benchmark on long documents

**File:** `scripts/benchmark_long_doc.py`

Dataset: CUAD (Contract Understanding Atticus Dataset — avg 9,000 tokens per document). Compare:
- Truncated baseline (384 tokens, entity recall = 0 after token 384)
- Naive chunking (no overlap, entities at boundaries lost)
- Sliding window (this feature, stride=128)

Metrics: entity recall at various document lengths, F1 on first 384 vs 512-768 vs 768+ token regions.

---

## Step 3.6 — Documentation

**File:** `docs/long_document_inference.md`

Sections: Why the 384-token limit exists, Sliding-window algorithm diagram, API reference, Recommended stride/overlap values for different document types, Performance characteristics.

---

## Commit sequence for Feature 3

```
feat: add gliner/long_doc.py with chunk_text_tokens + merge_entities utilities
feat(model): add predict_entities_long() on BaseEncoderGLiNER
feat(model): add batch_predict_entities_long() for batched long-doc inference
feat(config): add default_stride_ratio config field
feat(scripts): add benchmark_long_doc.py
docs: add long_document_inference.md + update index
```

---

---

# FEATURE 4 — Hard Negative Sampling

**Branch:** `feature/hard-negatives`
**Motivation:** arXiv:2402.16602 shows that semantically confusable entity types make far better training negatives than random types. E.g., when the positive type is "Medication", using "Chemical Compound" or "Drug Class" as negatives forces the model to learn finer-grained distinctions. GLiNER's current `get_negatives()` in `data_processing/utils.py` just does `random.sample` from all types in the batch — zero semantic awareness.

**Current implementation (from code read):**
```python
# gliner/data_processing/utils.py:58
def get_negatives(batch_list, sampled_neg=5, key="ner"):
    element_types = set()
    for b in batch_list:
        types = {el[-1] for el in b.get(key, [])}
        element_types.update(types)
    return random.sample(list(element_types), k=min(sampled_neg, len(element_types)))
```

---

## Step 4.1 — Type similarity index

**File:** `gliner/training/hard_negatives.py` (new)

```python
class TypeSimilarityIndex:
    """
    Builds a semantic similarity matrix over entity type strings using a
    small sentence encoder (default: all-MiniLM-L6-v2, 22M params).
    
    Given a type "Medication", returns nearest neighbour types sorted by
    cosine similarity — these are the "hard" negatives.
    
    Falls back to random sampling if sentence_transformers is not installed.
    """
    
    def __init__(
        self,
        encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        cache_dir: Optional[str] = None,
    ):
        ...
    
    def build(self, all_types: List[str]) -> None:
        """Encode all types and build a cosine similarity matrix."""
        ...
    
    def get_hard_negatives(
        self,
        positive_types: List[str],
        n: int,
        exclude: Optional[Set[str]] = None,
    ) -> List[str]:
        """Return n types that are semantically closest to positive_types but not in them."""
        ...
    
    def save(self, path: str) -> None: ...
    def load(self, path: str) -> None: ...
```

---

## Step 4.2 — Replace `get_negatives` with hard-negative-aware version

**File:** `gliner/data_processing/utils.py`

```python
def get_negatives(
    batch_list: List[Dict],
    sampled_neg: int = 5,
    key: str = "ner",
    similarity_index: Optional["TypeSimilarityIndex"] = None,
    hard_negative_ratio: float = 0.5,
) -> List[str]:
    """
    Sample negative entity types.
    
    If similarity_index is provided and hard_negative_ratio > 0, a fraction
    of negatives are drawn from semantically similar types (hard negatives)
    and the remainder from random sampling (easy negatives). The mix prevents
    over-specialisation to the similarity index.
    
    hard_negative_ratio=0.0 → original random-only behaviour (no regression).
    hard_negative_ratio=1.0 → all negatives are hard (experimental).
    Recommended: 0.5.
    """
```

---

## Step 4.3 — Wire into `TrainingArguments`

**File:** `gliner/training/trainer.py`

Add:
```python
hard_negative_ratio: float = 0.0        # 0 = random (default, no change), 0.5 = recommended
hard_negative_encoder: str = "sentence-transformers/all-MiniLM-L6-v2"
hard_negative_cache_dir: Optional[str] = None
```

In the custom Trainer, build the `TypeSimilarityIndex` once at training start (after the first data scan), then pass it to `get_negatives` in each batch.

---

## Step 4.4 — Type taxonomy integration (optional enhancement)

**File:** `gliner/training/type_taxonomy.py`

For OntoNotes 18-class and CoNLL-4 label sets, provide a hand-curated confusion matrix (which types look similar to which). This is used as a fallback when `sentence_transformers` is not installed but `hard_negative_ratio > 0`.

---

## Step 4.5 — Ablation script

**File:** `scripts/ablation_hard_negatives.py`

Train 5 configs (200 steps on CoNLL-2003):
- `hard_negative_ratio=0.0` (random, baseline)
- `hard_negative_ratio=0.25`
- `hard_negative_ratio=0.50` (recommended)
- `hard_negative_ratio=0.75`
- `hard_negative_ratio=1.00` (full hard)

Evaluate zero-shot on WNUT-17. Expected: peak F1 at ratio ≈ 0.5.

---

## Commit sequence for Feature 4

```
feat(training): add TypeSimilarityIndex for semantic hard negative mining
feat(data): extend get_negatives() with hard_negative_ratio parameter
feat(training): add hard_negative_ratio + hard_negative_encoder to TrainingArguments
feat(training): add OntoNotes/CoNLL type taxonomy fallback
feat(scripts): add ablation_hard_negatives.py
docs: add hard_negative_sampling.md + update training.md
```

---

---

# FEATURE 5 — Label-Aware Contrastive Loss

**Branch:** `feature/contrastive-loss`
**Motivation:** arXiv:2404.17178 adds a contrastive objective over span representations using the entity type label as the anchor. Spans of the same type should be closer in embedding space than spans of different types. This is applied as an auxiliary loss on top of the existing BCE/Focal/Dice loss and yields **+7% avg micro-F1** in few-shot NER settings without changing the model architecture.

**Mathematical formulation:**
Given span embeddings `{s_i}` with labels `{y_i}`:
```
L_contrastive = -1/|P(i)| Σ_{p∈P(i)} log [ exp(sim(s_i,s_p)/τ) / Σ_{a≠i} exp(sim(s_i,s_a)/τ) ]
```
Where `P(i)` = set of spans with the same label as `i`, `τ` = temperature, `sim` = cosine similarity.

Total loss: `L_total = L_NER + λ * L_contrastive`

---

## Step 5.1 — Implement `span_contrastive_loss`

**File:** `gliner/modeling/loss_functions.py`

```python
def span_contrastive_loss(
    span_embeddings: torch.Tensor,   # (B, N_spans, d)
    span_labels: torch.Tensor,       # (B, N_spans) — integer class IDs, -1 = ignored
    temperature: float = 0.07,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Supervised contrastive loss over span representations.
    
    Only positive spans (span_labels != -1) participate in the contrastive objective.
    For each anchor positive span, pulls same-type spans together and pushes
    different-type spans apart in the embedding space.
    
    Args:
        span_embeddings: L2-normalised span representation vectors.
        span_labels: Integer entity type ID per span. -1 = no entity (excluded).
        temperature: Logit scaling. Lower = sharper distribution. Default: 0.07.
        reduction: "mean" or "sum".
    
    Returns:
        Scalar contrastive loss.
    """
```

---

## Step 5.2 — Expose span embeddings from the forward pass

**File:** `gliner/modeling/base.py`

The `UniEncoderSpanModel.forward()` currently returns only logits. To compute contrastive loss we need the span embedding vectors before the final scoring dot-product. Add `return_span_embeddings: bool = False` to the forward signature. When True, also return `span_embeds` of shape `(B, L×K, d)`.

---

## Step 5.3 — Wire into loss dispatch in `BaseModel._loss()`

**File:** `gliner/modeling/base.py`

After computing `L_NER`:
```python
if self.config.contrastive_loss_coef > 0 and span_embeds is not None:
    L_contrastive = span_contrastive_loss(
        span_embeds,
        span_labels,           # integer class IDs extracted from the label mapping
        temperature=self.config.contrastive_temperature,
    )
    loss = L_NER + self.config.contrastive_loss_coef * L_contrastive
```

---

## Step 5.4 — Add contrastive loss config fields

**File:** `gliner/config.py`

```python
contrastive_loss_coef: float = 0.0      # 0 = disabled (default, no regression)
contrastive_temperature: float = 0.07
```

---

## Step 5.5 — Add to `TrainingArguments`

**File:** `gliner/training/trainer.py`

```python
contrastive_loss_coef: float = 0.0
contrastive_temperature: float = 0.07
```

---

## Step 5.6 — Ablation script

**File:** `scripts/ablation_contrastive_loss.py`

Sweep `contrastive_loss_coef` ∈ {0.0, 0.05, 0.1, 0.2, 0.5} on WNUT-17 zero-shot. Expected peak around 0.1–0.2.

---

## Commit sequence for Feature 5

```
feat(loss): implement span_contrastive_loss in loss_functions.py
feat(model): expose return_span_embeddings flag in UniEncoderSpanModel.forward
feat(model): wire contrastive loss into BaseModel._loss() dispatch
feat(config): add contrastive_loss_coef + contrastive_temperature config fields
feat(training): add contrastive_loss_coef to TrainingArguments
feat(scripts): add ablation_contrastive_loss.py
docs: add contrastive_loss.md + update training.md
```

---

---

# FEATURE 6 — ModernBERT Backbone

**Branch:** `feature/modernbert`
**Motivation:** ModernBERT (Dec 2024, answer.ai / HuggingFace) is a 2T-token-trained encoder with native Flash Attention (via flex_attn) and 8,192-token context. It outperforms DeBERTa-v3 on many NLP benchmarks. Knowledgator has `modern-gliner-bi-large-v1.0` as proof-of-concept. The upstream encoder.py already has a `_forward_modernbert` path (discovered in code read), but the ONNX export is broken (Issue #237).

**Current state in codebase:**
- `encoder.py` has `_forward_modernbert()` for packed attention (packing mode)
- Regular ModernBERT forward falls through `AutoModel` path — works for inference
- ONNX export fails because ModernBERT uses `flex_attn` ops not in ONNX opset 19
- No benchmark, no documentation, no config validation

---

## Step 6.1 — Config validation for ModernBERT

**File:** `gliner/config.py`

When `model_name` contains "ModernBERT" or "modernbert", automatically:
- Set `max_len = min(max_len, 8192)` (ModernBERT's maximum)
- Warn if `_attn_implementation` is set to something incompatible
- Suggest `use_flash_attention=False` (ModernBERT has its own attention, no flashdeberta needed)

---

## Step 6.2 — Fix ONNX export for ModernBERT

**File:** `gliner/model.py`

The `_create_onnx_wrapper` and `_run_torch_onnx_export` methods need to:
1. Detect ModernBERT backbone
2. Force `_attn_implementation="eager"` during ONNX export (same pattern as the packed-attention workaround already in `_forward_modernbert`)
3. Use `torch.onnx.export(dynamo=False, opset=14)` for ModernBERT (flex_attn not in opset 19)

---

## Step 6.3 — ModernBERT + Vocab Pruning integration

**File:** `scripts/prune_gliner_vocab.py`

ModernBERT uses a different tokenizer (tiktoken-based BPE, 50,368-token vocabulary). The pruning engine's `_prune_tokenizer_json` currently assumes Unigram model type. Add detection and a BPE-specific pruning path:
```python
if model_type == "BPE":
    _prune_bpe_tokenizer_json(tok_json_path, keep_ids, old_to_new)
```

---

## Step 6.4 — Benchmark: ModernBERT vs DeBERTa-v3

**File:** `scripts/benchmark_modernbert.py`

Compare `urchade/gliner_small-v2.1` (DeBERTa-v3-small) vs `knowledgator/modern-gliner-bi-base-v1.0` (ModernBERT):
- WNUT-17 / CoNLL-2003 zero-shot F1
- Latency at 384 / 1024 / 2048 / 4096 tokens
- Model size (MB)

---

## Step 6.5 — Documentation

**File:** `docs/modernbert_backbone.md`

Sections: Why ModernBERT, How to load, Context window differences, ONNX export (with the eager-mode note), Benchmark table.

---

## Commit sequence for Feature 6

```
feat(config): add ModernBERT config validation and max_len guard
fix(onnx): force eager attention during ModernBERT ONNX export
feat(prune): add BPE tokenizer pruning path for ModernBERT vocab
feat(scripts): add benchmark_modernbert.py
docs: add modernbert_backbone.md + update architectures.md
```

---

---

# FEATURE 7 — Joint NER + Relation Extraction

**Branch:** `feature/joint-ner-re`
**Motivation:** GLiNER-Relex (arXiv:2605.10108) achieves competitive joint NER+RE in one forward pass. The GLiNER-Robust codebase already has `UniEncoderSpanRelexModel`, `RelationsRepLayer`, config classes, and data processors for relation extraction — but there is no:
- Training script for joint NER+RE
- Pre-trained weights on standard RE benchmarks
- Evaluation script on CoNLL04 / FewRel / DocRED
- Documentation

**Current state in codebase:**
- `gliner/modeling/multitask/relations_layers.py` — `RelationsRepLayer` ✅
- `gliner/modeling/multitask/triples_layers.py` — `TriplesScoreLayer` ✅
- `UniEncoderSpanRelexConfig`, `UniEncoderSpanRelexModel`, `UniEncoderSpanRelexGLiNER` ✅
- `RelationExtractionSpanProcessor` ✅
- No training script, no benchmarks

---

## Step 7.1 — Training script for joint NER+RE

**File:** `scripts/train_relex.py`

```python
# Load CoNLL04 or a custom annotated dataset
# Supports training data format:
# {
#   "tokenized_text": [...],
#   "ner": [[start, end, "entity_type"], ...],
#   "relations": [[head_start, head_end, "entity_type", tail_start, tail_end, "entity_type", "relation_type"], ...]
# }
```

---

## Step 7.2 — Zero-shot RE inference API

**File:** `gliner/model.py` (on `UniEncoderSpanRelexGLiNER`)

Add:
```python
def predict_relations(
    self,
    text: str,
    entity_types: List[str],
    relation_types: List[str],
    threshold: float = 0.5,
) -> List[Dict]:
    """
    Returns list of:
    {
        "head": {"text": ..., "label": ..., "start": ..., "end": ...},
        "relation": "founded_by",
        "tail": {"text": ..., "label": ..., "start": ..., "end": ...},
        "score": 0.87
    }
    """
```

---

## Step 7.3 — Evaluation on standard RE benchmarks

**File:** `scripts/eval_relex.py`

Benchmarks: CoNLL04, FewRel, DocRED (subset).
Report: Entity F1, Relation F1 (strict), Relation F1 (partial).

---

## Step 7.4 — Documentation

**File:** `docs/relation_extraction.md`

Complete guide: training data format, inference API, benchmark results, comparison with specialized RE models.

---

## Commit sequence for Feature 7

```
feat(scripts): add train_relex.py for joint NER+RE training
feat(model): add predict_relations() method on UniEncoderSpanRelexGLiNER
feat(scripts): add eval_relex.py with CoNLL04/FewRel benchmarking
docs: add relation_extraction.md + update index
```

---

---

# FEATURE 8 — Curriculum Learning Sampler

**Branch:** `feature/curriculum-learning`
**Motivation:** Multiple 2024-2025 papers show training on easy spans first (short, frequent, unambiguous entity types) then progressively harder spans (long, nested, rare types) consistently improves final F1. Implemented as a custom PyTorch `Sampler` — no model changes required. The difficulty signal can be computed from training data statistics before training starts (zero additional inference cost).

---

## Step 8.1 — Span difficulty scorer

**File:** `gliner/training/curriculum.py`

```python
class SpanDifficultyScorer:
    """
    Assigns a difficulty score ∈ [0, 1] to each training example based on:
    
    1. Entity type frequency: rare types → harder (types appearing < threshold times)
    2. Span length: longer spans → harder (normalized by max_width=12)
    3. Span density: more entities per sentence → harder (more ambiguous context)
    4. Label set size: more entity types in the example → harder
    
    Difficulty = weighted combination:
        d = w1 * type_rarity + w2 * span_length + w3 * span_density + w4 * label_set_size
    
    All components normalized to [0, 1] across the training set.
    """
    
    def __init__(
        self,
        type_rarity_weight: float = 0.4,
        span_length_weight: float = 0.2,
        span_density_weight: float = 0.2,
        label_set_weight: float = 0.2,
    ):
        ...
    
    def fit(self, dataset: List[Dict]) -> None:
        """Compute difficulty scores for all examples. Called once before training."""
        ...
    
    def get_scores(self) -> np.ndarray:
        """Return difficulty score array aligned with dataset indices."""
        ...
```

---

## Step 8.2 — `CurriculumSampler`

**File:** `gliner/training/curriculum.py`

```python
class CurriculumSampler(torch.utils.data.Sampler):
    """
    Progressive curriculum sampler. In epoch 1, samples from the easiest
    fraction (curriculum_start_pct) of examples. By epoch curriculum_ramp_epochs,
    samples from the full dataset.
    
    After curriculum_ramp_epochs, switches to standard random sampling.
    
    Usage:
        sampler = CurriculumSampler(
            dataset, difficulty_scorer, 
            curriculum_start_pct=0.3,    # start with easiest 30%
            curriculum_ramp_epochs=5,    # reach full dataset by epoch 5
        )
        loader = DataLoader(dataset, sampler=sampler, batch_size=8)
        
        # Call at each epoch:
        sampler.set_epoch(epoch)
    """
    
    def set_epoch(self, epoch: int) -> None:
        """Update the active fraction of the dataset based on current epoch."""
        fraction = min(1.0, self.start_pct + (1.0 - self.start_pct) * epoch / self.ramp_epochs)
        n_active = int(fraction * len(self.dataset))
        self._active_indices = self._sorted_by_difficulty[:n_active]
```

---

## Step 8.3 — Wire into `TrainingArguments` and the custom Trainer

**File:** `gliner/training/trainer.py`

```python
use_curriculum: bool = False
curriculum_start_pct: float = 0.3       # start with easiest 30%
curriculum_ramp_epochs: int = 5         # full difficulty by epoch 5
curriculum_type_rarity_weight: float = 0.4
curriculum_span_length_weight: float = 0.2
curriculum_span_density_weight: float = 0.2
curriculum_label_set_weight: float = 0.2
```

In `GLiNERTrainer.get_train_dataloader()`, if `use_curriculum=True`, replace the default random sampler with `CurriculumSampler`.

---

## Step 8.4 — Ablation script

**File:** `scripts/ablation_curriculum.py`

Compare 3 configs (500 steps on CoNLL-2003):
- No curriculum (random sampling)
- Curriculum (start_pct=0.3, ramp=5 epochs)
- Anti-curriculum (hardest first — control)

Eval on WNUT-17 F1 at 100/200/300/500 steps (convergence curve).

---

## Step 8.5 — Documentation

**File:** `docs/curriculum_learning.md`

Sections: Motivation, Difficulty scoring formula, Configuration, Expected behaviour (convergence curve), Interaction with hard negatives (Features 4+8 are complementary).

---

## Commit sequence for Feature 8

```
feat(training): add SpanDifficultyScorer to curriculum.py
feat(training): add CurriculumSampler to curriculum.py
feat(training): add curriculum_* fields to TrainingArguments
feat(training): wire CurriculumSampler into GLiNERTrainer.get_train_dataloader
feat(scripts): add ablation_curriculum.py
docs: add curriculum_learning.md + update training.md
```

---

---

## Implementation Order & Dependencies

```
Feature 1 (FlashDeBERTa)        ← independent, start immediately
Feature 2 (Descriptions)         ← independent, start after Feature 1
Feature 3 (Sliding Window)       ← best after Feature 1 (Flash enables longer windows)
Feature 4 (Hard Negatives)       ← independent training-side change
Feature 5 (Contrastive Loss)     ← depends on Feature 4 (hard negatives amplify its effect)
Feature 6 (ModernBERT)           ← depends on Feature 1 (ONNX export fix is shared)
Feature 7 (Joint NER+RE)         ← independent, existing code just needs training + docs
Feature 8 (Curriculum)           ← best after Feature 4 (complementary samplers)
```

## Files Touch Map

| File | Features touching it |
|---|---|
| `gliner/config.py` | 1, 2, 3, 5, 6 |
| `gliner/modeling/encoder.py` | 1, 6 |
| `gliner/modeling/base.py` | 5 |
| `gliner/modeling/loss_functions.py` | 5 |
| `gliner/model.py` | 1, 2, 3, 6, 7 |
| `gliner/data_processing/utils.py` | 4 |
| `gliner/data_processing/processor.py` | 2, 4 |
| `gliner/training/trainer.py` | 4, 5, 8 |
| `gliner/training/curriculum.py` (new) | 8 |
| `gliner/training/hard_negatives.py` (new) | 4 |
| `gliner/long_doc.py` (new) | 3 |
| `gliner/descriptions.py` (new) | 2 |

## PR Target

All features target a PR to `urchade/GLiNER` main.
Internal files excluded from PRs: `ROADMAP.md`, `pruning_adr.md`, `results/`, `CLAUDE.md`.
