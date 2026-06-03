# Vocabulary Pruning for Edge Deployment

## Overview

Multilingual GLiNER models (based on mDeBERTa-v3) carry a vocabulary of over 250,000 tokens, resulting in a large `nn.Embedding` matrix. For a model deployed on a single target language, the vast majority of these token embeddings are never accessed at inference time.

The **Vocabulary Pruning Engine** (`scripts/prune_gliner_vocab.py`) solves this by:

1. Tokenising a representative corpus of the target language to discover which token IDs are actually used
2. Building a compact keep-set (active tokens + all special tokens)
3. Slicing the `word_embeddings.weight` tensor to keep only the selected rows
4. Rebuilding the fast tokenizer (`tokenizer.json`) to reflect the new compact vocabulary
5. Saving a fully self-contained pruned model that loads with the standard `GLiNER.from_pretrained()` API

The result is a smaller, faster model with **identical entity predictions** for the target language.

## When to Use

| Use case | Recommended |
|---|---|
| Single-language deployment (e.g. English-only API) | ✅ Yes |
| Edge / embedded CPU inference | ✅ Yes |
| Mobile or serverless environments | ✅ Yes |
| Multilingual deployment (multiple languages simultaneously) | ❌ No — use the original model |
| Research requiring cross-lingual transfer | ❌ No |

## Installation

No additional dependencies are required when using a local text file as the corpus.
To use Wikipedia (recommended for best coverage), install the `datasets` package:

```bash
pip install datasets
```

## Quick Start

### Prune to English using Wikipedia

```bash
python scripts/prune_gliner_vocab.py \
    --model_id urchade/gliner_multi-v2.1 \
    --dataset_for_vocab wikipedia \
    --output_dir ./pruned_model_en \
    --lang en
```

### Prune using a custom text file

Provide a plain `.txt` file with one sentence or paragraph per line:

```bash
python scripts/prune_gliner_vocab.py \
    --model_id urchade/gliner_multi-v2.1 \
    --dataset_for_vocab /path/to/corpus.txt \
    --output_dir ./pruned_model_en
```

### Aggressive pruning with a token cap

Use `--top_k` to keep only the N most frequent tokens (trades off a small amount of accuracy for a larger size reduction):

```bash
python scripts/prune_gliner_vocab.py \
    --model_id urchade/gliner_multi-v2.1 \
    --dataset_for_vocab wikipedia \
    --output_dir ./pruned_model_en_30k \
    --lang en \
    --top_k 30000
```

## CLI Reference

| Argument | Type | Default | Description |
|---|---|---|---|
| `--model_id` | `str` | **required** | HuggingFace model ID or local path |
| `--dataset_for_vocab` | `str` | **required** | `wikipedia` or path to a `.txt` file |
| `--output_dir` | `str` | **required** | Directory to save the pruned model |
| `--top_k` | `int` | `None` (keep all seen) | Cap corpus tokens to the top-K by frequency |
| `--lang` | `str` | `en` | Wikipedia language code (`en`, `fr`, `de`, …) |
| `--max_corpus_texts` | `int` | `100000` | Maximum articles / lines to read from the corpus |

## How It Works

### 1. Corpus Tokenisation

The corpus is tokenised with the model's own tokenizer. A frequency counter records how often each token ID appears across the corpus.

### 2. Keep-Set Construction

The keep-set is the union of:
- All tokens observed in the target-language corpus (or the top-K by frequency if `--top_k` is set)
- All standard HuggingFace special tokens (`[PAD]`, `[UNK]`, `[CLS]`, `[SEP]`, `[MASK]`, …)
- All byte-fallback tokens (required for correct UTF-8 handling of non-ASCII characters)
- All GLiNER-specific tokens (`[FLERT]`, `<<ENT>>`, `<<SEP>>`, and relation tokens where applicable)

### 3. ID Remapping

The keep-set is sorted in ascending order of the **old** token ID. Each old ID maps to a **new** ID equal to its position in this sorted list:

```
keep_ids  = sorted(keep_set)          # e.g. [0, 1, 5, 99, …, V-3, V-2, V-1]
old_to_new = {old: new               # {0→0, 1→1, 5→2, 99→3, …}
              for new, old in enumerate(keep_ids)}
```

### 4. Embedding Tensor Slicing

```python
E_old = model.embeddings.word_embeddings.weight.data  # shape (V, d)
E_new = E_old[keep_ids]                               # shape (K, d)
```

**Invariant:** `E_new[old_to_new[t]] == E_old[t]` for every kept token `t`.
No interpolation, no approximation — exact row selection.

### 5. Tokenizer Surgery

The fast tokenizer (`tokenizer.json`) stores the vocabulary as a plain JSON list where the list index is the token ID. The pruned tokenizer is built by selecting only the kept rows and remapping explicit ID references in `added_tokens`.

### 6. Config Update

`config.vocab_size` and `config.encoder_config.vocab_size` are set to the new compact size `K`, and `config.class_token_index` is remapped via `old_to_new`. This prevents `GLiNER.from_pretrained()` from incorrectly triggering `resize_embeddings()` on the already-pruned model.

## Validation

Use the companion script to verify correctness and measure the size and latency improvement:

```bash
python scripts/validate_pruned_model.py \
    --original_model_id urchade/gliner_multi-v2.1 \
    --pruned_model_dir ./pruned_model_en
```

The validator reports three levels of comparison:

| Status | Meaning |
|---|---|
| `PASS ✓` | Entity sets and scores are identical |
| `SCORE_DRIFT ~` | Same entities detected; confidence scores shift within ±0.02 |
| `ENTITY_FAIL ✗` | Different entities detected — pruning was too aggressive |

For conservative pruning (all seen tokens, no `--top_k`), all test cases should report `PASS ✓`.

### Loading the Pruned Model

The pruned model is a standard GLiNER model directory and loads identically to the original:

```python
from gliner import GLiNER

model = GLiNER.from_pretrained("./pruned_model_en")

entities = model.predict_entities(
    "Apple Inc. was founded by Steve Jobs in Cupertino.",
    ["person", "organization", "location"],
)
```

## Supported Architectures

| Architecture | Pruning Support | Notes |
|---|---|---|
| `UniEncoderSpan` | ✅ Full | Standard span-based models |
| `UniEncoderToken` | ✅ Full | Token-based models |
| `BiEncoderSpan` | ✅ Full | Text encoder pruned; labels encoder pruned if it shares the same vocabulary |
| `BiEncoderToken` | ✅ Full | Same as above |
| `UniEncoderSpanRelex` | ✅ Full | Relation token `<<REL>>` is always kept |
| `UniEncoderSpanDecoder` | ✅ Full | Decoder head is unaffected; only the encoder embedding is pruned |

## Pruning Mode Comparison

Results measured on `urchade/gliner_multi-v2.1` (mDeBERTa-v3, 250,105-token vocabulary)
with 100,000 English Wikipedia articles as the corpus:

| Mode | Vocab retained | Model size | Size reduction | Entity correctness |
|---|---|---|---|---|
| **Conservative** (default, no `--top_k`) | 90,840 / 250,105 **(36.3%)** | 666.5 MB | **42.3%** (1155.8 → 666.5 MB) | **Lossless — ALL PASS ✓** |
| **Aggressive** (`--top_k 30000`) | ~30k / 250,105 **(~12%)** | ~400 MB | **~65%** | Minor score shifts near detection threshold |

> **Note on inference latency:** Because embedding lookup is O(1) regardless of vocabulary size,
> the forward-pass latency on a warm CPU cache changes minimally. The primary benefit is
> **model memory footprint** — smaller cold-start time, reduced RAM usage, and the ability to
> fit on memory-constrained edge devices.

## Troubleshooting

**`tokenizer.json` not found after save:**
The model may use only the slow tokenizer (no fast tokenizer file). Vocabulary pruning requires the fast tokenizer (`tokenizer.json`). Convert your model to the fast tokenizer first:

```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("your/model", use_fast=True)
tok.save_pretrained("your/model")
```

**Labels encoder skipped:**
For BiEncoder models where the labels encoder has a different vocabulary size than the text encoder, the labels encoder is automatically skipped with a warning. This is correct behaviour when the two encoders use different tokenizers.

**`datasets` not installed:**
Wikipedia streaming requires the `datasets` package:
```bash
pip install datasets
```
Alternatively, pass a local `.txt` file to `--dataset_for_vocab`.
