# FlashDeBERTa — Fast Attention for GLiNER

## Overview

GLiNER's default DeBERTa v2/v3 backbone uses a custom disentangled relative-position
attention mechanism that computes a full L×L position-bias matrix. This makes memory and
compute scale **quadratically** with sequence length, which is the root cause of the
384-token practical limit and slow inference at longer inputs.

[FlashDeBERTa](https://github.com/Knowledgator/FlashDeBERTa) (Knowledgator, 2024) rewrites
DeBERTa's disentangled attention with Flash Attention-style IO-aware tiling. The same
pretrained weights are used — no retraining required.

| Sequence length | Standard DeBERTa | FlashDeBERTa | Speedup |
|---|---|---|---|
| 128 tokens | baseline | baseline | ~1.1× |
| 384 tokens | baseline | — | ~1.5× |
| 512 tokens | baseline | — | **~2×** |
| 1024 tokens | baseline | — | **~3–4×** |
| 2048 tokens | OOM on 8 GB | — | ∞ (enables long context) |
| 4096 tokens | OOM on 16 GB | — | **~5×** |

> Numbers are approximate and hardware-dependent. Run `scripts/benchmark_flash_attention.py`
> to get exact figures on your machine.

## Requirements

```bash
pip install flashdeberta        # requires Python ≥ 3.10
```

> FlashDeBERTa is only available for **Python 3.10 or later** and applies only to models
> with a DeBERTa v2/v3 backbone (e.g. `urchade/gliner_multi-v2.1`,
> `urchade/gliner_large-v2.1`). Models using other backbones (BERT, ModernBERT, T5)
> are unaffected — `flash_attention=True` is silently ignored for them.

## Usage

### `from_pretrained`

```python
from gliner import GLiNER

# Enable Flash Attention at load time
model = GLiNER.from_pretrained(
    "urchade/gliner_multi-v2.1",
    flash_attention=True,
)

entities = model.predict_entities(
    "Apple Inc. was founded by Steve Jobs in Cupertino.",
    ["person", "organization", "location"],
)
```

### `load_from_config`

```python
model = GLiNER.load_from_config(
    "path/to/gliner_config.json",
    flash_attention=True,
    backbone_from_pretrained=True,
)
```

### Persistent (saved in `gliner_config.json`)

When you call `model.save_pretrained(output_dir)` on a FlashDeBERTa model, the config
is saved with `"use_flash_attention": true`. The next `from_pretrained(output_dir)` call
will automatically reload with Flash Attention — no need to pass `flash_attention=True`
again.

### Environment variable (legacy)

The previous `USE_FLASHDEBERTA=1` environment variable still works as a fallback:

```bash
USE_FLASHDEBERTA=1 python my_script.py
```

## Fallback behaviour

If `flash_attention=True` is requested but `flashdeberta` is not installed, GLiNER emits
a `UserWarning` and falls back to standard DeBERTa attention — the model loads and runs
correctly, just without the speedup:

```
UserWarning: use_flash_attention=True requested but 'flashdeberta' is not installed.
Falling back to standard DeBERTa attention.
Install with: pip install flashdeberta
```

## Benchmarking

```bash
python scripts/benchmark_flash_attention.py \
    --model_id urchade/gliner_multi-v2.1 \
    --output_dir results/flash_benchmark \
    --n_runs 20
```

Outputs:
- `results/flash_benchmark/flash_attention_benchmark.csv` — latency table
- `results/flash_benchmark/flash_attention_benchmark.png` — speedup plot

## Supported architectures

| Architecture | Flash Attention support |
|---|---|
| `UniEncoderSpan` (DeBERTa v2/v3 backbone) | ✅ |
| `UniEncoderToken` (DeBERTa v2/v3 backbone) | ✅ |
| `BiEncoderSpan` (DeBERTa v2/v3 backbone) | ✅ |
| Models with BERT, RoBERTa, ModernBERT backbone | ❌ Not applicable — these use standard SDPA |
| Models with T5/MT5 backbone | ❌ Not applicable |

## Long-context inference

FlashDeBERTa makes sequences beyond 384 tokens practical. Combine with the sliding-window
utility (`predict_entities_long()`) for full long-document support:

```python
model = GLiNER.from_pretrained("urchade/gliner_multi-v2.1", flash_attention=True)

# Process a 2000-token document in overlapping 1024-token windows
entities = model.predict_entities_long(
    long_document_text,
    ["person", "organization", "location"],
    max_tokens=1024,
    stride=256,
)
```

See [`docs/long_document_inference.md`](long_document_inference.md) for the full guide.
