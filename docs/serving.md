# Serving

Production deployment for GLiNER via [Ray Serve](https://docs.ray.io/en/latest/serve/index.html), with dynamic batching, memory-aware batch sizing, precompiled power-of-two batch sizes, and multi-replica scaling. Source lives under [`gliner/serve/`](https://github.com/urchade/GLiNER/tree/main/gliner/serve).

## Installation

**With uv (recommended):**
```bash
uv pip install gliner[serve]
```

**With pip:**
```bash
pip install gliner ray[serve]
```

## Quick Start

### Start Server

```bash
python -m gliner.serve --model urchade/gliner_small-v2.1
```

### Make Predictions

**In-process (vLLM-style, recommended):**
```python
from gliner.serve import GLiNERFactory

with GLiNERFactory(model="urchade/gliner_small-v2.1") as llm:
    outputs = llm.predict(
        ["John works at Google", "Paris is in France"],
        labels=["person", "organization", "location"],
    )
    print(outputs)
```

`GLiNERFactory` bundles config → deploy → client into one lifecycle-managed
object. Passing a list of texts preserves dynamic batching — each text is
dispatched as a separate request so Ray Serve's ``@serve.batch`` accumulates
them into a single forward pass. Use `predict_async` for concurrent calls
from `asyncio`.

**Remote Python client** (attach to a running deployment):
```python
from gliner.serve import GLiNERClient

client = GLiNERClient()
result = client.predict(
    "John works at Google in Mountain View",
    labels=["person", "organization", "location"]
)
print(result)
# {'entities': [
#     {'start': 0, 'end': 4, 'text': 'John', 'label': 'person', 'score': 0.95},
#     {'start': 15, 'end': 21, 'text': 'Google', 'label': 'organization', 'score': 0.92},
#     {'start': 25, 'end': 38, 'text': 'Mountain View', 'label': 'location', 'score': 0.89}
# ]}
```

**HTTP request:**
```bash
curl -X POST http://localhost:8000/gliner \
  -H "Content-Type: application/json" \
  -d '{"text": "John works at Google", "labels": ["person", "organization"]}'
```

## Configuration Options

### Basic
```bash
python -m gliner.serve \
    --model urchade/gliner_small-v2.1 \
    --device cuda \
    --dtype bfloat16
```

### Performance
```bash
python -m gliner.serve \
    --model urchade/gliner_small-v2.1 \
    --enable-flashdeberta \
    --enable-sequence-packing \
    --max-batch-size 64
```

### Multi-replica
```bash
python -m gliner.serve \
    --model urchade/gliner_small-v2.1 \
    --num-replicas 4 \
    --num-gpus-per-replica 1
```

## Programmatic Usage

Preferred (vLLM-style):

```python
from gliner.serve import GLiNERFactory, GLiNERServeConfig

config = GLiNERServeConfig(
    model="urchade/gliner_small-v2.1",
    device="cuda",
    dtype="bfloat16",
    max_batch_size=32,
    enable_compilation=True,
    enable_flashdeberta=True,
)

llm = GLiNERFactory(config=config)
try:
    result = llm.predict("John works at Google", ["person", "organization"])
finally:
    llm.shutdown()
```

Low-level (direct handle, for advanced use — returns Ray ObjectRefs):

```python
from gliner.serve import GLiNERServeConfig, serve

handle = serve(GLiNERServeConfig(model="urchade/gliner_small-v2.1"))
ref = handle.predict.remote("John works at Google", ["person", "organization"])
result = ref.result()
```

## Relation Extraction

For models that support relation extraction:

```python
result = client.predict(
    "John works at Google",
    labels=["person", "organization"],
    relations=["works_at", "founded_by"]
)
# {'entities': [...], 'relations': [...]}
```

## All CLI Options

```
Model Configuration:
  --model               Model name or path (required)
  --device              cuda or cpu (default: cuda)
  --dtype               float32, float16/fp16, bfloat16/bf16 (default: bfloat16)
                        Weights are loaded directly at this precision; the fp32
                        intermediate is never materialized.
  --quantization        int8 (default: None). For precision changes use --dtype.

Batching:
  --max-batch-size      Max batch size (default: 32)
  --batch-wait-timeout-ms  Batch wait timeout (default: 50)
  --precompiled-batch-sizes  Comma-separated sizes (default: 1,2,4,8,16,32)

Replicas:
  --num-replicas        Number of replicas (default: 1)
  --num-gpus-per-replica  GPUs per replica (default: 1.0)

Performance:
  --enable-flashdeberta    Use FlashDeBERTa backend
  --enable-sequence-packing  Enable sequence packing
  --no-compile             Disable torch.compile

Memory:
  --target-memory-fraction  GPU memory fraction (default: 0.8)

Server:
  --route-prefix        HTTP route (default: /gliner)
  --ray-address         Ray cluster address
```

## Docker

**Build:**
```bash
docker build -t gliner-serve -f gliner/serve/Dockerfile .
```

**Run:**
```bash
docker run --gpus all -p 8000:8000 gliner-serve
```

**With custom model:**
```bash
docker run --gpus all -p 8000:8000 \
  -e GLINER_MODEL=urchade/gliner_medium-v2.1 \
  -e GLINER_ENABLE_FLASHDEBERTA=true \
  gliner-serve
```

**Environment variables:**
| Variable | Default | Description |
|----------|---------|-------------|
| `GLINER_MODEL` | `urchade/gliner_small-v2.1` | Model name or path |
| `GLINER_DEVICE` | `cuda` | Device (cuda/cpu) |
| `GLINER_DTYPE` | `bfloat16` | Data type |
| `GLINER_MAX_BATCH_SIZE` | `32` | Max batch size |
| `GLINER_NUM_REPLICAS` | `1` | Number of replicas |
| `GLINER_MEMORY_FRACTION` | `0.8` | GPU memory fraction |
| `GLINER_QUANTIZATION` | - | Quantization (`int8` only; use `GLINER_DTYPE` for precision) |
| `GLINER_ENABLE_FLASHDEBERTA` | `false` | Enable FlashDeBERTa |
| `GLINER_ENABLE_PACKING` | `false` | Enable sequence packing |
| `GLINER_DISABLE_COMPILE` | `false` | Disable torch.compile |
| `GLINER_ROUTE_PREFIX` | `/gliner` | HTTP route prefix |

## Shutdown

```python
from gliner.serve import shutdown
shutdown()
```
