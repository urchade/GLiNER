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

client = GLiNERClient()  # defaults to http://localhost:8000/gliner
result = client.predict(
    "John works at Google in Mountain View",
    labels=["person", "organization", "location"],
)
print(result)
# {'entities': [
#     {'start': 0, 'end': 4, 'text': 'John', 'label': 'person', 'score': 0.95},
#     {'start': 15, 'end': 21, 'text': 'Google', 'label': 'organization', 'score': 0.92},
#     {'start': 25, 'end': 38, 'text': 'Mountain View', 'label': 'location', 'score': 0.89}
# ]}
```

`GLiNERClient` is a pure HTTP client built on the Python standard library —
it does **not** import `ray` and does **not** join the Ray cluster, so it
runs from any Python process (including environments where `ray` is not
installed). Construct it with a custom URL/prefix or timeout as needed:

```python
client = GLiNERClient(
    base_url="http://gliner.internal:8000",
    route_prefix="/gliner",
    timeout=30.0,
    max_concurrency=32,   # bound on concurrent in-flight HTTP requests
)
```

Passing a list of texts preserves server-side dynamic batching — each text
is dispatched as its own HTTP request concurrently (threads for `predict`,
`asyncio.gather` for `predict_async`) so Ray Serve's `@serve.batch`
coalesces them into a single forward pass:

```python
outputs = client.predict(
    ["John works at Google", "Paris is in France"],
    labels=["person", "organization", "location"],
)  # → list[dict], one per input text
```

Network or server errors surface as `gliner.serve.client.GLiNERClientError`.

**HTTP request (no client library):**
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

GLiNER-RelEx models (e.g. `knowledgator/gliner-relex-large-v0.5`,
`knowledgator/gliner-token-relex-v1.0`) jointly extract entities and the
relations between them in a single forward pass. The server auto-detects
relation support by inspecting `model.config.model_type` and enables the
relex code path when it contains `"relex"` — no extra flag is needed.

### Start a RelEx server

```bash
python -m gliner.serve \
    --model knowledgator/gliner-relex-large-v1.0 \
    --dtype bfloat16 \
    --max-batch-size 16
```

### Predict via the client

```python
from gliner.serve import GLiNERClient

client = GLiNERClient()  # http://localhost:8000/gliner

text = "Bill Gates founded Microsoft in 1975. The company is headquartered in Redmond."

result = client.predict(
    text,
    labels=["person", "organization", "date", "location"],
    relations=["founded", "founded_in", "headquartered_in"],
    threshold=0.5,
    relation_threshold=0.5,
)

for ent in result["entities"]:
    print(f"  {ent['text']} ({ent['label']})")

for rel in result["relations"]:
    head = result["entities"][rel["head"]["entity_idx"]]
    tail = result["entities"][rel["tail"]["entity_idx"]]
    print(f"  {head['text']} --[{rel['relation']}]--> {tail['text']}")
```

For a batched call, pass a list of texts — each one dispatches as its own
request so the server can coalesce them into a single relex forward pass:

```python
results = client.predict(
    [
        "Bill Gates founded Microsoft in 1975.",
        "Apple is headquartered in Cupertino.",
    ],
    labels=["person", "organization", "location", "date"],
    relations=["founded", "founded_in", "headquartered_in"],
)
# results == [ {"entities": [...], "relations": [...]}, {...} ]
```

### In-process (GLiNERFactory)

```python
from gliner.serve import GLiNERFactory

with GLiNERFactory(model="knowledgator/gliner-relex-large-v0.5") as llm:
    out = llm.predict(
        "Bill Gates founded Microsoft in 1975.",
        labels=["person", "organization", "date"],
        relations=["founded", "founded_in"],
    )
```

### HTTP (curl)

```bash
curl -X POST http://localhost:8000/gliner \
  -H "Content-Type: application/json" \
  -d '{
        "text": "Bill Gates founded Microsoft in 1975.",
        "labels": ["person", "organization", "date"],
        "relations": ["founded", "founded_in"],
        "threshold": 0.5,
        "relation_threshold": 0.5
      }'
```

**Response shape for RelEx models:**
```python
{
    "entities":  [{"start", "end", "text", "label", "score"}, ...],
    "relations": [{"relation", "score",
                   "head": {"entity_idx": int, ...},
                   "tail": {"entity_idx": int, ...}}, ...],
}
```
For NER-only models the `"relations"` key is omitted; passing `relations=`
to such a model is a no-op.

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
  --batch-wait-timeout-ms  Batch wait timeout (default: 10)
  --precompiled-batch-sizes  Comma-separated sizes (default: 1,2,4,8,16,32)

Replicas:
  --num-replicas        Number of replicas (default: 1)
  --num-gpus-per-replica  GPUs per replica (default: 1.0)

Performance:
  --enable-flashdeberta    Use FlashDeBERTa backend
  --enable-sequence-packing  Enable sequence packing
  --no-compile             Disable torch.compile

Memory:
  --target-memory-fraction  GPU memory fraction (default: 0.9)
  --memory-overhead-factor  Safety margin on memory estimates (default: 1.3)

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
