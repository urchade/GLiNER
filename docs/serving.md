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

## CLI Options

The CLI entry point is the quickest way to start a standalone HTTP
deployment:

```bash
python -m gliner.serve --model urchade/gliner_small-v2.1
```

### Common examples

**Basic model settings:**
```bash
python -m gliner.serve \
    --model urchade/gliner_small-v2.1 \
    --device cuda \
    --dtype bfloat16
```

**Performance settings:**
```bash
python -m gliner.serve \
    --model urchade/gliner_small-v2.1 \
    --enable-flashdeberta \
    --enable-sequence-packing \
    --max-batch-size 64
```

**Multi-replica serving:**
```bash
python -m gliner.serve \
    --model urchade/gliner_small-v2.1 \
    --num-replicas 4 \
    --num-gpus-per-replica 1
```

**PolyLoRA serving:**
```bash
python -m gliner.serve \
    --model urchade/gliner_small-v2.1 \
    --enable-polylora \
    --polylora-max-gpu-adapters 8 \
    --polylora-disk-cache-dir /models/polylora-cache
```

### Full option reference

```
Model Configuration:
  --model                         Model name or path (required)
  --device                        cuda or cpu (default: cuda)
  --dtype                         float32, float16/fp16, bfloat16/bf16
                                  (default: bfloat16)
  --quantization                  int8 (default: None). For precision changes,
                                  use --dtype.

Model Limits:
  --max-model-len                 Maximum sequence length (default: 2048)
  --max-span-width                Maximum entity span width (default: 12)
  --max-labels                    Maximum labels per request; -1 is unlimited
                                  (default: -1)

Thresholds:
  --default-threshold             Default entity threshold (default: 0.5)
  --default-relation-threshold    Default relation threshold (default: 0.5)

Replica Configuration:
  --num-replicas                  Number of replicas (default: 1)
  --num-gpus-per-replica          GPUs per replica (default: 1.0)
  --num-cpus-per-replica          CPUs per replica (default: 1.0)

Batching Configuration:
  --max-batch-size                Maximum Ray Serve batch size (default: 32)
  --batch-wait-timeout-ms         Batch wait timeout in milliseconds
                                  (default: 10.0)
  --request-timeout-s             Request timeout in seconds (default: 30.0)
  --max-ongoing-requests          Maximum in-flight requests per replica
                                  (default: 256)
  --queue-capacity                Maximum queued requests (default: 4096)
  --precompiled-batch-sizes       Comma-separated batch sizes to precompile
                                  (default: 1,2,4,8,16,32)

Server Configuration:
  --route-prefix                  HTTP route prefix (default: /gliner)
  --port                          HTTP port (default: 8000)
  --ray-address                   Ray cluster address (default: local)

Performance Options:
  --tokenizer-threads             Tokenizer thread count (default: 4)
  --decoding-threads              Decoding thread count (default: 4)
  --no-compile                    Disable torch.compile precompilation
  --enable-sequence-packing       Enable inference sequence packing
  --enable-flashdeberta           Enable FlashDeBERTa
  --warmup-iterations             Warmup iterations per compiled batch size
                                  (default: 3)

Memory Configuration:
  --target-memory-fraction        Target GPU memory fraction (default: 0.9)
  --memory-overhead-factor        Safety margin on memory estimates
                                  (default: 1.3)

PolyLoRA Configuration:
  --enable-polylora               Enable PolyLoRA adapter serving
  --polylora-adapter-weight-modules
                                  Comma-separated target module names
  --polylora-max-rank             Maximum LoRA rank (default: 16)
  --polylora-max-gpu-adapters     Maximum GPU adapter slots (default: 8)
  --polylora-max-cpu-adapters     Maximum CPU adapters (default: 128)
  --polylora-disk-cache-dir       Disk cache directory for adapters
  --polylora-max-disk-adapters    Maximum disk-cached adapters
  --polylora-base-adapter-id      Reserved base adapter id
                                  (default: __base__)
  --polylora-use-triton-kernels / --no-polylora-use-triton-kernels
                                  Enable or disable PolyLoRA Triton kernels
                                  (default: enabled)
  --polylora-adapter-id-pattern   Regular expression for valid adapter ids
                                  (default: ^[A-Za-z0-9_.-]{1,128}$)
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
    enable_compilation=False,
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

## PolyLoRA

Install polylora first:
```bash
pip install polylora
```

PolyLoRA support lets one deployment route requests through different LoRA
adapters without starting one replica per adapter. Enable it with
`--enable-polylora` or `GLiNERServeConfig(enable_polylora=True)`. The
`polylora` package must be importable in the serving environment.

PolyLoRA is currently implemented for GLiNER text encoder models. During
startup, the server wraps `model.model.token_rep_layer.bert_layer.model` with
`PolyLoraModel`; architectures without that path raise `NotImplementedError`.

### Start with PolyLoRA

```bash
python -m gliner.serve \
    --model urchade/gliner_small-v2.1 \
    --enable-polylora \
    --polylora-max-rank 16 \
    --polylora-max-gpu-adapters 8 \
    --polylora-max-cpu-adapters 128 \
    --polylora-disk-cache-dir /models/polylora-cache
```

Use `--polylora-adapter-weight-modules` when your adapter weights target a
specific set of module names:

```bash
python -m gliner.serve \
    --model urchade/gliner_small-v2.1 \
    --enable-polylora \
    --polylora-adapter-weight-modules query,value
```

### Select an adapter per request

Pass `adapter_id` in Python or HTTP requests. If `adapter_id` is omitted while
PolyLoRA is enabled, the request uses `polylora_base_adapter_id` (default:
`"__base__"`), which means base-model inference.

```python
from gliner.serve import GLiNERClient

client = GLiNERClient()
result = client.predict(
    "John works at Google",
    labels=["person", "organization"],
    adapter_id="customer-a",
)
```

```bash
curl -X POST http://localhost:8000/gliner \
  -H "Content-Type: application/json" \
  -d '{
        "text": "John works at Google",
        "labels": ["person", "organization"],
        "adapter_id": "customer-a"
      }'
```

Adapter ids must match `polylora_adapter_id_pattern` and cannot equal the
reserved base adapter id. Unknown adapter ids return a 404 response.

### Adapter cache status

The client exposes the `/adapter-cache` endpoint:

```python
client.adapter_cache_status()
client.adapter_cache_status("customer-a")
client.is_adapter_cached("customer-a")
```

The raw HTTP endpoint is also available:

```bash
curl http://localhost:8000/gliner/adapter-cache
curl "http://localhost:8000/gliner/adapter-cache?adapter_id=customer-a"
```

The response includes whether PolyLoRA is enabled, the base adapter id, loaded
adapter ids, disk-cached adapter ids when disk cache is configured, and current
GPU adapter slots.

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

## Docker

**Build:**
```bash
docker build -t gliner-serve -f gliner/serve/Containerfile .
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
