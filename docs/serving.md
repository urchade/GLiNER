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

Label descriptions and per-text label sets use the same API as local inference:

```python
outputs = client.predict(
    ["John works at Google", "Paris is in France"],
    labels=[
        {"person": "A human individual", "organization": "A company or institution"},
        {"location": "A geographical place"},
    ],
)
```

Each dictionary key is returned as the entity label; its value is the prompt encoded
for that text. A single dictionary is shared across every input text. When supplying
a list of dictionaries, its length and order must match the input texts, and each
dictionary contains all labels for its corresponding text. For a single text, put
all labels in one dictionary. See [Label descriptions](usage.md#label-descriptions)
for examples of shared and per-text label sets.

Network or server errors surface as `gliner.serve.client.GLiNERClientError`.

:::{warning}
The prediction response does not include a truncation flag. `--max-model-len`
overrides the loaded model's text-only `config.max_len`; requests above it can
still return HTTP success after GLiNER keeps only the prefix and emits a Python
warning inside the replica. If the service contract requires full-input
coverage, validate or window requests explicitly. See [Input limits and
truncation](usage.md#input-limits-and-truncation) for a preflight that uses the public model API.
:::

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

## Analyze API (`/v1/analyze`)

A second HTTP API runs next to the `POST /gliner` endpoint above: **one `input`, many named tasks**. It follows the Knowledgator unified analyze contract, limited to the two task types GLiNER can run: `span` (NER) and, on relation-extraction checkpoints, `relation`. The `POST /gliner` endpoint is unchanged.

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/gliner/v1/analyze` | One input, many tasks |
| `POST` | `/gliner/v1/analyze/batch` | Many independent items |
| `GET` | `/gliner/v1/models` | Served model id and aliases |
| `GET` | `/gliner/v1/models/{model_id}/capabilities` | Task types, default thresholds, limits |
| `GET` | `/gliner/v1/health`, `/gliner/v1/ready` | Liveness and readiness |
| `GET` | `/gliner/v1/adapter-cache` | PolyLoRA adapter residency |

### Request

```bash
curl -X POST http://localhost:8000/gliner/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{
        "model": "knowledgator/gliner-relex-base-v1.0",
        "input": "Bill Gates founded Microsoft in 1975.",
        "tasks": {
          "people": {"type": "span", "labels": ["person"], "threshold": 0.3},
          "orgs":   {"type": "span", "labels": {"organization": {"title": "company", "metadata": {"id": "ORG"}}},
                     "flat_ner": false},
          "edges":  {"type": "relation", "labels": ["founded"],
                     "entity_labels": ["person", "organization"],
                     "threshold": 0.5, "entity_threshold": 0.4}
        },
        "options": {"adapter_id": "customer-a", "include_usage": true}
      }'
```

```json
{
  "model": "knowledgator/gliner-relex-base-v1.0",
  "results": {
    "people": {"type": "span", "threshold": 0.3,
               "value": [{"label": "person", "text": "Bill Gates", "start": 0, "end": 10, "source": "", "score": 0.99}]},
    "orgs":   {"type": "span", "threshold": 0.5,
               "value": [{"label": "organization", "text": "Microsoft", "start": 19, "end": 28, "source": "", "score": 0.99}]},
    "edges":  {"type": "relation", "threshold": 0.5, "entity_threshold": 0.4,
               "value": [{"head": {"label": "person", "text": "Bill Gates", "start": 0, "end": 10, "source": ""},
                          "relation": "founded",
                          "tail": {"label": "organization", "text": "Microsoft", "start": 19, "end": 28, "source": ""},
                          "score": 0.99}]}
  },
  "usage": {"rows": 3, "tokens": 25, "latency_ms": 40.1}
}
```

`model` is required and must be the served id (`--model-id`, default `--model`) or an alias (`--model-aliases`).

### Task fields

| Field | `span` | `relation` | Meaning |
|---|---|---|---|
| `labels` | required | required | Entity types (`span`) or relation types (`relation`): a string, a list, `{id: description}` or `{id: {title, description, metadata}}` |
| `entity_labels` | – | required | Entity types the relations connect |
| `threshold` | entity cutoff | relation cutoff | Per task; defaults to `--default-threshold` / `--default-relation-threshold` |
| `entity_threshold` | – | entity cutoff | Defaults to `--default-threshold` |
| `flat_ner` (alias `flat`) | default `true` | default `true` | Forbid overlapping spans |
| `multi_label` | default `false` | default `false` | Let one span carry several labels |

Options: `adapter_id` (PolyLoRA adapter for every task of the request), `include_usage`, `include_label_definitions`.

Labels: the model sees the label's `title` (or its id when there is no title), following `--label-template` (default `{title}`; for example `"{title}: {description}"`). Results always report the label **id**. `metadata` is never shown to the model; it comes back with `include_label_definitions`.

### Structured input

`input` can be JSON. It is rendered into one Markdown text, with keys as headings, and every located result reports `source`, the JSON Pointer of the field it came from. Offsets are relative to that field:

```json
"input": {"ticket": "Satya Nadella emailed Microsoft support.", "customer": {"name": "Bill Gates"}}
```

returns spans such as `{"text": "Bill Gates", "start": 0, "end": 10, "source": "/customer/name", ...}`. Spans that overlap a heading or cross two fields are dropped. Media nodes (`$type`) are rejected with `UNSUPPORTED_MODALITY`.

### Batching

Each task becomes one model row: the input with the task's own labels, thresholds and flags. Tasks with different label sets are not merged into one row, because GLiNER attends over the labels and the text together, so merging would change predictions. Instead, all rows of a request go through Ray Serve's dynamic batcher together, alongside the rows of concurrent requests and `POST /gliner` calls. A request with three tasks costs three rows of a shared forward pass. Identical rows are run once, both within a request and across the items of a `/v1/analyze/batch` request.

Inputs longer than the model window (`--max-model-len` words minus the label prompt) run over overlapping windows (`--analyze-window-overlap`, default 0.25), and the results are merged.

### Errors

Requests are validated in full before anything runs. Errors carry a code and the JSON path of the problem:

```json
{"error": {"code": "UNSUPPORTED_TASK_TYPE", "message": "GLiNER does not serve 'classify' tasks", "path": "/tasks/intent/type"}}
```

| Code | HTTP | When |
|---|---|---|
| `INVALID_REQUEST` | 422 | Malformed request, unknown field, bad threshold, missing labels |
| `UNSUPPORTED_TASK_TYPE` | 422 | Any task type other than `span`/`relation`; `relation` on an NER-only model |
| `UNSUPPORTED_FIELD` | 422 | `instruction` (GLiNER is not instruction-conditioned), label `examples`/`aliases` |
| `UNSUPPORTED_MODALITY` | 422 | Image, audio or layout input |
| `INPUT_TOO_LONG` | 413 | Rendered input over `--max-input-chars` |
| `PAYLOAD_TOO_LARGE` | 413 | Body over `--max-request-bytes` |
| `MODEL_NOT_FOUND` | 404 | Unknown `model` |
| `UNKNOWN_ADAPTER` | 404 | Unknown `options.adapter_id` |
| `INTERNAL` | 500 | Execution failure |

In a batch, each item succeeds or fails on its own, and item error paths are prefixed with `/items/{index}`.

### Python

```python
from gliner.serve import GLiNERClient, GLiNERAPIError

client = GLiNERClient()  # model id is read from /v1/models unless given
response = client.analyze(
    {"title": "Satya Nadella", "body": "He leads Microsoft."},
    tasks={"ents": {"type": "span", "labels": ["person", "organization"], "threshold": 0.4}},
)
response["results"]["ents"]["value"]

client.analyze_batch([{"id": "a", "input": "...", "tasks": {...}}, ...])
client.extract_spans("Alice works at Acme", ["person", "organization"], threshold=0.4)
client.extract_relations("Alice works at Acme", ["works at"], ["person", "organization"])
client.capabilities()
```

`GLiNERFactory` (in-process deployment) has the same `analyze`, `analyze_batch`, their `*_async` variants, and `capabilities`. `GLiNERServer.analyze` runs the API without Ray.

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
