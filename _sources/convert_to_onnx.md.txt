# ONNX Runtime and OpenVINO

GLiNER can export supported PyTorch checkpoints to ONNX or OpenVINO IR and run
them through the same high-level `predict_entities` and `inference` APIs used by
PyTorch models.

| Runtime | Model artifact | Typical use |
|---|---|---|
| PyTorch | `model.safetensors` or `pytorch_model.bin` | Training and standard inference |
| ONNX Runtime | `.onnx` | Portable CPU or CUDA inference |
| OpenVINO | `.xml` + `.bin`, or `.onnx` | CPU, GPU, NPU, or `AUTO` inference through OpenVINO |

File-backed runtime models still use `gliner_config.json` and the tokenizer
files from the export directory. Both export methods write these files
automatically; keep an OpenVINO `.xml` file beside its matching `.bin` file.

## Installation

For ONNX Runtime CPU support, install the optional `onnx` extra:

```bash
pip install "gliner[onnx]"
```

Install the Python `onnx` package when exporting a PyTorch checkpoint to ONNX:

```bash
pip install "gliner[onnx]" onnx
```

For ONNX Runtime with CUDA execution providers, install the GPU extra:

```bash
pip install "gliner[gpu]"
```

Add `onnx` to the same command if this environment will also export models.
Choose only one of `gliner[onnx]` and `gliner[gpu]`: both runtime packages
provide the same Python module and must not be installed together. The default
`gliner` installation does not install either ONNX Runtime package.

For direct OpenVINO export and inference, install the OpenVINO extra. Direct
OpenVINO conversion does not create an intermediate ONNX model:

```bash
pip install "gliner[openvino]"
```

## Convert a model

Load the source checkpoint with the default PyTorch runtime before exporting:

```python
from gliner import GLiNER

model = GLiNER.from_pretrained("urchade/gliner_small-v2.1")
```

Runtime-backed models cannot be exported again. Keep the original PyTorch
checkpoint if you expect to produce multiple deployment formats.

### Export to ONNX

```python
paths = model.export_to_onnx(
    save_dir="exports/onnx",
    onnx_filename="model.onnx",
    opset=19,
    quantize=False,
)

print(paths["onnx_path"])
```

`export_to_onnx` returns:

```python
{
    "onnx_path": "exports/onnx/model.onnx",
    "quantized_path": None,
}
```

To create an additional dynamically quantized model:

```python
paths = model.export_to_onnx(
    save_dir="exports/onnx",
    onnx_filename="model.onnx",
    quantized_filename="model_int8.onnx",
    quantize=True,
    opset=19,
)

print(paths["quantized_path"])
```

Quantization is best-effort. If ONNX Runtime quantization is unavailable or
conversion fails, GLiNER emits a warning and returns `None` for
`quantized_path`; the regular ONNX model is still retained.

### Export directly to OpenVINO

```python
paths = model.export_to_openvino(
    save_dir="exports/openvino",
    openvino_filename="model.xml",
    compress_to_fp16=False,
)

print(paths["openvino_path"])
print(paths["weights_path"])
```

`export_to_openvino` converts the wrapped PyTorch graph directly and returns:

```python
{
    "openvino_path": "exports/openvino/model.xml",
    "weights_path": "exports/openvino/model.bin",
}
```

Set `compress_to_fp16=True` to let OpenVINO compress floating-point weights to
FP16 while saving. OpenVINO conversion does not use an ONNX opset argument.

### Conversion scripts

The repository also includes command-line conversion scripts:

```bash
# ONNX plus dynamically quantized ONNX
python scripts/convert_to_onnx.py \
    --model_path urchade/gliner_small-v2.1 \
    --save_path exports/onnx \
    --file_name model.onnx \
    --quantized_file_name model_int8.onnx

# OpenVINO IR
python scripts/convert_to_openvino.py \
    --model_path urchade/gliner_small-v2.1 \
    --save_path exports/openvino \
    --file_name model.xml
```

The ONNX script creates the regular and quantized files. Use the Python methods
when application code needs explicit control over quantization or other export
settings.

### Exported files

A typical export produces one of these layouts:

```text
exports/onnx/
├── gliner_config.json
├── model.onnx
├── model_int8.onnx       # only when quantization succeeds
├── tokenizer.json
└── tokenizer_config.json # exact tokenizer files depend on the checkpoint

exports/openvino/
├── gliner_config.json
├── model.xml
├── model.bin
├── tokenizer.json
└── tokenizer_config.json # exact tokenizer files depend on the checkpoint
```

## Run an exported model

Select the backend with `runtime` and select its artifact with
`runtime_model_file`. The artifact filename is resolved relative to the model
directory passed to `from_pretrained`.

### ONNX Runtime on CPU

```python
from gliner import GLiNER

model = GLiNER.from_pretrained(
    "exports/onnx",
    runtime="onnxruntime",
    runtime_model_file="model.onnx",
    local_files_only=True,
)

entities = model.predict_entities(
    "Apple was founded by Steve Jobs in California.",
    ["organization", "person", "location"],
)
```

CPU execution is the default. You may provide the execution provider
explicitly:

```python
model = GLiNER.from_pretrained(
    "exports/onnx",
    runtime="onnxruntime",
    runtime_model_file="model.onnx",
    runtime_options={"providers": ["CPUExecutionProvider"]},
)
```

### ONNX Runtime with custom session settings

```python
import onnxruntime as ort

from gliner import GLiNER

session_options = ort.SessionOptions()
session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
session_options.intra_op_num_threads = 4

model = GLiNER.from_pretrained(
    "exports/onnx",
    runtime="onnxruntime",
    runtime_model_file="model.onnx",
    runtime_options={
        "session_options": session_options,
        "providers": ["CPUExecutionProvider"],
    },
)
```

For CUDA, install `gliner[gpu]` and request the CUDA provider. Including the CPU
provider gives ONNX Runtime a fallback for unsupported operators:

```python
model = GLiNER.from_pretrained(
    "exports/onnx",
    runtime="onnxruntime",
    runtime_model_file="model.onnx",
    runtime_options={
        "providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
    },
)
```

### OpenVINO IR

```python
from gliner import GLiNER

model = GLiNER.from_pretrained(
    "exports/openvino",
    runtime="openvino",
    runtime_model_file="model.xml",
    local_files_only=True,
    runtime_options={
        "device_name": "CPU",
        "config": {},
    },
)

entities = model.predict_entities(
    "Apple was founded by Steve Jobs in California.",
    ["organization", "person", "location"],
)
```

Set `device_name` to a device supported by the local OpenVINO installation,
such as `CPU`, `GPU`, `NPU`, or `AUTO`. OpenVINO compile properties belong in
`runtime_options["config"]`; `map_location` does not select an OpenVINO device.

### Use an ONNX model with OpenVINO

OpenVINO can also compile the ONNX artifact directly, so an IR conversion is
optional:

```python
model = GLiNER.from_pretrained(
    "exports/onnx",
    runtime="openvino",
    runtime_model_file="model.onnx",
    runtime_options={"device_name": "AUTO"},
)
```

### Runtime options reference

| Runtime | Option | Description |
|---|---|---|
| ONNX Runtime | `providers` | Ordered ONNX Runtime execution providers |
| ONNX Runtime | `session_options` | An `onnxruntime.SessionOptions` instance |
| ONNX Runtime | `session` | An already-created `onnxruntime.InferenceSession` |
| OpenVINO | `device_name` | Compilation device; defaults to `CPU` |
| OpenVINO | `config` | OpenVINO compilation properties |
| OpenVINO | `core` | An already-created `openvino.Core` |
| OpenVINO | `compiled_model` | An already-compiled OpenVINO model |

When supplying `session` or `compiled_model`, pass it through `runtime_options`.
These advanced forms are useful when the application owns runtime lifecycle or
caching.

Runtime aliases `onnx` and `ort` map to `onnxruntime`; `ov` maps to `openvino`.
The canonical names used in documentation are `onnxruntime` and `openvino`.

## Supported architectures

ONNX and OpenVINO export use the same architecture-specific graph wrappers:

| Architecture | ONNX | OpenVINO | Notes |
|---|---:|---:|---|
| Uni-encoder span | Yes | Yes | Standard span prediction |
| Uni-encoder token | Yes | Yes | Token-level prediction |
| Bi-encoder span | Yes | Yes | Text and label encoders |
| Bi-encoder token | Yes | Yes | Token-level bi-encoder |
| Relation extraction span | Yes | Yes | Named entity and relation outputs |
| Relation extraction token | Yes | Yes | Token-level entity and relation outputs |
| Generative decoder | No | No | Requires iterative generation |
| Streaming span | No | No | Requires runtime state and cache updates |

Use the PyTorch runtime for unsupported architectures.

## Validate an export

Run the repository smoke test against either backend:

```bash
python test_onnx.py exports/onnx/model.onnx
python test_onnx.py exports/onnx/model.onnx --runtime openvino
python test_onnx.py exports/openvino/model.xml --runtime openvino
```

For application validation, compare entity text, offsets, and labels on a
representative dataset. Floating-point scores may differ slightly between
backends, so compare them with a tolerance rather than exact equality.

## Troubleshooting

### The runtime artifact cannot be found

`runtime_model_file` must name the artifact inside the directory passed as the
first argument:

```python
GLiNER.from_pretrained(
    "exports/openvino",
    runtime="openvino",
    runtime_model_file="model.xml",
)
```

For OpenVINO IR, keep `model.xml` and `model.bin` together with the same stem.

### OpenVINO is not installed

```bash
pip install "gliner[openvino]"
```

### ONNX export reports that `onnx` is missing

```bash
pip install onnx
```

### An architecture cannot be exported or loaded

Generative-decoder and streaming models currently require the PyTorch runtime.
Use a supported uni-encoder, bi-encoder, or relation-extraction checkpoint for
ONNX Runtime or OpenVINO.

### PyTorch-only options are rejected

`variant`, `dtype`, `from_pretrained(..., quantize=...)`,
`compile_torch_model`, and
`low_cpu_mem_usage` configure PyTorch loading and cannot be applied while
loading an exported runtime graph. Choose precision during export or with the
target runtime instead.
