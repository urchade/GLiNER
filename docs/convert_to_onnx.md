# ONNX Export & Deployment

## Overview

GLiNER models can be exported to ONNX format for optimized inference across different platforms and frameworks. ONNX (Open Neural Network Exchange) provides:

- **Cross-platform compatibility**: Deploy on web, mobile, embedded systems
- **Optimized inference**: Hardware-specific optimizations and acceleration
- **Production deployment**: Integrate with existing ML infrastructure
- **Reduced dependencies**: Lighter runtime without full PyTorch stack

## Converting Models to ONNX

### Installation

First, ensure you have GLiNER installed with ONNX support:

```bash
pip install gliner[onnx]
```

### Conversion Script

Save the following script as `convert_to_onnx.py`:

```python
import os
import argparse
from gliner import GLiNER

def main(args):
    # Load the GLiNER model
    gliner_model = GLiNER.from_pretrained(args.model_path)
    
    # Export to ONNX format
    gliner_model.export_to_onnx(
        save_dir=args.save_path, 
        onnx_filename=args.file_name, 
        quantized_filename=args.quantized_file_name,
        quantize=args.quantize,
        opset=args.opset,
    )
    
    print(f"Model exported successfully to {args.save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert GLiNER model to ONNX format')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path or HuggingFace model ID (e.g., urchade/gliner_small-v2.1)')
    parser.add_argument('--save_path', type=str, default='./onnx_models',
                        help='Directory to save ONNX model')
    parser.add_argument('--file_name', type=str, default='model.onnx',
                        help='Name of the ONNX model file')
    parser.add_argument('--quantized_file_name', type=str, default='model_quantized.onnx',
                        help='Name of the quantized ONNX model file')
    parser.add_argument('--opset', type=int, default=19,
                        help='ONNX opset version (default: 19)')
    parser.add_argument('--quantize', action='store_true',
                        help='Also create a quantized INT8 version')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    
    main(args)
    print("Done!")
```

### Usage Examples

#### Basic Conversion

Convert a model from HuggingFace Hub:

```bash
python convert_to_onnx.py \
    --model_path urchade/gliner_small-v2.1 \
    --save_path ./onnx_models
```

#### Convert with Quantization

Create both standard and quantized versions:

```bash
python convert_to_onnx.py \
    --model_path urchade/gliner_small-v2.1 \
    --save_path ./onnx_models \
    --quantize
```

#### Convert Local Model

Convert a locally trained or fine-tuned model:

```bash
python convert_to_onnx.py \
    --model_path ./my_finetuned_model \
    --save_path ./onnx_models \
    --file_name my_model.onnx
```

#### Custom Configuration

Specify all parameters:

```bash
python convert_to_onnx.py \
    --model_path knowledgator/gliner-multitask-large-v0.5 \
    --save_path ./production_models \
    --file_name gliner_large.onnx \
    --quantized_file_name gliner_large_int8.onnx \
    --opset 19 \
    --quantize
```

### Parameters Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--model_path` | `str` | **Required** | Path to local model or HuggingFace model ID |
| `--save_path` | `str` | `./onnx_models` | Output directory for ONNX files |
| `--file_name` | `str` | `model.onnx` | Name for the ONNX model file |
| `--quantized_file_name` | `str` | `model_quantized.onnx` | Name for quantized model file |
| `--opset` | `int` | `19` | ONNX opset version (14-19 supported) |
| `--quantize` | `bool` | `False` | Create INT8 quantized version |

### Quantization Benefits

Quantized models (INT8) offer:

- **50-75% smaller file size**: Faster downloads and reduced storage
- **2-4x faster inference on CPU**: Especially on AVX512-capable processors
- **Lower memory usage**: Important for edge deployment
- **Minimal accuracy loss**: Typically < 1% F1 score difference

:::tip When to Use Quantization
Use quantized models for:
- CPU-based production deployments
- Mobile and edge devices
- Bandwidth-constrained environments
- High-throughput scenarios

Use standard models for:
- GPU inference (GPUs are optimized for FP16/FP32)
- Maximum accuracy requirements
- Research and experimentation
:::

### Output Structure

After conversion, your directory will contain:

```
onnx_models/
├── model.onnx              # Standard ONNX model (FP32)
├── model_quantized.onnx    # Quantized model (INT8, if --quantize used)
```

## Running ONNX Models

### Python (Native GLiNER Support)

GLiNER provides native support for loading and running ONNX models:

```python
from gliner import GLiNER

# Load ONNX model
model = GLiNER.from_pretrained(
    "path/to/model",
    load_onnx_model=True,
    onnx_model_file="model.onnx"
)

# Use exactly like PyTorch models
entities = model.predict_entities(
    "Apple Inc. was founded by Steve Jobs.",
    ["organization", "person"]
)
```

#### ONNX Runtime Configuration

Configure ONNX Runtime for optimal performance:

```python
import onnxruntime as ort

# CPU with optimizations
session_options = ort.SessionOptions()
session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
session_options.intra_op_num_threads = 4

model = GLiNER.from_pretrained(
    "path/to/model",
    load_onnx_model=True,
    onnx_model_file="model.onnx",
    session_options=session_options
)

# GPU (CUDA) inference
model = GLiNER.from_pretrained(
    "path/to/model",
    load_onnx_model=True,
    onnx_model_file="model.onnx",
    map_location="cuda"
)
```

### Cross-Platform Frameworks

GLiNER ONNX models are compatible with multiple frameworks and languages:

#### 🦀 Rust: gline-rs

High-performance inference engine for production systems.

```rust
use gline_rs::{GLiNER, Parameters, RuntimeParameters};

let model = GLiNER::<SpanMode>::new(
    Parameters::default(),
    RuntimeParameters::default(),
    "tokenizer.json",
    "model.onnx",
)?;

let input = TextInput::from_str(
    &["Apple Inc. was founded by Steve Jobs."],
    &["organization", "person"],
)?;

let output = model.inference(input)?;
```

**Key Features:**
- 4x faster than Python on CPU
- Memory-safe and thread-safe
- GPU/NPU support via execution providers
- Zero-copy operations
- Production-grade error handling

**Resources:**
- Repository: [github.com/fbilhaut/gline-rs](https://github.com/fbilhaut/gline-rs)
- Crates.io: [crates.io/crates/gline-rs](https://crates.io/crates/gline-rs)
- Documentation: Available in repository

---

#### 🌐 JavaScript/TypeScript: GLiNER.js

Browser and Node.js inference engine.

```javascript
import { Gliner } from 'gliner';

const gliner = new Gliner({
  tokenizerPath: "path/to/tokenizer",
  onnxSettings: {
    modelPath: "model.onnx",
    executionProvider: "webgpu", // or "cpu", "wasm", "webgl"
  },
});

await gliner.initialize();

const results = await gliner.inference({
  texts: ["Apple Inc. was founded by Steve Jobs."],
  entities: ["organization", "person"],
  threshold: 0.5,
});
```

**Key Features:**
- WebGPU/WebGL acceleration in browsers
- Web Workers support for non-blocking inference
- TypeScript definitions included
- WASM multi-threading on compatible browsers
- Node.js support

**Resources:**
- Repository: [github.com/Ingvarstep/GLiNER.js](https://github.com/Ingvarstep/GLiNER.js)
- NPM: [npmjs.com/package/gliner](https://www.npmjs.com/package/gliner)
- Examples: Available in repository

---

#### ⚡ C++: GLiNER.cpp

Lightweight inference for embedded and high-performance systems.

```cpp
#include "GLiNER/model.hpp"

gliner::Config config{12, 512};  // max_width, max_length
gliner::Model model(
    "model.onnx",
    "tokenizer.json",
    config
);

std::vector<std::string> texts = {
    "Apple Inc. was founded by Steve Jobs."
};
std::vector<std::string> entities = {"organization", "person"};

auto output = model.inference(texts, entities);
```

**Key Features:**
- Minimal dependencies (no Python runtime)
- CUDA GPU acceleration
- OpenMP multi-threading
- Low memory footprint
- Direct ONNX Runtime integration

**Resources:**
- Repository: [github.com/Knowledgator/GLiNER.cpp](https://github.com/Knowledgator/GLiNER.cpp)
- Build instructions: See repository README

---

### Framework Comparison

| Framework | Language | Performance (vs Python) | GPU Support | Target Use Case |
|-----------|----------|-------------------------|-------------|-----------------|
| **GLiNER (Python)** | Python | 1x (baseline) | ✅ CUDA | Research, prototyping |
| **gline-rs** | Rust | ~4x faster (CPU) | ✅ CUDA, TensorRT, DirectML | Production servers, microservices |
| **GLiNER.js** | JavaScript | ~2x faster | ✅ WebGPU, WebGL | Web apps, browser extensions |
| **GLiNER.cpp** | C++ | ~3-5x faster (CPU) | ✅ CUDA | Embedded, mobile, native apps |

*Performance estimates based on community benchmarks with different hardware configurations*

## Supported Model Architectures

Not all GLiNER architectures support ONNX export:

| Architecture | ONNX Support | Notes |
|-------------|--------------|-------|
| **UniEncoderSpan** | ✅ Full | Standard span-based models |
| **UniEncoderToken** | ✅ Full | Token-based models |
| **BiEncoderSpan** | ✅ Full | Separate text/label encoders |
| **BiEncoderToken** | ✅ Full | Bi-encoder with token prediction |
| **UniEncoderSpanDecoder** | ❌ Not supported | Generative decoder incompatible with static graphs |
| **UniEncoderSpanRelex** | ✅ Full | Entity + relation extraction |

:::warning Decoder Models
Models with generative decoders (`UniEncoderSpanDecoder`) cannot be exported to ONNX because the decoder requires iterative generation, which is not suitable for static computation graphs. Consider using the encoder-only variants or PyTorch for these models.
:::

## Advanced ONNX Features

### Bi-Encoder Export Options

For bi-encoder models, you can export with pre-computed label embeddings:

```python
gliner_model.export_to_onnx(
    save_dir="./onnx_models",
    from_labels_embeddings=True  # Use pre-computed embeddings mode
)
```

This creates two export variants:
1. **Standard**: Includes both text and label encoders
2. **With embeddings**: Optimized for pre-computed label embeddings

### Custom Opset Versions

Different ONNX runtimes support different opset versions:

```python
# For older ONNX Runtime versions
gliner_model.export_to_onnx(save_dir="./onnx_models", opset=14)

# For latest features (default)
gliner_model.export_to_onnx(save_dir="./onnx_models", opset=19)
```

### Programmatic Export

Export from Python code:

```python
from gliner import GLiNER

model = GLiNER.from_pretrained("urchade/gliner_small-v2.1")

# Export with all options
paths = model.export_to_onnx(
    save_dir="./models",
    onnx_filename="gliner.onnx",
    quantized_filename="gliner_int8.onnx",
    quantize=True,
    opset=19
)

print(f"Standard model: {paths['onnx_path']}")
print(f"Quantized model: {paths['quantized_path']}")
```

## Troubleshooting

### Common Issues

**Issue: "No module named 'onnxruntime'"**
```bash
pip install onnxruntime  # CPU
# or
pip install onnxruntime-gpu  # GPU
```

**Issue: "Quantization failed"**
- Ensure `onnxruntime` includes quantization tools
- Try without `--quantize` flag first
- Check ONNX Runtime version (≥1.10 recommended)

**Issue: "Opset version not supported"**
- Use `--opset 14` for older runtimes
- Update ONNX Runtime: `pip install -U onnxruntime`

### Validation

Test your ONNX model after conversion:

```python
from gliner import GLiNER

# Load ONNX model
onnx_model = GLiNER.from_pretrained(
    "./onnx_models",
    load_onnx_model=True,
    onnx_model_file="model.onnx"
)

# Compare with PyTorch model
pytorch_model = GLiNER.from_pretrained("urchade/gliner_small-v2.1")

text = "Apple Inc. was founded by Steve Jobs."
labels = ["organization", "person"]

# Should produce identical results
onnx_results = onnx_model.predict_entities(text, labels)
pytorch_results = pytorch_model.predict_entities(text, labels)

print("ONNX:", onnx_results)
print("PyTorch:", pytorch_results)
```

## Best Practices

1. **Always validate after conversion**: Test inference on representative samples
2. **Use quantization for CPU deployments**: Significant speedup with minimal accuracy loss
3. **Keep tokenizer files**: ONNX models need the original `tokenizer.json`
4. **Version your exports**: Include model version and opset in filenames
5. **Test target runtime**: Ensure compatibility with your deployment environment
6. **Profile performance**: Measure inference time on actual hardware
7. **Document export settings**: Keep track of quantization and opset versions
