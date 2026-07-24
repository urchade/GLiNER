# GLiNER Documentation

**GLiNER** is a framework for training and deploying Named Entity Recognition (NER) models that can identify any entity type. Its architectures include bidirectional transformer encoders, scalable bi-encoders, relation extraction models, and a causal StreamingSpan model for incremental text. It provides a practical alternative to both traditional NER models, which are limited to predefined entity types, and Large Language Models (LLMs), which offer flexibility but require significant computational resources.

This documentation includes installation guides, tutorials, advanced topics, and full API reference.
```{toctree}
:maxdepth: 2
:caption: User Guide

intro
instalation
quickstart
input_limits
usage
streaming
configs
training
architectures
add_custom_architecture
convert_to_onnx
serving
```

```{toctree}
:maxdepth: 2
:caption: API Reference

api/gliner.model
api/gliner.streaming
api/gliner.config
api/gliner.training
api/gliner.modeling
api/gliner.data_processing
api/gliner.evaluation
api/gliner.onnx
api/gliner.decoding
api/gliner.utils
```
