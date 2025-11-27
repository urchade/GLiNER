# GLiNER Documentation

**GLiNER** is a framework for training and deploying Named Entity Recognition (NER) models that can identify any entity type using bidirectional transformer encoders (BERT-like). Beyond standard NER, GLiNER supports multiple tasks including joint entity and relation extraction through specialized architectures. It provides a practical alternative to both traditional NER models, which are limited to predefined entity types, and Large Language Models (LLMs), which offer flexibility but require significant computational resources.

This documentation includes installation guides, tutorials, advanced topics, and full API reference.
```{toctree}
:maxdepth: 2
:caption: User Guide

intro
instalation
quickstart
usage
configs
training
architectures
add_custom_architectures
convert_to_onnx
```

```{toctree}
:maxdepth: 2
:caption: API Reference

api/gliner.model
api/gliner.config
api/gliner.training
api/gliner.modeling
api/gliner.data_processing
api/gliner.evaluation
api/gliner.onnx
api/gliner.decoding
api/gliner.utils
```