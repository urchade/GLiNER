# Creating Custom GLiNER Architectures

This guide shows you how to extend GLiNER with custom architectures by implementing your own model, processor, decoder, and high-level wrapper classes.

## Overview {#overview}

A complete GLiNER architecture consists of four main components:

```
┌─────────────────────────────────────────────────────────┐
│                   High-Level Wrapper                    │
│              (e.g., UniEncoderSpanGLiNER)               │
│  - User-facing API (predict_entities, train, etc.)      │
│  - Model instantiation and management                   │
└────────────────┬────────────────────────────────────────┘
                 │
        ┌────────┴────────┬──────────────┬─────────────┐
        │                 │              │             │
┌───────▼───────┐  ┌──────▼─────┐  ┌─────▼─────┐  ┌────▼────┐
│ Configuration │  │   Model    │  │ Processor │  │ Decoder │
│    (Config)   │  │  (Module)  │  │           │  │         │
└───────────────┘  └────────────┘  └───────────┘  └─────────┘
     │                    │              │              │
     │                    │              │              │
     └────────────────────┴──────────────┴──────────────┘
              All components work together
```

### Component Responsibilities

| Component | Purpose | Base Class |
|-----------|---------|------------|
| **Configuration** | Store model hyperparameters and settings | `BaseGLiNERConfig` |
| **Model** | Neural network architecture (forward pass, loss) | `BaseModel` |
| **Processor** | Data preprocessing and tokenization | `BaseProcessor` |
| **Decoder** | Convert model outputs to entity predictions | `BaseDecoder` |
| **High-Level Wrapper** | User-facing API and orchestration | `BaseGLiNER` |

## Architecture Components {#architecture-components}

### 1. Configuration Class

The configuration class stores all hyperparameters and settings:

```python
from gliner.config import BaseGLiNERConfig

class MyCustomConfig(BaseGLiNERConfig):
    """Configuration for custom GLiNER architecture."""
    
    def __init__(
        self,
        # Custom parameters
        pooling_strategy: str = "mean",
        use_cnn: bool = False,
        cnn_filters: int = 256,
        # Inherit base parameters
        **kwargs
    ):
        super().__init__(**kwargs)
        
        # Add custom parameters
        self.pooling_strategy = pooling_strategy
        self.use_cnn = use_cnn
        self.cnn_filters = cnn_filters
        
        # Set model type for identification
        self.model_type = "custom_gliner"
```

### 2. Model Class

The model implements the neural network architecture:

```python
from gliner.modeling.base import BaseModel
from gliner.modeling.outputs import GLiNERBaseOutput
import torch
from torch import nn

class MyCustomModel(BaseModel):
    """Custom GLiNER model architecture."""
    
    def __init__(self, config, from_pretrained=False, cache_dir=None):
        super().__init__(config, from_pretrained, cache_dir)
        
        # Initialize your custom layers
        self.encoder = ...  # Your encoder
        self.classifier = ...  # Your classifier
        
    def forward(self, input_ids, attention_mask, **kwargs):
        """Forward pass through the model."""
        # 1. Encode inputs
        embeddings = self.encoder(input_ids, attention_mask)
        
        # 2. Compute logits
        logits = self.classifier(embeddings)
        
        # 3. Compute loss if labels provided
        loss = None
        if "labels" in kwargs:
            loss = self.loss(logits, kwargs["labels"], **kwargs)
        
        # 4. Return structured output
        return GLiNERBaseOutput(
            logits=logits,
            loss=loss,
            # Add other outputs as needed
        )
    
    def loss(self, logits, labels, **kwargs):
        """Compute training loss."""
        # Implement your loss function
        return torch.nn.functional.binary_cross_entropy_with_logits(
            logits, labels
        )
    
    def get_representations(self, input_ids, attention_mask, **kwargs):
        """Extract intermediate representations."""
        # Return embeddings for analysis/debugging
        return self.encoder(input_ids, attention_mask)
```

### 3. Processor Class

The processor handles data preprocessing:

```python
from gliner.data_processing import BaseProcessor
import torch

class MyCustomProcessor(BaseProcessor):
    """Custom data processor."""
    
    def preprocess_example(self, tokens, ner, classes_to_id):
        """Preprocess a single training example."""
        # Convert tokens and entities to model inputs
        # Return dict with required keys
        return {
            "tokens": tokens,
            "labels": self._create_labels(tokens, ner, classes_to_id),
            # Add other fields as needed
        }
    
    def _create_labels(self, tokens, ner, classes_to_id):
        """Create label tensor from entity annotations."""
        # Implement label creation logic
        pass
    
    def create_batch_dict(self, batch, class_to_ids, id_to_classes):
        """Collate preprocessed examples into a batch."""
        return {
            "tokens": [ex["tokens"] for ex in batch],
            "labels": torch.stack([ex["labels"] for ex in batch]),
            "classes_to_id": class_to_ids,
            "id_to_classes": id_to_classes,
        }
    
    def tokenize_and_prepare_labels(self, batch, prepare_labels, *args, **kwargs):
        """Tokenize inputs and prepare labels for the model."""
        # Tokenize
        tokenized = self.tokenize_inputs(batch["tokens"], batch["classes_to_id"])
        
        # Add labels if needed
        if prepare_labels:
            tokenized["labels"] = self.create_labels(batch)
        
        return tokenized
    
    def create_labels(self, batch):
        """Create labels tensor from batch."""
        # Implement label creation from batch
        pass
```

### 4. Decoder Class

The decoder converts model outputs to predictions:

```python
from gliner.decoding import BaseDecoder

class MyCustomDecoder(BaseDecoder):
    """Custom decoder for model outputs."""
    
    def decode(
        self,
        tokens,
        id_to_classes,
        model_output,
        threshold=0.5,
        **kwargs
    ):
        """Decode model output into entity predictions."""
        # 1. Apply sigmoid/softmax to logits
        probs = torch.sigmoid(model_output)
        
        # 2. Find predictions above threshold
        predictions = []
        for i, sample_probs in enumerate(probs):
            sample_preds = []
            # Extract entities from probabilities
            # Convert to standard format: (start, end, label, score)
            sample_preds.append((start, end, label, score))
            predictions.append(sample_preds)
        
        return predictions
```

### 5. High-Level Wrapper Class

The wrapper ties everything together:

```python
from gliner import BaseGLiNER

class MyCustomGLiNER(BaseGLiNER):
    """High-level wrapper for custom GLiNER architecture."""
    
    # Register component classes
    config_class = MyCustomConfig
    model_class = MyCustomModel
    data_processor_class = MyCustomProcessor
    decoder_class = MyCustomDecoder
    data_collator_class = MyCustomDataCollator  # If needed
    
    def _create_model(self, config, backbone_from_pretrained, cache_dir, **kwargs):
        """Create model instance."""
        return self.model_class(config, backbone_from_pretrained, cache_dir, **kwargs)
    
    def _create_data_processor(self, config, cache_dir, tokenizer=None, **kwargs):
        """Create data processor instance."""
        if tokenizer is None:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(config.model_name, cache_dir=cache_dir)
        
        return self.data_processor_class(config, tokenizer, None)
    
    def resize_embeddings(self):
        """Resize token embeddings if needed."""
        # Implement if you add special tokens
        pass
    
    def inference(self, texts, labels, **kwargs):
        """Run inference on texts."""
        # Use parent implementation or customize
        return super().inference(texts, labels, **kwargs)
    
    def evaluate(self, test_data, **kwargs):
        """Evaluate on test data."""
        # Use parent implementation or customize
        return super().evaluate(test_data, **kwargs)
```
