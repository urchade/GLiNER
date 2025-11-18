# Components & Configs

GLiNER supports multiple architecture variants, each with its own configuration class. This page documents the configuration parameters for each architecture and provides training examples.

## Architecture Overview

| Architecture | Config Class | Use Case |
|-------------|--------------|----------|
| [UniEncoderSpan](#uniencoder-span-configuration) | `UniEncoderSpanConfig` | Standard span-based NER, original GLiNER |
| [UniEncoderToken](#uniencoder-token-configuration) | `UniEncoderTokenConfig` | Token-level NER, long-form extraction |
| [BiEncoderSpan](#biencoder-span-configuration) | `BiEncoderSpanConfig` | Span NER with separate label encoder |
| [BiEncoderToken](#biencoder-token-configuration) | `BiEncoderTokenConfig` | Token NER with separate label encoder |
| [UniEncoderSpanDecoder](#uniencoder-span-decoder-configuration) | `UniEncoderSpanDecoderConfig` | Generative label prediction |
| [UniEncoderSpanRelex](#uniencoder-span-relex-configuration) | `UniEncoderSpanRelexConfig` | Joint entity and relation extraction |

## Base Configuration Parameters

All GLiNER architectures share these base configuration parameters from `BaseGLiNERConfig`:

### Core Parameters

#### `model_name` 
`str`, *optional*, defaults to `"microsoft/deberta-v3-small"`

Base encoder model identifier from Hugging Face Hub or local path.

---

#### `name`
`str`, *optional*, defaults to `"gliner"`

Optional display name for this model configuration.

---

#### `max_width`
`int`, *optional*, defaults to `12`

Maximum span width (in number of tokens) allowed when generating candidate spans. Only applies to span-based architectures.

---

#### `hidden_size`
`int`, *optional*, defaults to `512`

Dimensionality of hidden representations in internal layers.

---

#### `dropout`
`float`, *optional*, defaults to `0.4`

Dropout rate applied to intermediate layers.

---

#### `fine_tune`
`bool`, *optional*, defaults to `True`

Whether to fine-tune the encoder during training.

---

#### `subtoken_pooling`
`str`, *optional*, defaults to `"first"`

Currently only first token pooling is supported. More approaches will be added in the future.

---

#### `span_mode` <sup><a href="https://github.com/urchade/GLiNER/blob/main/gliner/modeling/span_rep.py" target="_blank" rel="noopener noreferrer">[source]</a></sup>
`str`, *optional*, defaults to `"markerV0"`

Defines the strategy for constructing span representations from encoder outputs. Only applies to span-based architectures.

**Available options:**

- `"markerV0"` — Projects the start and end token representations with MLPs, concatenates them, and then applies a final projection. Lightweight and default.
- `"marker"` — Similar to `markerV0` but with deeper two-layer projections; better for complex tasks.
- `"query"` — Uses learned per-span-width query vectors and dot-product interaction.
- `"mlp"` — Applies a feedforward MLP and reshapes output into span format; fast but position-agnostic.
- `"cat"` — Concatenates token features with learned span width embeddings before projection.
- `"conv_conv"` — Uses multiple 1D convolutions with increasing kernel sizes; captures internal structure.
- `"conv_max"` — Max pooling over tokens in span; emphasizes the strongest token.
- `"conv_mean"` — Mean pooling across span tokens.
- `"conv_sum"` — Sum pooling; raw additive representation.
- `"conv_share"` — Shared convolution kernel over span widths; parameter-efficient alternative.
---

#### `post_fusion_schema` <sup><a href="https://github.com/urchade/GLiNER/blob/main/gliner/modeling/layers.py#L298" target="_blank" rel="noopener noreferrer">[source]</a></sup>
`str`, *optional*, defaults to `""`

Defines the multi-step attention schema used to fuse span and label embeddings. The value is a string with hyphen-separated tokens that determine the sequence of attention operations applied in the `CrossFuser` module.

Each token in the schema defines one of the following attention types:

- `"l2l"` — **label-to-label self-attention** (intra-label interaction)
- `"t2t"` — **token-to-token self-attention** (intra-span interaction)
- `"l2t"` — **label-to-token cross-attention** (labels attend to span tokens)
- `"t2l"` — **token-to-label cross-attention** (tokens attend to labels)

**Examples:**

- `"l2l-l2t-t2t"` — apply label self-attention → label-to-token attention → token self-attention
- `"l2t"` — a single step where labels attend to span tokens
- `""` — disables fusion entirely (no interaction is applied)

:::tip
The number of fusion layers (`num_post_fusion_layers`) controls how many times the entire schema is repeated.
:::
---

#### `num_post_fusion_layers`
`int`, *optional*, defaults to `1`

Number of layers applied after span-label fusion.

---

#### `vocab_size`
`int`, *optional*, defaults to `-1`

Vocabulary size override if needed. Automatically set during model initialization.

---

#### `max_neg_type_ratio`
`int`, *optional*, defaults to `1`

Controls the ratio of negative (non-matching) types during training.

---

#### `max_types`
`int`, *optional*, defaults to `25`

Maximum number of entity types supported per batch.

---

#### `max_len`
`int`, *optional*, defaults to `384`

Maximum sequence length accepted by the encoder.

---

#### `words_splitter_type`
`str`, *optional*, defaults to `"whitespace"`

Heuristic used for word-level splitting during inference.  
**Choices:** `"whitespace"`, `"spacy"`, `"moses"`, `stanza`, `universal`

---

#### `num_rnn_layers`
`int`, *optional*, defaults to `1`

Number of LSTM layers to apply on top of encoder outputs. Set to 0 to disable LSTM.

---

#### `fuse_layers`
`bool`, *optional*, defaults to `False`

If `True`, combine representations from multiple encoders (labels and main encoder).

---

#### `embed_ent_token`
`bool`, *optional*, defaults to `True`

If `True`, `<<ENT>>` tokens will be pooled for each label. If `False`, the first token of each label will be pooled as label embedding.

---

#### `class_token_index`
`int`, *optional*, defaults to `-1`

Index of the entity token in the vocabulary. Set automatically during initialization.

---

#### `encoder_config`
`dict` or `PretrainedConfig`, *optional*

A nested config dictionary for the encoder model. If a dict is passed, its `model_type` must be set or inferred.

---

#### `ent_token`
`str`, *optional*, defaults to `"<<ENT>>"`

Special token used to mark entity type boundaries in the input.

---

#### `sep_token`
`str`, *optional*, defaults to `"<<SEP>>"`

Token used to separate entity types from input text.

---

#### `_attn_implementation`
`str`, *optional*

Optional override for attention logic. Can be used to disable Flash Attention if installed.

**Example:**
```python
model = GLiNER.from_pretrained(
    "urchade/gliner_mediumv2.1", 
    _attn_implementation="eager"  # Disable Flash Attention
)
```

---

## UniEncoder Span Configuration

`UniEncoderSpanConfig` is used for the original GLiNER architecture with span-based prediction.

### Architecture-Specific Parameters

This architecture uses all [base parameters](#base-configuration-parameters) without additional architecture-specific parameters.

### Usage Example

```python
from gliner import GLiNERConfig, GLiNER

# Create config for UniEncoderSpan
config = GLiNERConfig(
    model_name="microsoft/deberta-v3-small",
    max_width=12,
    hidden_size=512,
    span_mode="markerV0",
    # labels_encoder=None  # Makes it UniEncoder
    # labels_decoder=None  # No decoder
    # relations_layer=None  # No relations
)

# Initialize model from config
model = GLiNER.from_config(config)
```

### Training Config Example

```yaml
# Model Configuration
model_name: microsoft/deberta-v3-base
labels_encoder: null  # UniEncoder
name: "span level gliner"
max_width: 12
hidden_size: 768
dropout: 0.4
fine_tune: true
subtoken_pooling: first
span_mode: markerV0
post_fusion_schema: ""
num_post_fusion_layers: 1

# Training Parameters
num_steps: 30000
train_batch_size: 8
eval_every: 1000
warmup_ratio: 0.1
scheduler_type: "cosine"

# Loss Configuration
loss_alpha: -1
loss_gamma: 0
label_smoothing: 0
loss_reduction: "sum"

# Learning Rate Configuration
lr_encoder: 1e-5
lr_others: 5e-5
weight_decay_encoder: 0.01
weight_decay_other: 0.01
max_grad_norm: 1.0

# Data Configuration
train_data: "data.json"
prev_path: null  # Training from scratch
save_total_limit: 3

# Advanced Settings
max_types: 25
max_len: 384
```

---

## UniEncoder Token Configuration

`UniEncoderTokenConfig` is used for token-level classification, suitable for long-form entity extraction.

### Architecture-Specific Parameters

#### `span_mode`
`str`, *required*, fixed to `"token-level"`

This parameter is automatically set to `"token-level"` and cannot be changed for this architecture.

### Usage Example

```python
from gliner import GLiNERConfig, GLiNER

# Create config for UniEncoderToken
config = GLiNERConfig(
    model_name="microsoft/deberta-v3-small",
    hidden_size=512,
    span_mode="token-level",  # Automatically set for this architecture
)

model = GLiNER.from_config(config)
```

### Training Config Example

```yaml
# Model Configuration
model_name: microsoft/deberta-v3-base
labels_encoder: null
name: "token level gliner"
hidden_size: 768
dropout: 0.4
fine_tune: true
subtoken_pooling: first
span_mode: token-level  # Token-level prediction
num_rnn_layers: 1  # LSTM helps with token sequences

# Training Parameters (same as span)
num_steps: 30000
train_batch_size: 8
eval_every: 1000
warmup_ratio: 0.1
scheduler_type: "cosine"

# Loss Configuration
loss_alpha: -1
loss_gamma: 0
label_smoothing: 0
loss_reduction: "sum"

# Learning Rate Configuration
lr_encoder: 1e-5
lr_others: 5e-5
weight_decay_encoder: 0.01
weight_decay_other: 0.01
max_grad_norm: 1.0

# Data Configuration
train_data: "data.json"
prev_path: null
save_total_limit: 3

# Advanced Settings
max_types: 25
max_len: 384
```

---

## BiEncoder Span Configuration

`BiEncoderSpanConfig` uses separate encoders for text and entity labels, enabling pre-computation of label embeddings.

### Architecture-Specific Parameters

#### `labels_encoder`
`str`, *required*

Model identifier or path for the label encoder. Typically a sentence transformer model.

**Examples:**
- `"sentence-transformers/all-MiniLM-L6-v2"`
- `"BAAI/bge-small-en-v1.5"`

---

#### `labels_encoder_config`
`dict` or `PretrainedConfig`, *optional*

Nested configuration for the label encoder model.

### Important Notes

:::warning Embedding Resizing Not Supported
Unlike UniEncoder models, BiEncoder models do not support token embedding resizing. The vocabulary is fixed to the pretrained encoder's vocabulary.
:::

### Usage Example

```python
from gliner import GLiNERConfig, GLiNER

# Create config for BiEncoderSpan
config = GLiNERConfig(
    model_name="microsoft/deberta-v3-base",
    labels_encoder="sentence-transformers/all-MiniLM-L6-v2",  # Bi-encoder
    max_width=12,
    hidden_size=768,
    span_mode="markerV0",
)

model = GLiNER.from_config(config)

# Pre-compute label embeddings for efficiency
labels = ["person", "organization", "location"]
labels_embeddings = model.encode_labels(labels)

# Use pre-computed embeddings for inference
entities = model.batch_predict_with_embeds(
    texts=["Apple Inc. was founded by Steve Jobs."],
    labels_embeddings=labels_embeddings,
    labels=labels
)
```

### Training Config Example

```yaml
# Model Configuration
model_name: microsoft/deberta-v3-base
labels_encoder: sentence-transformers/all-MiniLM-L6-v2  # Bi-encoder
name: "bi-encoder span gliner"
max_width: 12
hidden_size: 768
dropout: 0.4
fine_tune: true
subtoken_pooling: first
span_mode: markerV0
post_fusion_schema: "l2t-t2l"  # Cross-attention fusion

# Training Parameters
num_steps: 30000
train_batch_size: 8
eval_every: 1000
warmup_ratio: 0.1
scheduler_type: "cosine"

# Loss Configuration (Focal loss recommended)
loss_alpha: 0.25
loss_gamma: 2.0
label_smoothing: 0
loss_reduction: "sum"

# Learning Rate Configuration
lr_encoder: 1e-5
lr_others: 5e-5
weight_decay_encoder: 0.01
weight_decay_other: 0.01
max_grad_norm: 1.0

# Data Configuration
train_data: "data.json"
prev_path: null
save_total_limit: 3

# Advanced Settings
max_types: 100  # Can handle many more types
max_len: 384
```

---

## BiEncoder Token Configuration

`BiEncoderTokenConfig` combines bi-encoder architecture with token-level prediction.

### Architecture-Specific Parameters

#### `labels_encoder`
`str`, *required*

Model identifier for the label encoder.

#### `span_mode`
`str`, *required*, fixed to `"token-level"`

Automatically set to `"token-level"` for this architecture.

### Usage Example

```python
from gliner import GLiNERConfig, GLiNER

# Create config for BiEncoderToken
config = GLiNERConfig(
    model_name="microsoft/deberta-v3-base",
    labels_encoder="sentence-transformers/all-MiniLM-L6-v2",
    hidden_size=768,
    span_mode="token-level",
)

model = GLiNER.from_config(config)
```

### Training Config Example

```yaml
# Model Configuration
model_name: microsoft/deberta-v3-base
labels_encoder: sentence-transformers/all-MiniLM-L6-v2
name: "bi-encoder token gliner"
hidden_size: 768
dropout: 0.4
fine_tune: true
subtoken_pooling: first
span_mode: token-level
num_rnn_layers: 1

# Training Parameters
num_steps: 30000
train_batch_size: 8
eval_every: 1000
warmup_ratio: 0.1
scheduler_type: "cosine"

# Loss Configuration
loss_alpha: 0.25
loss_gamma: 2.0
label_smoothing: 0
loss_reduction: "sum"

# Learning Rate Configuration
lr_encoder: 1e-5
lr_others: 5e-5
weight_decay_encoder: 0.01
weight_decay_other: 0.01
max_grad_norm: 1.0

# Data Configuration
train_data: "data.json"
prev_path: null
save_total_limit: 3

# Advanced Settings
max_types: 100
max_len: 384
```

---

## UniEncoder Span Decoder Configuration

`UniEncoderSpanDecoderConfig` extends span-based NER with a generative decoder for label generation.

### Architecture-Specific Parameters

#### `labels_decoder`
`str`, *required*

Model identifier for the generative decoder (e.g., GPT-2).

**Examples:**
- `"gpt2"`
- `"distilgpt2"`
- `"EleutherAI/gpt-neo-125M"`

---

#### `decoder_mode`
`str`, *optional*

Defines how decoder inputs are constructed.

**Choices:**
- `"prompt"` — Use entity type embeddings as decoder context
- `"span"` — Use span token representations as decoder context

---

#### `full_decoder_context`
`bool`, *optional*, defaults to `True`

Whether to provide full context to the decoder (all tokens in span) or just boundary markers.

---

#### `blank_entity_prob`
`float`, *optional*, defaults to `0.1`

Probability of using a generic "entity" label during training for improved generalization.

---

#### `labels_decoder_config`
`dict` or `PretrainedConfig`, *optional*

Nested configuration for the decoder model.

---

#### `decoder_loss_coef`
`float`, *optional*, defaults to `0.5`

Weight for the decoder generation loss in the total loss.

---

#### `span_loss_coef`
`float`, *optional*, defaults to `0.5`

Weight for the span classification loss in the total loss.

### Usage Example

```python
from gliner import GLiNERConfig, GLiNER

# Create config for UniEncoderSpanDecoder
config = GLiNERConfig(
    model_name="microsoft/deberta-v3-base",
    labels_decoder="gpt2",  # Add decoder
    decoder_mode="span",
    full_decoder_context=True,
    blank_entity_prob=0.1,
    decoder_loss_coef=0.5,
    span_loss_coef=0.5,
)

model = GLiNER.from_config(config)
```

### Training Config Example

```yaml
# Model Configuration
model_name: microsoft/deberta-v3-base
labels_decoder: gpt2  # Generative decoder
decoder_mode: span
full_decoder_context: true
blank_entity_prob: 0.1
name: "span decoder gliner"
max_width: 12
hidden_size: 768
dropout: 0.4
fine_tune: true
span_mode: markerV0

# Loss Configuration
decoder_loss_coef: 0.5
span_loss_coef: 0.5

# Training Parameters
num_steps: 30000
train_batch_size: 4  # Smaller due to decoder
eval_every: 1000
warmup_ratio: 0.1
scheduler_type: "cosine"

# Loss Configuration
loss_alpha: -1
loss_gamma: 0
label_smoothing: 0.1  # Helps with generation
loss_reduction: "sum"

# Learning Rate Configuration
lr_encoder: 1e-5
lr_others: 5e-5
weight_decay_encoder: 0.01
weight_decay_other: 0.01
max_grad_norm: 1.0

# Data Configuration
train_data: "data.json"
prev_path: null
save_total_limit: 3

# Advanced Settings
max_types: 25
max_len: 384
```

---

## UniEncoder Span Relex Configuration

`UniEncoderSpanRelexConfig` extends span-based NER with relation extraction capabilities.

### Architecture-Specific Parameters

#### `relations_layer`
`str`, *required*

Type of relation representation layer to use.

**Choices:**
- `"dot"` — Dot product between entity representations
- `"gcn"` — Graph convolutional network for modeling interactions between entities
- `"gat"` — Graph attention network for modeling interactions between entities

---

#### `triples_layer`
`str`, *optional*

Type of triple scoring layer for (head, relation, tail) scoring.

**Choices:**
- `"distmult"` — DistMult scoring function
- `"complex"` — ComplEx scoring function
- `"transe"` — TransE scoring function

---

#### `embed_rel_token`
`bool`, *optional*, defaults to `True`

Whether to embed relation type tokens similar to entity tokens.

---

#### `rel_token_index`
`int`, *optional*, defaults to `-1`

Index of the relation token in vocabulary. Set automatically during initialization.

---

#### `rel_token`
`str`, *optional*, defaults to `"<<REL>>"`

Special token used to mark relation types in the input.

---

#### `span_loss_coef`
`float`, *optional*, defaults to `1.0`

Weight for entity span classification loss.

---

#### `adjacency_loss_coef`
`float`, *optional*, defaults to `1.0`

Weight for entity pair adjacency prediction loss.

---

#### `relation_loss_coef`
`float`, *optional*, defaults to `1.0`

Weight for relation type classification loss.

### Usage Example

```python
from gliner import GLiNERConfig, GLiNER

# Create config for UniEncoderSpanRelex
config = GLiNERConfig(
    model_name="microsoft/deberta-v3-base",
    relations_layer="biaffine",  # Enable relations
    triples_layer="distmult",
    rel_token="<<REL>>",
    span_loss_coef=1.0,
    adjacency_loss_coef=1.0,
    relation_loss_coef=1.0,
)

model = GLiNER.from_config(config)
```

### Training Config Example

```yaml
# Model Configuration
model_name: microsoft/deberta-v3-base
relations_layer: biaffine  # Enable relation extraction
triples_layer: distmult
rel_token: "<<REL>>"
embed_rel_token: true
name: "span relex gliner"
max_width: 12
hidden_size: 768
dropout: 0.4
fine_tune: true
span_mode: markerV0

# Loss Configuration
span_loss_coef: 1.0
adjacency_loss_coef: 1.0
relation_loss_coef: 1.0

# Training Parameters
num_steps: 30000
train_batch_size: 6  # Smaller due to relation computation
eval_every: 1000
warmup_ratio: 0.1
scheduler_type: "cosine"

# Loss Configuration
loss_alpha: -1
loss_gamma: 0
label_smoothing: 0
loss_reduction: "sum"

# Learning Rate Configuration
lr_encoder: 1e-5
lr_others: 5e-5
weight_decay_encoder: 0.01
weight_decay_other: 0.01
max_grad_norm: 1.0

# Data Configuration
train_data: "data_with_relations.json"  # Must include relation annotations
prev_path: null
save_total_limit: 3

# Advanced Settings
max_types: 25
max_len: 384
```

### Data Format for Relation Extraction

```python
train_data = [
    {
        "tokenized_text": ["John", "works", "at", "Microsoft"],
        "ner": [[0, 0, "person"], [3, 3, "organization"]],
        "relations": [[0, 1, "works_at"]]  # (head_entity_idx, tail_entity_idx, relation_type)
    }
]
```

---

## TrainingArguments

Custom extension of `transformers.TrainingArguments` with additional parameters for GLiNER models.

### GLiNER-Specific Parameters

#### `others_lr`  
`float`, *optional*  
Learning rate for non-encoder parameters (e.g., span layers, label encoder). If not specified, uses main `learning_rate`.

---

#### `others_weight_decay`  
`float`, *optional*, defaults to `0.0`  
Weight decay for non-encoder parameters.

---

#### `focal_loss_alpha`  
`float`, *optional*, defaults to `-1`  
Alpha parameter for focal loss. If ≥ 0, focal loss is activated.

Focal loss formula:  
`FL(p_t) = -α × (1 - p_t)^γ × log(p_t)`

---

#### `focal_loss_gamma`  
`float`, *optional*, defaults to `0`  
Gamma parameter for focal loss. Higher values increase focus on hard examples.

---

#### `focal_loss_prob_margin`  
`float`, *optional*, defaults to `0.0`  
Probability margin for focal loss adjustment.

---

#### `label_smoothing`  
`float`, *optional*, defaults to `0.0`  
Label smoothing factor ε for regularization.

---

#### `loss_reduction`  
`str`, *optional*, defaults to `"sum"`  
How to aggregate loss across samples.  
**Choices:** `"sum"`, `"mean"`

---

#### `negatives`  
`float`, *optional*, defaults to `1.0`  
Ratio of negative to positive spans during training.

---

#### `masking`  
`str`, *optional*, defaults to `"none"`  
Masking strategy for negative sampling.  
**Choices:** `"none"`, `"global"`, `"label"`, `"span"`