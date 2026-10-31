# Training

GLiNER can be easily fine-tuned thanks to its architecture and carefully pre-trained models available on [Hugging Face](https://huggingface.co/models?sort=trending&search=gliner).

## Quickstart

### Installation

To install GLiNER with training dependencies:

```bash
# Install with training support
pip install gliner[training]
```

### Simple Training Example

```python
from gliner import GLiNER

# Load a pretrained model
model = GLiNER.from_pretrained("urchade/gliner_small-v2.1")

# Define your training data
train_data = [
    {
        "tokenized_text": ["Apple", "Inc.", "is", "headquartered", "in", "Cupertino"],
        "ner": [[0, 1, "organization"], [5, 5, "location"]]
    },
    {
        "tokenized_text": ["Steve", "Jobs", "founded", "Apple"],
        "ner": [[0, 1, "person"], [3, 3, "organization"]]
    }
]

# Train the model
trainer = model.train_model(
    train_dataset=train_data,
    eval_dataset=train_data,  # Use separate eval data in practice
    output_dir="./my_model",
    max_steps=1000,
    learning_rate=5e-5,
    per_device_train_batch_size=8,
)

# Save the trained model
trainer.save_model()
```

## Dataset Format

### Basic Format

GLiNER expects training data as a list of dictionaries with the following structure:

```python
{
    "tokenized_text": List[str],  # List of tokens
    "ner": List[List[Union[int, str]]]  # Entity annotations: [[start, end, label], ...]
}
```

**Important Notes:**
- `tokenized_text`: Pre-tokenized text as a list of tokens (words)
- `ner`: List of entity annotations where each annotation is `[start_idx, end_idx, entity_type]`
- Indices are **token-level** (not character-level)
- `start_idx` and `end_idx` are **inclusive** (both point to tokens in the entity span)

### Basic Example

```python
train_data = [
    {
        "tokenized_text": ["Barack", "Obama", "was", "born", "in", "Hawaii", "."],
        "ner": [
            [0, 1, "person"],      # "Barack Obama" spans tokens 0-1
            [5, 5, "location"]     # "Hawaii" is token 5
        ]
    },
    {
        "tokenized_text": ["Microsoft", "was", "founded", "in", "1975"],
        "ner": [
            [0, 0, "organization"],  # "Microsoft" is token 0
            [4, 4, "date"]           # "1975" is token 4
        ]
    }
]
```

### Advanced Format: Explicit Labels and Negatives

Examples with `"ner": []` are supported. If no candidate entity types are
available, the processor uses one empty-string label (`""`) with all-zero
targets. For uni-encoders, this produces an entity marker without label text.
Bi-encoders use the empty-string label only when the shared batch label pool is
empty. To train a negative example against specific entity types, supply
`ner_labels` or `ner_negatives` as described below.

You can improve training by explicitly defining:
1. **Positive labels** (`ner_labels`): Entity types present in this example, recomened to specify for fixed label set use cases.
2. **Negative labels** (`ner_negatives`): Entity types to use as negative examples

This gives you fine-grained control over the label sampling during training.

#### Example with Explicit Labels

```python
train_data = [
    {
        "tokenized_text": ["Apple", "Inc.", "hired", "Tim", "Cook", "in", "1998"],
        "ner": [
            [0, 1, "organization"],
            [3, 4, "person"],
            [6, 6, "date"]
        ],
        # Explicitly define which labels are relevant for this example
        "ner_labels": ["organization", "person", "date", "location"],
        
        # Explicitly define negative examples to use
        "ner_negatives": ["product", "event", "money"]
    }
]
```

**Benefits:**
- **Better control**: Explicitly specify which entity types to consider
- **Hard negatives**: Include similar entity types as negatives (e.g., "person" as negative when "organization" is positive)
- **Curriculum learning**: Start with easy negatives, gradually add harder ones
- **Domain adaptation**: Focus on specific entity types relevant to your domain

#### Example with Hard Negatives

```python
# Training example that teaches the model to distinguish between similar types
train_data = [
    {
        "tokenized_text": ["Google", "CEO", "Sundar", "Pichai", "announced", "Pixel"],
        "ner": [
            [0, 0, "organization"],
            [1, 1, "position"],
            [2, 3, "person"],
            [5, 5, "product"]
        ],
        "ner_labels": ["organization", "person", "position", "product"],
        
        # Use similar entity types as hard negatives
        "ner_negatives": [
            "company",        # Similar to "organization"
            "individual",     # Similar to "person"
            "job_title",      # Similar to "position"
            "brand"           # Similar to "product"
        ]
    }
]
```

### Relation Extraction Dataset Format

For relation extraction models (UniEncoderSpanRelex), include relation annotations:

```python
train_data = [
    {
        "tokenized_text": ["John", "Smith", "works", "at", "Microsoft", "in", "Seattle"],
        "ner": [
            [0, 1, "person"],
            [4, 4, "organization"],
            [6, 6, "location"]
        ],
        # Relations: [head_entity_idx, tail_entity_idx, relation_type]
        "relations": [
            [0, 1, "works_at"],      # person 0 works_at organization 1
            [1, 2, "located_in"]     # organization 1 located_in location 2
        ],
        # Optional: explicit relation types
        "rel_labels": ["works_at", "located_in", "founded_by"],
        "rel_negatives": ["competitor_of", "subsidiary_of"]
    }
]
```

**Relation Indices:**
- Head and tail indices refer to the position in the `ner` list (not token positions)
- Ensure indices are valid (within bounds of the `ner` list)
- Relations should be ordered and consistent with entity ordering

### Decoder-Based Models Dataset Format

For generative decoder models (UniEncoderSpanDecoder), the format is the same, but you can optionally train the decoder:

```python
train_data = [
    {
        "tokenized_text": ["Apple", "released", "iPhone", "15"],
        "ner": [
            [0, 0, "company"],
            [2, 3, "product"]
        ],
        # The model will learn to generate these labels
        "ner_labels": ["company", "product", "technology"]
    }
]
```

## Training Configuration

### Using Configuration Files

Create a YAML configuration file for reproducible training:

```yaml
# config.yaml

model:
  model_name: "microsoft/deberta-v3-base"
  span_mode: "markerV0"
  max_width: 12
  hidden_size: 768
  dropout: 0.4
  max_len: 384
  max_types: 25

training:
  # Model checkpoint
  prev_path: null  # Set to model path for fine-tuning
  
  # Training schedule
  num_steps: 10000
  train_batch_size: 8
  eval_every: 1000
  warmup_ratio: 0.1
  scheduler_type: "cosine"
  
  # Learning rates
  lr_encoder: 1e-5      # Learning rate for encoder
  lr_others: 5e-5       # Learning rate for other components
  weight_decay_encoder: 0.01
  weight_decay_other: 0.01
  max_grad_norm: 1.0
  
  # Loss configuration
  loss_alpha: -1        # Focal loss alpha (-1 disables)
  loss_gamma: 0         # Focal loss gamma (0 disables)
  loss_reduction: "sum"
  negatives: 1.0        # Negative sampling ratio
  masking: "none"       # Options: "none", "global", "label", "span"
  
  # Checkpointing
  save_total_limit: 3
  
  # Component freezing (optional)
  freeze_components: null  # e.g., ["text_encoder"]

data:
  root_dir: "models"
  train_data: "data/train.json"
  val_data_dir: "data/val.json"
```

### Training Script

Save this as `train.py`:

```python
import argparse
import json
from pathlib import Path
from gliner import GLiNER
from gliner.utils import load_config_as_namespace, namespace_to_dict

def load_json_data(path: str):
    """Load JSON dataset."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_model(model_cfg: dict, train_cfg: dict):
    """Build or load GLiNER model."""
    prev_path = train_cfg.get("prev_path")
    
    if prev_path and str(prev_path).lower() not in ("none", "null", ""):
        print(f"Loading pretrained model from: {prev_path}")
        return GLiNER.from_pretrained(prev_path)
    
    print("Initializing model from config...")
    return GLiNER.from_config(model_cfg)

def main(cfg_path: str):
    """Main training function."""
    # Load config
    cfg = load_config_as_namespace(cfg_path)
    
    # Convert to dicts for model building
    model_cfg = namespace_to_dict(cfg.model)
    train_cfg = namespace_to_dict(cfg.training)
    
    # Setup output directory
    output_dir = Path(cfg.data.root_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load datasets
    print(f"Loading training data from: {cfg.data.train_data}")
    train_dataset = load_json_data(cfg.data.train_data)
    print(f"Training samples: {len(train_dataset)}")
    
    eval_dataset = None
    if hasattr(cfg.data, "val_data_dir") and cfg.data.val_data_dir.lower() not in ("none", "null", ""):
        print(f"Loading validation data from: {cfg.data.val_data_dir}")
        eval_dataset = load_json_data(cfg.data.val_data_dir)
        print(f"Validation samples: {len(eval_dataset)}")
    
    # Build model
    model = build_model(model_cfg, train_cfg)
    print(f"Model type: {model.__class__.__name__}")
    
    # Get freeze components
    freeze_components = train_cfg.get("freeze_components", None)
    if freeze_components:
        print(f"Freezing components: {freeze_components}")
    
    # Train
    print("\nStarting training...")
    trainer = model.train_model(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        output_dir=output_dir,
        
        # Schedule
        max_steps=cfg.training.num_steps,
        lr_scheduler_type=cfg.training.scheduler_type,
        warmup_ratio=cfg.training.warmup_ratio,
        
        # Batch & optimization
        per_device_train_batch_size=cfg.training.train_batch_size,
        per_device_eval_batch_size=cfg.training.train_batch_size,
        learning_rate=float(cfg.training.lr_encoder),
        others_lr=float(cfg.training.lr_others),
        weight_decay=float(cfg.training.weight_decay_encoder),
        others_weight_decay=float(cfg.training.weight_decay_other),
        max_grad_norm=float(cfg.training.max_grad_norm),
        
        # Loss
        focal_loss_alpha=float(cfg.training.loss_alpha),
        focal_loss_gamma=float(cfg.training.loss_gamma),
        focal_loss_prob_margin=float(getattr(cfg.training, "loss_prob_margin", 0.0)),
        loss_reduction=cfg.training.loss_reduction,
        negatives=float(cfg.training.negatives),
        masking=cfg.training.masking,
        
        # Logging & saving
        save_steps=cfg.training.eval_every,
        logging_steps=cfg.training.eval_every,
        save_total_limit=cfg.training.save_total_limit,
        
        # Freezing
        freeze_components=freeze_components,
    )
    
    trainer.save_model()
    print(f"\n✓ Training complete! Model saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GLiNER model")
    parser.add_argument("--config", type=str, default="configs/config.yaml", 
                       help="Path to config file (YAML or JSON)")
    args = parser.parse_args()
    
    main(args.config)
```

Run training:

```bash
python train.py --config config.yaml
```

## Loss Functions and Calibration

Pick the loss with `loss_type` (in `create_training_args`, `train_model` kwargs or `training.loss_type` in the YAML config):

| `loss_type` | Description |
|---|---|
| `focal` (default) | Per-(span, class) BCE with optional focal weighting (`focal_loss_alpha`, `focal_loss_gamma`). |
| `bce` | Same as `focal` with `alpha=-1, gamma=0`. |
| `dice` | Self-adjusting Dice loss (`dice_gamma`); span models only. |
| `ce` | Multi-label cross-entropy against a fixed "no entity" logit at 0: every positive class must beat it, and it must beat every negative class. |
| `ce_calibrated` | `ce` plus `ce_calib_weight` × mean per-class BCE, which keeps every sigmoid probability calibrated. |

Because the "no entity" logit is fixed at 0, models trained with `ce` / `ce_calibrated` keep the usual `sigmoid(score) > threshold` decoding. These losses ignore `focal_loss_*`, `label_smoothing` and `focal_loss_prob_margin`, and support every architecture (span, token, bi-encoder, decoder, relation extraction). The relation adjacency loss always stays BCE.

```python
model.train_model(
    train_dataset=train_data,
    eval_dataset=None,
    output_dir="models",
    loss_type="ce_calibrated",
    ce_calib_weight=0.5,
)
```

### Post-hoc calibration of a trained model

`scripts/calibrate.py` freezes a trained model and fits a temperature `T` (and, with `--fit-bias`, a bias `b`) on a held-out dataset in GLiNER format, so that `sigmoid(logit / T + b)` matches how often predictions are correct:

```bash
python scripts/calibrate.py --model models/checkpoint-1000 --data dev.json --output-dir models/calibrated
```

The weights are untouched. The values are stored in the config as `logit_temperature` / `logit_bias` (and `rel_logit_temperature` / `rel_logit_bias` for relations, which can be set by hand), and every decoder applies them. With the temperature alone, predictions at threshold 0.5 stay the same; scores and the meaning of other thresholds change. A bias also moves the 0.5 boundary. The script reports NLL, Brier score, ECE and entry-level precision/recall/F1 before and after, and writes `calibration_report.json` next to the model.

### Calibration head

`--method head` fits a small calibration head, adapted from GLiClass, with the backbone frozen. For every *candidate* (span, label) pair (raw probability at least `--candidate-threshold`, default 0.01), an MLP looks at:

- the span's boundary and mean word embeddings,
- the label embedding,
- score statistics: the logit, its gap to the best and mean class, span width, and the number of labels.

It predicts an inverse temperature β and a bias b, and the calibrated score is `sigmoid(β · logit + b)`. Pairs below the threshold keep their raw score. For token-level models, the pair is (word, label), and β and b scale its start / end / inside logits together.

```bash
python scripts/calibrate.py --model models/checkpoint-1000 --data dev.json --eval-data test.json \
    --method head --output-dir models/calibrated
```

How the head is fitted:

- It starts from a global Platt fit on the candidates and is pulled back toward it by a prior (`--prior-weight`, `--bias-prior-weight`).
- It trains on 80% of `--data` and early-stops on the remaining 20% (`--eval-ratio`).
- `--balance` (off by default) reweights candidates by raw confidence bin and correctness, as in GLiClass. On NER data it made held-out NLL worse.

The script reports, for the raw model, global scaling and the head, which share of decoded spans in each score bin are correct, plus the calibration error (ECE).

How the head is used:

- It is saved with the model (`use_calibrator` and `calibrator_*` in the config, weights under `calibrator.*`) and applied on every PyTorch inference path.
- Raw logits stay available as `uncalibrated_logits` on the model output.
- `model.calibrator_disabled()` switches it off temporarily; `model.add_calibrator()` and `model.remove_calibrator()` attach and detach it.
- It is not applied during training (when labels are passed), with the ONNX / OpenVINO runtimes, or for streaming models.

## Best Practices

1. **Start with a pretrained model**: Fine-tuning is almost always better than training from scratch
2. **Use validation data**: Monitor overfitting with a separate validation set
3. **Experiment with negative sampling**: Try different `negatives` and `masking` strategies
4. **Use focal loss for imbalanced data**: Set `focal_loss_alpha` and `focal_loss_gamma`
5. **Freeze components for quick adaptation**: Freeze encoder when fine-tuning on small datasets
6. **Include hard negatives**: Explicitly define similar entity types as negatives
7. **Save multiple checkpoints**: Set `save_total_limit` > 1 to keep best models
8. **Monitor training**: Use TensorBoard or W&B to track loss and metrics
9. **Start small**: Test your pipeline on a subset before full training
10. **Validate data format**: Ensure indices are correct and within bounds
