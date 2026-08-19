# Joint NER + Relation Extraction

## Overview

`UniEncoderSpanRelexGLiNER` extends GLiNER's span-based NER architecture with a
relation extraction head, performing both tasks in a single forward pass. This
enables knowledge-graph construction, document understanding, and structured
information extraction without requiring two separate models.

Reference: [GLiNER-Relex (arXiv:2605.10108)](https://arxiv.org/abs/2605.10108)

## Architecture

```
Text → DeBERTa encoder → Span representations
                              │
                              ├── NER head   → entity predictions
                              │
                              └── RE head    → entity-pair adjacency
                                               → relation type classification
```

The RE head scores all entity-pair combinations using the span representations
from the NER head. No separate encoding pass is needed.

## Inference

### `predict_entities()` — entities only

```python
from gliner.model import UniEncoderSpanRelexGLiNER

model = UniEncoderSpanRelexGLiNER.from_pretrained("knowledgator/gliner-relex-large-v1.0")

entities = model.predict_entities(
    "Apple Inc. was founded by Steve Jobs in Cupertino, California.",
    labels=["organization", "person", "location"],
)
# → [{"text": "Apple Inc.", "label": "organization", ...}, ...]
```

### `predict_relations()` — entities + relations

```python
entities, relations = model.predict_relations(
    "Apple Inc. was founded by Steve Jobs in Cupertino, California.",
    labels=["organization", "person", "location"],
    relations=["founded_by", "headquartered_in", "born_in"],
    threshold=0.5,
)

for rel in relations:
    print(f"{rel['head']['text']} --[{rel['relation']}]--> {rel['tail']['text']}  (score={rel['score']:.2f})")
# → Apple Inc. --[founded_by]--> Steve Jobs  (score=0.87)
# → Apple Inc. --[headquartered_in]--> Cupertino  (score=0.74)
```

### API reference

```python
entities, relations = model.predict_relations(
    text:                str,
    labels:              List[str],          # entity type labels
    relations:           List[str],          # relation type labels
    threshold:           float = 0.5,        # entity confidence threshold
    adjacency_threshold: float = None,       # entity-pair adjacency threshold (defaults to threshold)
    relation_threshold:  float = None,       # relation type threshold (defaults to threshold)
    flat_ner:            bool = True,
    multi_label:         bool = False,
)
# Returns:
#   entities: List[Dict] — same format as predict_entities()
#   relations: List[Dict] — {"head": entity_dict, "relation": str, "tail": entity_dict, "score": float}
```

> **Description conditioning (Feature 2):** Entity labels and relation types accept full
> natural-language descriptions via the same API as `predict_entities()`.
> ```python
> entities, relations = model.predict_relations(
>     text,
>     labels={"organization": "a legally incorporated company or institution", ...},
>     relations=["founded_by", "headquartered_in"],
> )
> ```

### Long-document inference

Use `predict_entities_long()` for the entity extraction step, then pass found entities
directly to the model for relation scoring:

```python
from gliner.long_doc import predict_entities_long

# Step 1: extract entities with sliding window
entities = predict_entities_long(model, long_text, labels, max_tokens=512, stride=128)

# Step 2: run relation scoring on the full entity set (no sliding window needed for RE)
_, relations = model.predict_relations(long_text, labels, relation_types, threshold=0.5)
```

## Training

Use `scripts/train_relex.py` to train on any dataset with the joint NER+RE format:

```bash
python scripts/train_relex.py \
    --model_id microsoft/deberta-v3-small \
    --train_data data/conll04_train.json \
    --eval_data  data/conll04_dev.json \
    --output_dir results/relex_conll04 \
    --max_steps 5000 \
    --batch_size 8
```

### Training data format

JSON or JSON Lines file. Each example must have `tokenized_text`, `ner`, and `relations` keys:

```json
{
  "tokenized_text": ["Apple", "was", "founded", "by", "Steve", "Jobs"],
  "ner": [
    [0, 0, "organization"],
    [4, 5, "person"]
  ],
  "relations": [
    [4, 5, "person", 0, 0, "organization", "founded_by"]
  ]
}
```

Relation tuple format: `[head_start, head_end, head_type, tail_start, tail_end, tail_type, relation_type]`

### Training with hard negatives + contrastive loss

```bash
python scripts/train_relex.py \
    --model_id microsoft/deberta-v3-small \
    --train_data data/conll04_train.json \
    --output_dir results/relex_conll04_enhanced \
    --hard_negative_ratio 0.5 \
    --contrastive_coef 0.1 \
    --use_curriculum
```

## Supported datasets

| Dataset | Entity types | Relation types | Size |
|---|---|---|---|
| CoNLL04 | 4 (PER, ORG, LOC, OTH) | 5 (Work_For, Kill, OrgBased_In, Live_In, Located_In) | 1,137 train |
| FewRel | 80 relation types | — | 56,000 train |
| DocRED | 6 entity types | 96 relation types | 3,053 train |
| Re-TACRED | 4 entity types | 40 relation types | 68,124 train |

## Recommended pretrained models

| Model | Description |
|---|---|
| `knowledgator/gliner-relex-large-v1.0` | Large, high accuracy |
| `knowledgator/gliner-relex-small-v1.0` | Small, fast |

## Troubleshooting

**No relations predicted:**
- Lower `relation_threshold` (try 0.3)
- Ensure both entity types in the relation appear in `labels`
- Verify the model was trained with relation annotations (not NER-only)

**Entity predictions differ from `predict_entities()`:**
- This is expected: the RE head sees entity context that can shift span scores slightly
- Use `return_relations=False` in `inference()` to get NER-only output without RE overhead
