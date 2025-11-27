# Advanced Usage

## 🚀 Basic Use Case

After installing the GLiNER library, import the `GLiNER` class. You can load your chosen model with `GLiNER.from_pretrained` and use `inference` to identify entities within your text.

```python
from gliner import GLiNER

# Load a GLiNER model
model = GLiNER.from_pretrained("urchade/gliner_small-v2.1")

# Sample text for entity prediction
text = """
Cristiano Ronaldo dos Santos Aveiro (Portuguese pronunciation: [kɾiʃˈtjɐnu ʁɔˈnaldu]; 
born 5 February 1985) is a Portuguese professional footballer who plays as a forward for 
and captains both Saudi Pro League club Al Nassr and the Portugal national team. Widely 
regarded as one of the greatest players of all time, Ronaldo has won five Ballon d'Or 
awards, a record three UEFA Men's Player of the Year Awards, and four European Golden 
Shoes, the most by a European player.
"""

# Define labels for entity extraction
labels = ["person", "award", "date", "teams", "competition"]

# Perform entity prediction
entities = model.predict_entities(text, labels, threshold=0.5)

# Display predicted entities and their labels
for entity in entities:
    print(entity["text"], "=>", entity["label"])
```

<details>
<summary>Expected Output</summary>

```bash
Cristiano Ronaldo dos Santos Aveiro => person
5 February 1985 => date
Al Nassr => teams
Portugal national team => teams
Ballon d'Or => award
UEFA Men's Player of the Year Awards => award
European Golden Shoes => award
```
</details>

### Understanding the Output

Each predicted entity is a dictionary with the following structure:

```python
{
    'start': int,      # Start character position in text
    'end': int,        # End character position in text
    'text': str,       # Extracted text span
    'label': str,      # Predicted entity type
    'score': float     # Confidence score (0-1)
}
```

Example:
```python
for entity in entities:
    print(f"Text: {entity['text']}")
    print(f"Label: {entity['label']}")
    print(f"Score: {entity['score']:.3f}")
    print(f"Position: [{entity['start']}:{entity['end']}]")
    print("---")
```

## Batch Processing

For processing multiple texts efficiently, use the `inference` method:

```python
from gliner import GLiNER

model = GLiNER.from_pretrained("urchade/gliner_small-v2.1")

# Multiple texts to process
texts = [
    "Apple Inc. was founded by Steve Jobs in Cupertino, California.",
    "Google LLC is headquartered in Mountain View.",
    "Amazon was started by Jeff Bezos in Seattle."
]

labels = ["organization", "person", "location"]

# Process all texts at once
all_entities = model.inference(texts, labels, batch_size=3, threshold=0.5)

# Display results for each text
for i, entities in enumerate(all_entities):
    print(f"\nText {i+1}: {texts[i]}")
    print("Entities:")
    for entity in entities:
        print(f"  - {entity['text']} ({entity['label']}): {entity['score']:.2f}")
```

**Benefits of Batch Processing:**
- **Faster**: Process multiple texts in parallel
- **Efficient**: Better GPU utilization
- **Scalable**: Handle large document collections

## Using Different Model Architectures

GLiNER supports multiple architecture variants, each optimized for different scenarios.

### UniEncoder Models (Standard)

Best for general-purpose NER with up to ~30 entity types:

```python
from gliner import GLiNER

# Load a standard UniEncoder model
model = GLiNER.from_pretrained("urchade/gliner_small-v2.1")

text = "Apple Inc. was founded by Steve Jobs in 1976."
labels = ["company", "person", "date"]

entities = model.predict_entities(text, labels)
for entity in entities:
    print(f"{entity['text']} => {entity['label']}")
```

### BiEncoder Models (Scalable)

Best for handling many entity types (50-200+) with pre-computed label embeddings:

```python
from gliner import GLiNER

# Load a BiEncoder model
model = GLiNER.from_pretrained("knowledgator/gliner-bi-small-v1.0")

# BiEncoders handle many entity types efficiently
labels = [
    "person", "organization", "location", "date", "product", "event",
    "technology", "software", "hardware", "programming_language",
    "framework", "library", "database", "protocol", "standard",
    # ... can handle 100+ types efficiently
]

text = "Python is a programming language created by Guido van Rossum."
entities = model.predict_entities(text, labels)

# For production: pre-compute label embeddings
label_embeddings = model.encode_labels(labels, batch_size=16)

# Then use cached embeddings for faster inference
entities = model.predict_with_embeds(
    text, 
    label_embeddings, 
    labels,
    threshold=0.5
)
```

**BiEncoder Advantages:**
- Handle 100+ entity types without performance degradation
- Pre-compute label embeddings once, reuse across documents
- Faster inference when processing many documents with same entity types

### Token-Level Models

Best for extracting long entity spans (multi-sentence entities, summaries):

```python
from gliner import GLiNER

# Load a token-level model
model = GLiNER.from_pretrained("knowledgator/gliner-multitask-large-v0.5")

# Token-level models excel at long entities
text = """
The European Union is a political and economic union of 27 member states 
that are located primarily in Europe. The EU has developed an internal 
single market through a standardised system of laws.
"""

labels = ["organization", "number", "location", "concept"]

entities = model.predict_entities(text, labels)
for entity in entities:
    print(f"{entity['text'][:50]}... => {entity['label']}")
```

### Relation Extraction Models

Extract both entities and relationships between them:

```python
from gliner import GLiNER

# Load a relation extraction model
model = GLiNER.from_pretrained("knowledgator/gliner-relex-large-v0.5")

text = "Bill Gates founded Microsoft in 1975. The company is headquartered in Redmond."

# Define entity types and relation types
entity_labels = ["person", "organization", "date", "location"]
relation_labels = ["founded", "founded_in", "headquartered_in"]

# Extract entities and relations
entities, relations = model.inference(
    [text],
    labels=entity_labels,
    relations=relation_labels,
    threshold=0.5,
    relation_threshold=0.5
)

# Display entities
print("Entities:")
for entity in entities[0]:
    print(f"  {entity['text']} ({entity['label']})")

# Display relations
print("\nRelations:")
for relation in relations[0]:
    head = entities[0][relation['head']['entity_idx']]
    tail = entities[0][relation['tail']['entity_idx']]
    print(f"  {head['text']} --[{relation['relation']}]--> {tail['text']}")
```

<details>
<summary>Expected Output</summary>

```bash
Entities:
  Bill Gates (person)
  Microsoft (organization)
  1975 (date)
  Redmond (location)

Relations:
  Bill Gates --[founded]--> Microsoft
  Microsoft --[founded_in]--> 1975
  Microsoft --[headquartered_in]--> Redmond
```
</details>

## Advanced Configuration

### Adjusting the Threshold

Control the precision-recall tradeoff:

```python
from gliner import GLiNER

model = GLiNER.from_pretrained("urchade/gliner_small-v2.1")
text = "Apple Inc. is a technology company."
labels = ["company", "industry"]

# High threshold: Higher precision, lower recall
entities_high = model.predict_entities(text, labels, threshold=0.7)
print(f"High threshold (0.7): {len(entities_high)} entities")

# Low threshold: Lower precision, higher recall
entities_low = model.predict_entities(text, labels, threshold=0.3)
print(f"Low threshold (0.3): {len(entities_low)} entities")

# Default threshold
entities_default = model.predict_entities(text, labels)  # threshold=0.5
print(f"Default threshold (0.5): {len(entities_default)} entities")
```

Relation extraction model also has two additional threshold parameters:
- adjacency_threshold: Confidence threshold for adjacency matrix reconstruction (defaults to threshold).
- relation_threshold: Confidence threshold for relations (defaults to threshold).

```python
from gliner import GLiNER

model = GLiNER.from_pretrained("urchade/gliner_small-v2.1")
text = "Apple Inc. is a technology company founded in 1976."
labels = ["company", "industry", "date"]
relations = ["founded in"]

results = model.predict_entities(text, labels, relations=relations, threshold=0.3, adjacency_threshold=0.25, relation_threshold=0.7)
```
Use a lower adjacency threshold so the model can rerank and classify more pairs of entities that may be linked. Set a higher relations threshold for more specificity and better precision. Feel free to adapt all three thresholds based on your use case.### Flat vs Nested NER

Control whether entities can overlap:

```python
from gliner import GLiNER

model = GLiNER.from_pretrained("urchade/gliner_small-v2.1")
text = "The University of California, Berkeley is located in California."
labels = ["university", "location"]

# Flat NER: No overlapping entities (default)
entities_flat = model.predict_entities(text, labels, flat_ner=True)
print("Flat NER:", [e['text'] for e in entities_flat])
# Output: ['University of California, Berkeley', 'California']

# Nested NER: Allow overlapping entities
entities_nested = model.predict_entities(text, labels, flat_ner=False)
print("Nested NER:", [e['text'] for e in entities_nested])
# Output: ['University of California, Berkeley', 'California, Berkeley', 'California']
```

### Multi-label Classification

Allow entities to have multiple types:

```python
from gliner import GLiNER

model = GLiNER.from_pretrained("urchade/gliner_small-v2.1")
text = "Dr. Smith is a cardiologist at Mayo Clinic."
labels = ["person", "doctor", "specialist", "professional", "organization", "hospital"]

# Single label per entity (default)
entities_single = model.predict_entities(text, labels, multi_label=False)
print("Single label:")
for e in entities_single:
    print(f"  {e['text']}: {e['label']}")

# Multiple labels per entity
entities_multi = model.predict_entities(text, labels, multi_label=True)
print("\nMulti-label:")
for e in entities_multi:
    print(f"  {e['text']}: {e['label']}")
```

## Local Models and Caching

### Loading from Local Directory

```python
from gliner import GLiNER

# Load from local directory
model = GLiNER.from_pretrained("/path/to/local/model")

# Or load from HuggingFace Hub with local cache
model = GLiNER.from_pretrained(
    "urchade/gliner_small-v2.1",
    cache_dir="./model_cache"  # Cache models locally
)
```

### Device Selection

```python
from gliner import GLiNER

# Load on GPU
model = GLiNER.from_pretrained(
    "urchade/gliner_small-v2.1",
    map_location="cuda"  # Use GPU
)

# Load on CPU
model = GLiNER.from_pretrained(
    "urchade/gliner_small-v2.1",
    map_location="cpu"
)

# Check device
print(f"Model is on: {model.device}")
```

### Model Compilation (PyTorch 2.0+)

Compile models for faster inference:

```python
from gliner import GLiNER

# Load and compile (requires PyTorch 2.0+ and CUDA)
model = GLiNER.from_pretrained(
    "urchade/gliner_small-v2.1",
    compile_torch_model=True
)

# Expect ~2x speedup on compatible hardware
```


### ⚡ Accelerating Inference with Sequence Packing

Sequence packing allows GLiNER to combine multiple short requests into a single transformer pass while keeping a block-diagonal attention mask. This drastically reduces the number of padding tokens the encoder needs to process and yields higher throughput.

1. **Configure packing once for all predictions**

   ```python
   from gliner import GLiNER, InferencePackingConfig

   model = GLiNER.from_pretrained("urchade/gliner_medium-v2.1", map_location="cuda")

   packing_cfg = InferencePackingConfig(
       max_length=512,
       sep_token_id=model.data_processor.transformer_tokenizer.sep_token_id,
       streams_per_batch=1,
   )

   # Enable packing for every subsequent `run`/`predict_*` call.
   model.configure_inference_packing(packing_cfg)

   texts = ["Email CEO to approve budget", "Schedule yearly medical checkup"]
   labels = ["person", "organization", "action"]

   predictions = model.inference(texts, labels, batch_size=16)
   ```

   You can override or disable the default configuration on a per-call basis by passing `packing_config=<new_cfg>` or `packing_config=None` respectively when invoking `model.inference` or `model.predict_entities`.

2. **Benchmark the impact**

   The `bench/bench_gliner_e2e.py` script can stress the full GLiNER pipeline in addition to encoder-only Hugging Face models:

   ```bash
   python bench/bench_gliner_e2e.py
   ```

   To isolate and measure the impact on the encoder:
   ```bash
   python bench/bench_infer_packing.py --batch_size 32 --scenario short_zipf
   ```

### 🔌 Usage with spaCy

GLiNER can be seamlessly integrated with spaCy. To begin, install the `gliner-spacy` library via pip:

```bash
pip install gliner-spacy
```

Following installation, you can add GLiNER to a spaCy NLP pipeline. Here's how to integrate it with a blank English pipeline; however, it's compatible with any spaCy model.

```python
import spacy
from gliner_spacy.pipeline import GlinerSpacy

# Configuration for GLiNER integration
custom_spacy_config = {
    "gliner_model": "urchade/gliner_mediumv2.1",
    "chunk_size": 250,
    "labels": ["person", "organization", "email"],
    "style": "ent",
    "threshold": 0.3,
    "map_location": "cpu" # only available in v.0.0.7
}

# Initialize a blank English spaCy pipeline and add GLiNER
nlp = spacy.blank("en")
nlp.add_pipe("gliner_spacy", config=custom_spacy_config)

# Example text for entity detection
text = "This is a text about Bill Gates and Microsoft."

# Process the text with the pipeline
doc = nlp(text)

# Output detected entities
for ent in doc.ents:
    print(ent.text, ent.label_, ent._.score) # ent._.score only available in v. 0.0.7
```

#### Expected Output

```
Bill Gates => person
Microsoft => organization
```


## 🏃‍♀️ Using FlashDeBERTa

Most GLiNER models use the DeBERTa encoder as their backbone. This architecture offers strong token classification performance and typically requires less data to achieve good results. However, a major drawback has been its slower inference speed, and until recently, there was no flash attention implementation compatible with DeBERTa's disentangled attention mechanism.

To address this, [FlashDeBERTa](https://github.com/Knowledgator/FlashDeBERTa) was introduced.

### Installation

```bash
pip install flashdeberta -U
```

:::tip
Before using FlashDeBERTa, please make sure that you have `transformers>=4.47.0`.
:::

### Usage

GLiNER will automatically detect and use FlashDeBERTa. If needed, you can switch to the standard `eager` attention mechanism:

```python
from gliner import GLiNER

# Use FlashDeBERTa (automatic if installed)
model = GLiNER.from_pretrained("urchade/gliner_medium-v2.1")

# Or explicitly use eager attention
model = GLiNER.from_pretrained(
    "urchade/gliner_medium-v2.1",
    _attn_implementation="eager"
)
```

**Performance Boost**: FlashDeBERTa provides up to a **3× speed boost** for typical sequence lengths—and even greater improvements for longer sequences.

## 🛠️ High-Level Pipelines {#pipelines}

GLiNER-Multitask models are designed to extract relevant information from plain text based on user-provided custom prompts. These encoder-based multitask models enable efficient and controllable information extraction with a single model, reducing computational and storage costs.

**Supported Tasks:**
- **Named Entity Recognition (NER)**: Identify and categorize entities
- **Relation Extraction**: Detect relationships between entities
- **Summarization**: Extract key sentences
- **Sentiment Extraction**: Identify sentiment-bearing text spans
- **Key-Phrase Extraction**: Extract important phrases and keywords
- **Question-Answering**: Find answers to questions in text
- **Open Information Extraction**: Extract information based on open prompts
- **Text Classification**: Classify text against predefined labels

### Classification

The `GLiNERClassifier` pipeline performs text classification tasks:

```python
from gliner import GLiNER
from gliner.multitask import GLiNERClassifier

# Initialize
model = GLiNER.from_pretrained("knowledgator/gliner-multitask-large-v0.5")
classifier = GLiNERClassifier(model=model)

# Single-label classification
text = "SpaceX successfully launched a new rocket into orbit."
labels = ['science', 'technology', 'business', 'sports']

predictions = classifier(text, classes=labels, multi_label=False)
print(predictions)
# Output: [[{'label': 'technology', 'score': 0.84}]]

# Multi-label classification
predictions_multi = classifier(text, classes=labels, multi_label=True)
print(predictions_multi)
# Output: [[{'label': 'technology', 'score': 0.84}, {'label': 'science', 'score': 0.72}]]
```

**Evaluation on Dataset:**

```python
# Evaluate on HuggingFace dataset
metrics = classifier.evaluate('dair-ai/emotion')
print(metrics)
# Output: {'micro': 0.4465, 'macro': 0.4243, 'weighted': 0.4884}
```

### Question-Answering

The `GLiNERQuestionAnswerer` pipeline extracts answers from text:

```python
from gliner import GLiNER
from gliner.multitask import GLiNERQuestionAnswerer

# Initialize
model = GLiNER.from_pretrained("knowledgator/gliner-multitask-large-v0.5")
answerer = GLiNERQuestionAnswerer(model=model)

# Extract answer
text = "SpaceX was founded by Elon Musk in 2002 to reduce space transportation costs."
question = "Who founded SpaceX?"

predictions = answerer(text, questions=question)
print(predictions)
# Output: [[{'answer': 'Elon Musk', 'score': 0.998}]]

# Multiple questions
questions = ["Who founded SpaceX?", "When was SpaceX founded?", "What is SpaceX's goal?"]
predictions = answerer(text, questions=questions)
for q, pred in zip(questions, predictions):
    print(f"Q: {q}")
    print(f"A: {pred[0]['answer']} (score: {pred[0]['score']:.3f})")
```

**Evaluation on SQuAD:**

```python
from gliner.multitask import GLiNERSquadEvaluator

evaluator = GLiNERSquadEvaluator(model_id="knowledgator/gliner-multitask-large-v0.5")
metrics = evaluator.evaluate(threshold=0.25)
print(metrics)
# Output: {'exact': 29.41, 'f1': 29.80, 'total': 11873, ...}
```

### Relation Extraction

The `GLiNERRelationExtractor` pipeline extracts relationships between entities:

```python
from gliner import GLiNER
from gliner.multitask import GLiNERRelationExtractor

# Initialize
model = GLiNER.from_pretrained("knowledgator/gliner-multitask-large-v0.5")
relation_extractor = GLiNERRelationExtractor(model=model)

# Extract relations
text = "Elon Musk founded SpaceX in 2002 to reduce space transportation costs."
entities = ['person', 'company', 'year', 'goal']
relations = ['founded', 'founded_in', 'goal']

predictions = relation_extractor(
    text, 
    entities=entities, 
    relations=relations,
    threshold=0.5
)

for pred in predictions[0]:
    print(f"{pred['source']} --[{pred['relation']}]--> {pred['target']}")
    print(f"  Score: {pred['score']:.3f}")
```

<details>
<summary>Expected Output</summary>

```bash
Elon Musk --[founded]--> SpaceX
  Score: 0.958
SpaceX --[founded_in]--> 2002
  Score: 0.912
```
</details>

### Open Information Extraction

The `GLiNEROpenExtractor` pipeline extracts information based on custom prompts:

```python
from gliner import GLiNER
from gliner.multitask import GLiNEROpenExtractor

# Initialize with custom prompt
model = GLiNER.from_pretrained("knowledgator/gliner-multitask-large-v0.5")
extractor = GLiNEROpenExtractor(
    model=model,
    prompt="Extract all companies related to space technologies"
)

# Extract information
text = """
Elon Musk founded SpaceX in 2002 to reduce space transportation costs. 
Also Elon is founder of Tesla, NeuroLink and many other companies.
"""

labels = ['company']
predictions = extractor(text, labels=labels, threshold=0.5)

for pred in predictions[0]:
    print(f"{pred['text']} (score: {pred['score']:.3f})")
```

<details>
<summary>Expected Output</summary>

```bash
SpaceX (score: 0.962)
Tesla (score: 0.936)
NeuroLink (score: 0.912)
```
</details>

**Custom Prompts for Different Tasks:**

```python
# Extract product descriptions
extractor = GLiNEROpenExtractor(
    model=model,
    prompt="Extract product descriptions and features from the text"
)

# Extract technical specifications
extractor = GLiNEROpenExtractor(
    model=model,
    prompt="Extract technical specifications and requirements"
)

# Extract contact information
extractor = GLiNEROpenExtractor(
    model=model,
    prompt="Extract all contact information including emails and phone numbers"
)
```

### Summarization

The `GLiNERSummarizer` pipeline extracts key sentences for summarization:

```python
from gliner import GLiNER
from gliner.multitask import GLiNERSummarizer

# Initialize
model = GLiNER.from_pretrained("knowledgator/gliner-multitask-large-v0.5")
summarizer = GLiNERSummarizer(model=model)

# Extract summary
text = """
Microsoft was founded by Bill Gates and Paul Allen on April 4, 1975 to develop 
and sell BASIC interpreters for the Altair 8800. During his career at Microsoft, 
Gates held the positions of chairman, chief executive officer, president and chief 
software architect, while also being the largest individual shareholder until May 2014.
"""

summary = summarizer(text, threshold=0.1)
print(summary)
```

<details>
<summary>Expected Output</summary>

```bash
['Microsoft was founded by Bill Gates and Paul Allen on April 4, 1975 to develop 
and sell BASIC interpreters for the Altair 8800.']
```
</details>

**Controlling Summary Length:**

```python
# More selective (higher threshold = shorter summary)
summary_short = summarizer(text, threshold=0.5)

# More comprehensive (lower threshold = longer summary)
summary_long = summarizer(text, threshold=0.05)
```

## Advanced Relation Extraction with UTCA

For more nuanced control over relation extraction, use the [utca](https://github.com/Knowledgator/utca) framework:

### Installation

```bash
pip install utca -U
```

### Setting Up the Pipeline

```python
from utca.core import RenameAttribute
from utca.implementation.predictors import GLiNERPredictor, GLiNERPredictorConfig
from utca.implementation.tasks import (
    GLiNER,
    GLiNERPreprocessor,
    GLiNERRelationExtraction,
    GLiNERRelationExtractionPreprocessor,
)

# Initialize predictor
predictor = GLiNERPredictor(
    GLiNERPredictorConfig(
        model_name="knowledgator/gliner-multitask-large-v0.5",
        device="cuda:0",  # Use "cpu" for CPU inference
    )
)

# Create pipeline
pipe = (
    GLiNER(  # Extract entities
        predictor=predictor,
        preprocess=GLiNERPreprocessor(threshold=0.7)
    )
    | RenameAttribute("output", "entities")  # Prepare for relation extraction
    | GLiNERRelationExtraction(  # Extract relations
        predictor=predictor,
        preprocess=(
            GLiNERPreprocessor(threshold=0.5)
            | GLiNERRelationExtractionPreprocessor()
        )
    )
)
```

### Running the Pipeline

```python
text = """
Microsoft was founded by Bill Gates and Paul Allen on April 4, 1975 to develop 
and sell BASIC interpreters for the Altair 8800. During his career at Microsoft, 
Gates held the positions of chairman, chief executive officer, president and chief 
software architect, while also being the largest individual shareholder until May 2014.
"""

result = pipe.run({
    "text": text,
    "labels": ["organization", "person", "position", "date"],
    "relations": [
        {
            "relation": "founder",
            "pairs_filter": [("organization", "person")],  # Only consider org-person pairs
            "distance_threshold": 100,  # Max distance between entities (in characters)
        },
        {
            "relation": "inception_date",
            "pairs_filter": [("organization", "date")],
        },
        {
            "relation": "held_position",
            "pairs_filter": [("person", "position")],
        }
    ]
})

# Display results
for relation in result["output"]:
    source = relation['source']['span']
    target = relation['target']['span']
    rel_type = relation['relation']
    score = relation['score']
    print(f"{source} --[{rel_type}]--> {target} (score: {score:.3f})")
```

<details>
<summary>Expected Output</summary>

```bash
Microsoft --[founder]--> Bill Gates (score: 0.968)
Microsoft --[founder]--> Paul Allen (score: 0.863)
Microsoft --[inception_date]--> April 4, 1975 (score: 0.997)
Bill Gates --[held_position]--> chairman (score: 0.966)
Bill Gates --[held_position]--> chief executive officer (score: 0.947)
Bill Gates --[held_position]--> president (score: 0.973)
Bill Gates --[held_position]--> chief software architect (score: 0.950)
```
</details>

### Advanced UTCA Features

**Distance Filtering:**

```python
# Only extract relations where entities are close together
relations = [
    {
        "relation": "works_for",
        "pairs_filter": [("person", "organization")],
        "distance_threshold": 50,  # Entities must be within 50 characters
    }
]
```

**Multiple Relation Types:**

```python
# Define complex relation schemas
relations = [
    {
        "relation": "employed_by",
        "pairs_filter": [("person", "organization")],
    },
    {
        "relation": "located_in",
        "pairs_filter": [("organization", "location"), ("person", "location")],
    },
    {
        "relation": "acquired_by",
        "pairs_filter": [("organization", "organization")],
    },
]
```

## Practical Examples

### Example 1: Extract Company Information

```python
from gliner import GLiNER

model = GLiNER.from_pretrained("urchade/gliner_small-v2.1")

text = """
Apple Inc. is headquartered in Cupertino, California. The company was founded 
by Steve Jobs, Steve Wozniak, and Ronald Wayne in April 1976. Tim Cook is the 
current CEO. Apple's main products include iPhone, iPad, and Mac computers.
"""

labels = ["company", "location", "person", "position", "product", "date"]
entities = model.predict_entities(text, labels, threshold=0.5)

# Organize by type
from collections import defaultdict
by_type = defaultdict(list)
for entity in entities:
    by_type[entity['label']].append(entity['text'])

for label, items in by_type.items():
    print(f"{label}: {', '.join(set(items))}")
```

### Example 2: Process Scientific Papers

```python
from gliner import GLiNER

model = GLiNER.from_pretrained("urchade/gliner_medium-v2.1")

abstract = """
We introduce GPT-4, a large-scale multimodal model developed by OpenAI. 
The model was trained on a diverse dataset and exhibits strong performance 
on various benchmarks including MMLU, HumanEval, and GSM-8K.
"""

labels = [
    "model_name", "organization", "dataset", "benchmark", 
    "metric", "task", "method"
]

entities = model.predict_entities(abstract, labels, threshold=0.4)

print("Extracted Information:")
for entity in entities:
    print(f"  {entity['label']}: {entity['text']}")
```

### Example 3: Analyze News Articles

```python
from gliner import GLiNER

model = GLiNER.from_pretrained("knowledgator/gliner-bi-small-v1.0")

article = """
Tesla CEO Elon Musk announced on Twitter that the company will open a new 
Gigafactory in Austin, Texas. The facility will produce the Cybertruck and 
Model Y vehicles. Construction began in July 2020 and operations started in 2021.
"""

labels = [
    "person", "position", "company", "location", "facility", 
    "product", "date", "event"
]

# Process with BiEncoder for efficiency
entities = model.predict_entities(article, labels, threshold=0.5)

# Group related entities
print("Key Information:")
print(f"- Company: {[e['text'] for e in entities if e['label'] == 'company']}")
print(f"- Location: {[e['text'] for e in entities if e['label'] == 'location']}")
print(f"- Products: {[e['text'] for e in entities if e['label'] == 'product']}")
print(f"- Timeline: {[e['text'] for e in entities if e['label'] == 'date']}")
```

## Tips and Best Practices

1. **Choose the right model architecture**:
   - UniEncoder: General purpose, < 30 entity types
   - BiEncoder: Many entity types (50-200+)
   - Token-level: Long entity spans
   - Relation extraction: Knowledge graph construction

2. **Optimize threshold for your use case**:
   - High precision: threshold = 0.6-0.8
   - Balanced: threshold = 0.4-0.6
   - High recall: threshold = 0.2-0.4

3. **Use batch processing for multiple documents**:
   - More efficient GPU utilization
   - Faster overall processing

4. **Pre-compute label embeddings (BiEncoder)**:
   - Cache embeddings when processing many documents
   - Significant speedup for production use

5. **Enable FlashDeBERTa**:
   - ~3x speed improvement
   - No accuracy loss

6. **Use appropriate labels**:
   - Specific labels work better than generic ones
   - "company" > "entity"
   - "medication" > "word"

## Troubleshooting

### Low Accuracy

```python
# Try lowering the threshold
entities = model.predict_entities(text, labels, threshold=0.3)

# Use more specific labels
labels = ["tech_company", "software_product", "founder"]  # Specific
# instead of
labels = ["organization", "thing", "person"]  # Too generic

# Try a larger model
model = GLiNER.from_pretrained("urchade/gliner_large-v2.1")
```

### Slow Inference

```python
# Enable FlashDeBERTa
# pip install flashdeberta

# Compile model
model = GLiNER.from_pretrained(
    "urchade/gliner_small-v2.1",
    compile_torch_model=True
)

# Use batch processing
entities_batch = model.inference(texts, labels, batch_size=16)

# For BiEncoder: pre-compute embeddings
label_embeds = model.encode_labels(labels)
entities = model.predict_with_embeds(text, label_embeds, labels)
```

### Out of Memory

```python
# Reduce batch size
entities = model.inference(texts, labels, batch_size=4)

# Use a smaller model
model = GLiNER.from_pretrained("urchade/gliner_small-v2.1")

# Process on CPU
model = GLiNER.from_pretrained(
    "urchade/gliner_small-v2.1",
    map_location="cpu"
)
```