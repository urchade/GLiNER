# 🚀 GLiNER: Generalist and Lightweight Model for Named Entity Recognition

GLiNER is a Named Entity Recognition (NER) model capable of identifying any entity type using a bidirectional transformer encoder (BERT-like). It provides a practical alternative to traditional NER models, which are limited to predefined entities, and Large Language Models (LLMs) that, despite their flexibility, are costly and large for resource-constrained scenarios.

* **Paper**: 📄 [GLiNER: Generalist Model for Named Entity Recognition using Bidirectional Transformer](https://arxiv.org/abs/2311.08526)
* **Getting Started:** &nbsp; [<img align="center" src="https://colab.research.google.com/assets/colab-badge.svg" />](https://colab.research.google.com/drive/1mhalKWzmfSTqMnR0wQBZvt9-ktTsATHB?usp=sharing)
* **Demo:** 🤗 [Hugging Face](https://huggingface.co/spaces/urchade/gliner_mediumv2.1)

## Models Status
### 📢 Updates
- 🆕 [gliner-spacy](https://github.com/theirstory/gliner-spacy) is available. Install it with `pip install gliner-spacy`. See Example of usage below.
- 🧬 `gliner_large_bio-v0.1` is a gliner model specialized for biomedical text. It is available under the Apache 2.0 license.
- 🚀 `gliner_mediumv2.1` is available under the Apache 2.0 license. It should have similar performance as `gliner_base` and `gliner_medium`.
- 📘 Finetuning notebook is available: examples/finetune.ipynb
- 📚 Training dataset preprocessing scripts are now available in the `data/` directory, covering both [Pile-NER](https://huggingface.co/datasets/Universal-NER/Pile-NER-type) and [NuNER](https://huggingface.co/datasets/numind/NuNER) datasets.

### 🌟 Available Models on Hugging Face

#### 🇬🇧 For English
- **GLiNER Base**: `urchade/gliner_base` *(CC BY NC 4.0)*
- **GLiNER Small**: `urchade/gliner_small` *(CC BY NC 4.0)*
- **GLiNER Small v2**: `urchade/gliner_smallv2` *(Apache 2.0)*
- **GLiNER Medium**: `urchade/gliner_medium` *(CC BY NC 4.0)*
- **GLiNER Medium v2**: `urchade/gliner_mediumv2` *(Apache 2.0)*
- **GLiNER Medium v2.1**: `urchade/gliner_mediumv2.1` *(Apache 2.0)*
- **GLiNER Large**: `urchade/gliner_large` *(CC BY NC 4.0)*
- **GLiNER Large v2**: `urchade/gliner_largev2` *(Apache 2.0)*

#### 🌍 For Other Languages
- **Korean**: 🇰🇷 `taeminlee/gliner_ko`
- **Italian**: 🇮🇹 `DeepMount00/universal_ner_ita`
- **Multilingual**: 🌐 `urchade/gliner_multi` *(CC BY NC 4.0)*

#### 🔬 Domain Specific Models
- **Biomedical**: `urchade/gliner_large_bio-v0.1` *(Apache 2.0)*

## 🛠 Installation & Usage

To begin using the GLiNER model, first install the GLiNER Python library through pip:

```bash
!pip install gliner
```

### 🚀 Basic Use Case

After the installation of the GLiNER library, import the `GLiNER` class. Following this, you can load your chosen model with `GLiNER.from_pretrained` and utilize `predict_entities` to discern entities within your text.

```python
from gliner import GLiNER

# Initialize GLiNER with the base model
model = GLiNER.from_pretrained("urchade/gliner_base")

# Sample text for entity prediction
text = """
Cristiano Ronaldo dos Santos Aveiro (Portuguese pronunciation: [kɾiʃˈtjɐnu ʁɔˈnaldu]; born 5 February 1985) is a Portuguese professional footballer who plays as a forward for and captains both Saudi Pro League club Al Nassr and the Portugal national team. Widely regarded as one of the greatest players of all time, Ronaldo has won five Ballon d'Or awards,[note 3] a record three UEFA Men's Player of the Year Awards, and four European Golden Shoes, the most by a European player. He has won 33 trophies in his career, including seven league titles, five UEFA Champions Leagues, the UEFA European Championship and the UEFA Nations League. Ronaldo holds the records for most appearances (183), goals (140) and assists (42) in the Champions League, goals in the European Championship (14), international goals (128) and international appearances (205). He is one of the few players to have made over 1,200 professional career appearances, the most by an outfield player, and has scored over 850 official senior career goals for club and country, making him the top goalscorer of all time.
"""

# Labels for entity prediction
labels = ["person", "award", "date", "competitions", "teams"]

# Perform entity prediction
entities = model.predict_entities(text, labels, threshold=0.5)

# Display predicted entities and their labels
for entity in entities:
    print(entity["text"], "=>", entity["label"])
```

#### Expected Output

```
Cristiano Ronaldo dos Santos Aveiro => person
5 February 1985 => date
Al Nassr => teams
Portugal national team => teams
Ballon d'Or => award
UEFA Men's Player of the Year Awards => award
European Golden Shoes => award
UEFA Champions Leagues => competitions
UEFA European Championship => competitions
UEFA Nations League => competitions
Champions League => competitions
European Championship => competitions
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
    "gliner_model": "urchade/gliner_multi",
    "chunk_size": 250,
    "labels": ["people", "company", "punctuation"],
    "style": "ent"
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
    print(ent.text, ent.label_)
```

#### Expected Output

```
Bill Gates => person
Microsoft => organization
```

##  📊 NER Benchmark Results

<img align="center" src="https://cdn-uploads.huggingface.co/production/uploads/6317233cc92fd6fee317e030/Y5f7tK8lonGqeeO6L6bVI.png" />

## 🛠️ Areas of Improvements / research

- [ ] Allow longer context (eg. train with long context transformers such as Longformer, LED, etc.)
- [ ] Use Bi-encoder (entity encoder and span encoder) allowing precompute entity embeddings
- [ ] Filtering mechanism to reduce number of spans before final classification to save memory and computation when the number entity types is large
- [ ] Improve understanding of more detailed prompts/instruction, eg. "Find the first name of the person in the text"
- [ ] Better loss function: for instance use ```Focal Loss``` (see [this paper](https://proceedings.neurips.cc/paper/2020/file/aeb7b30ef1d024a76f21a1d40e30c302-Paper.pdf)) instead of ```BCE``` to handle class imbalance, as some entity types are more frequent than others
- [ ] Improve multi-lingual capabilities: train on more languages, and use multi-lingual training data
- [ ] Decoding: allow a span to have multiple labels, eg: "Cristiano Ronaldo" is both a "person" and "football player"
- [ ] Dynamic thresholding (in ```model.predict_entities(text, labels, threshold=0.5)```): allow the model to predict more entities, or less entities, depending on the context. Actually, the model tend to predict less entities where the entity type or the domain are not well represented in the training data.
- [ ] Train with EMAs (Exponential Moving Averages) or merge multiple checkpoints to improve model robustness (see [this paper](https://openreview.net/forum?id=tq_J_MqB3UB)
- [ ] Extend the model to relation extraction but need dataset with relation annotations. Our preliminary work [ATG](https://github.com/urchade/ATG).


## 👨‍💻 Model Authors
The model authors are:
* [Urchade Zaratiana](https://huggingface.co/urchade)
* Nadi Tomeh
* Pierre Holat
* Thierry Charnois

## 📚 Citation

If you find GLiNER useful in your research, please consider citing our paper:

```bibtex
@misc{zaratiana2023gliner,
      title={GLiNER: Generalist Model for Named Entity Recognition using Bidirectional Transformer}, 
      author={Urchade Zaratiana and Nadi Tomeh and Pierre Holat and Thierry Charnois},
      year={2023},
      eprint={2311.08526},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
We appreciate your support!
