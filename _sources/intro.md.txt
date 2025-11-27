# Introduction to 👑 GLiNER

GLiNER is a Named Entity Recognition (NER) model capable of identifying any entity type using a bidirectional transformer encoder (BERT-like). It provides a practical alternative to traditional NER models, which are limited to predefined entities, and Large Language Models (LLMs) that, despite their flexibility, are costly and large for resource-constrained scenarios.

* **Paper**: 📄 [GLiNER: Generalist Model for Named Entity Recognition using Bidirectional Transformer](https://arxiv.org/abs/2311.08526)
* **Getting Started:** &nbsp; [<img align="center" src="https://colab.research.google.com/assets/colab-badge.svg" />](https://colab.research.google.com/drive/1mhalKWzmfSTqMnR0wQBZvt9-ktTsATHB?usp=sharing)
* **Demo:** 🤗 [Hugging Face](https://huggingface.co/spaces/urchade/gliner_mediumv2.1)

## Models Status
### 📢 Updates
- 🔍 Join the GLiNER **discord** server: [https://discord.gg/Y2yVxpSQnG](https://discord.gg/Y2yVxpSQnG)
- Synthetic data generation example is available (examples/synthetic_data_generation.ipynb).
- 🆕 `gliner_multi_pii-v1` is available. This version has been optimized to recognize and classify Personally Identifiable Information (PII) within text. This version has been finetuned on six languages (English, French, German, Spanish, Italian, Portugese).
- 🚀 `gliner_multi-v2.1`, `gliner_small-v2.1`, `gliner_medium-v2.1`, and `gliner_large-v2.1` are available under the Apache 2.0 license.
- 🆕 [gliner-spacy](https://github.com/theirstory/gliner-spacy) is available. Install it with `pip install gliner-spacy`. See Example of usage [below](https://github.com/urchade/GLiNER/tree/main#-usage-with-spacy).
- 🧬 `gliner_large_bio-v0.1` is a gliner model specialized for biomedical text. It is available under the Apache 2.0 license.
- 📚 Training dataset preprocessing scripts are now available in the `data/` directory, covering both [Pile-NER](https://huggingface.co/datasets/Universal-NER/Pile-NER-type) and [NuNER](https://huggingface.co/datasets/numind/NuNER) datasets.

### Finetuning GLiNER
- 📘 See this [directory](https://github.com/urchade/GLiNER/tree/main/examples/finetuning)

### Demonstating GLiNER
![GLiNER Gradio demo](assets/demo.jpg "Demo")

- 🖥️ You can run an interactive Gradio demo to play with existing or your custom GLiNER models, see the [script](https://github.com/urchade/GLiNER/tree/main/demo.py)

### 🌟 Available Models on Hugging Face

#### 🇬🇧 For English
- **GLiNER Base**: `urchade/gliner_base` *(CC BY NC 4.0)*
- **GLiNER Small**: `urchade/gliner_small` *(CC BY NC 4.0)*
- **GLiNER Small v2**: `urchade/gliner_small-v2` *(Apache 2.0)*
- **GLiNER Small v2.1**: `urchade/gliner_small-v2.1` *(Apache 2.0)*
- **GLiNER Medium**: `urchade/gliner_medium` *(CC BY NC 4.0)*
- **GLiNER Medium v2**: `urchade/gliner_medium-v2` *(Apache 2.0)*
- **GLiNER Medium v2.1**: `urchade/gliner_medium-v2.1` *(Apache 2.0)*
- **GLiNER Large**: `urchade/gliner_large` *(CC BY NC 4.0)*
- **GLiNER Large v2**: `urchade/gliner_large-v2` *(Apache 2.0)*
- **GLiNER Large v2.1**: `urchade/gliner_large-v2.1` *(Apache 2.0)*


- **GLiNER NuNerZero span**: `numind/NuNER_Zero-span`  *(MIT)* - +4.5% more powerful GLiNER Large v2.1
- **GLiNER News**: `EmergentMethods/gliner_medium_news-v2.1` *(Apache 2.0)* 9.5% improvement over GLiNER Large v2.1 on 18 benchmark datasets

##### 🇬🇧 English word-level Entity Recognition

Word-level models work **better for finding multi-word entities, highlighting sentences or paragraphs**. They require additional output postprocessing that can be found in the corresponding model card.
- **GLiNER NuNerZero**: `numind/NuNER_Zero`  *(MIT)* - +3% more powerful GLiNER Large v2.1, better suitable to detect multi-word entities
- **GLiNER NuNerZero 4k context**: `numind/NuNER_Zero-4k`  *(MIT)* - 4k-long-context NuNerZero

#### 🌍 For Other Languages
- **Korean**: 🇰🇷 `taeminlee/gliner_ko`
- **Italian**: 🇮🇹 `DeepMount00/universal_ner_ita`
- **Multilingual**: 🌐 
   - `urchade/gliner_multi` *(CC BY NC 4.0)*
   - `urchade/gliner_multi-v2.1` *(Apache 2.0)* 
   - `knowledgator/gliner-x-large-v0.5`  *(Apache 2.0)*, see other model sizes in the [collection](https://huggingface.co/collections/knowledgator/gliner-x-684320a3f1220315c651d2f5)

#### 🔬 Domain Specific Models
- **Personally Identifiable Information**: 🔍 `urchade/gliner_multi_pii-v1` *(Apache 2.0)*
    - This model is capable of recognizing various types of *personally identifiable information* (PII), including but not limited to these entity types: `person`, `organization`, `phone number`, `address`, `passport number`, `email`, `credit card number`, `social security number`, `health insurance id number`, `date of birth`, `mobile phone number`, `bank account number`, `medication`, `cpf`, `driver's license number`, `tax identification number`, `medical condition`, `identity card number`, `national id number`, `ip address`, `email address`, `iban`, `credit card expiration date`, `username`, `health insurance number`, `registration number`, `student id number`, `insurance number`, `flight number`, `landline phone number`, `blood type`, `cvv`, `reservation number`, `digital signature`, `social media handle`, `license plate number`, `cnpj`, `postal code`, `passport_number`, `serial number`, `vehicle registration number`, `credit card brand`, `fax number`, `visa number`, `insurance company`, `identity document number`, `transaction number`, `national health insurance number`, `cvc`, `birth certificate number`, `train ticket number`, `passport expiration date`, and `social_security_number`.
- **Biomedical**: 
   - 🧬 `urchade/gliner_large_bio-v0.1` *(Apache 2.0)*
   - [GLiNER-BioMed collection of models](https://huggingface.co/collections/Ihor/gliner-biomed-67b5dbb61225c634020de01a), trained on high-quality data corresponding to scientific articles, clinical trials, patents, drug labels, etc.
- **Birds attribute extraction**: 🐦 `wjbmattingly/gliner-large-v2.1-bird`  *(Apache 2.0)*

#### 📚 Multi-task Models
- **GLiNER multi-task large v0.5** `knowledgator/gliner-multitask-large-v0.5` *(Apache 2.0)* - +4.5% on NER benchmarks over GLiNER Large v2.1, supports prompting, relation extraction, summarization and question-answering tasks.
- **GLiNER multi-task v1.0** `knowledgator/gliner-multitask-v1.0` *(Apache 2.0)* - +5.0% on NER benchmarks over GLiNER Large v2.1, supports prompting, relation extraction, summarization, classification and question-answering tasks.
- **GLiNER Llama multi-task v1.0** `knowledgator/gliner-llama-multitask-1B-v1.0` *(Apache 2.0)* - +3.5% on NER benchmarks over GLiNER Large v2.1, supports prompting, relation extraction, summarization, classification and question-answering tasks.

## 🛠 Installation & Usage

To provide instructions on how to install the GLiNER model from source, you can add steps for cloning the repository and installing it manually. Here’s how you can incorporate those instructions:

---

## 🛠 Installation & Usage

To begin using the GLiNER model, you can install the GLiNER Python library through pip, conda, or directly from the source.

### Install via Pip

```bash
!pip install gliner
```

If you intend to use the GPU-backed ONNX runtime, install GLiNER with the GPU feature. This also installs the `onnxruntime-gpu` dependency.

```bash
!pip install gliner[gpu]
```

If you need to use a multi-lingual model, like this [one](https://huggingface.co/collections/knowledgator/gliner-x-684320a3f1220315c651d2f5), please install additional dependencies for running tokenisers for different languages.

```bash
!pip install gliner[tokenizers]
```

In the case of a model that uses Stanza tokeniser, you can install GLiNER with this type of tokeniser in the following way:

```bash
!pip install gliner[stanza]
```

### Install via Conda

```bash
conda install -c conda-forge gliner
```

### Install from Source

To install the GLiNER library from source, follow these steps:

1. **Clone the Repository:**

   First, clone the GLiNER repository from GitHub:

   ```bash
   git clone https://github.com/urchade/GLiNER
   ```

2. **Navigate to the Project Directory:**

   Change to the directory containing the cloned repository:

   ```bash
   cd GLiNER
   ```

3. **Install Dependencies:**

   It's a good practice to create and activate a virtual environment before installing dependencies:

   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows use: venv\Scripts\activate
   ```

   Install the required dependencies listed in the `requirements.txt` file:

   ```bash
   pip install -r requirements.txt
   ```

4. **Install the GLiNER Package:**

   Finally, install the GLiNER package using the setup script:

   ```bash
   pip install .
   ```

5. **Verify Installation:**

   You can verify the installation by importing the library in a Python script:

   ```python
   import gliner
   print(gliner.__version__)
   ```
---

### 🚀 Basic Use Case

After the installation of the GLiNER library, import the `GLiNER` class. Following this, you can load your chosen model with `GLiNER.from_pretrained` and utilize `predict_entities` to discern entities within your text.

```python
from gliner import GLiNER

# Initialize GLiNER with the base model
model = GLiNER.from_pretrained("urchade/gliner_mediumv2.1")

# Sample text for entity prediction
text = """
Cristiano Ronaldo dos Santos Aveiro (Portuguese pronunciation: [kɾiʃˈtjɐnu ʁɔˈnaldu]; born 5 February 1985) is a Portuguese professional footballer who plays as a forward for and captains both Saudi Pro League club Al Nassr and the Portugal national team. Widely regarded as one of the greatest players of all time, Ronaldo has won five Ballon d'Or awards,[note 3] a record three UEFA Men's Player of the Year Awards, and four European Golden Shoes, the most by a European player. He has won 33 trophies in his career, including seven league titles, five UEFA Champions Leagues, the UEFA European Championship and the UEFA Nations League. Ronaldo holds the records for most appearances (183), goals (140) and assists (42) in the Champions League, goals in the European Championship (14), international goals (128) and international appearances (205). He is one of the few players to have made over 1,200 professional career appearances, the most by an outfield player, and has scored over 850 official senior career goals for club and country, making him the top goalscorer of all time.
"""

# Labels for entity prediction
labels = ["Person", "Award", "Date", "Competitions", "Teams"]

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
European Championship => competitions
```

## 🛠 Areas of Improvements / research

- [ ] Decoding: allow a span to have multiple labels, eg: "Cristiano Ronaldo" is both a "person" and "football player"
- [ ] Dynamic thresholding (in ```model.predict_entities(text, labels, threshold=0.5)```): allow the model to predict more entities, or less entities, depending on the context. Actually, the model tend to predict less entities where the entity type or the domain are not well represented in the training data.
- [ ] Train with EMAs (Exponential Moving Averages) or merge multiple checkpoints to improve model robustness (see [this paper](https://openreview.net/forum?id=tq_J_MqB3UB))


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

```bibtex
@misc{stepanov2024glinermultitaskgeneralistlightweight,
      title={GLiNER multi-task: Generalist Lightweight Model for Various Information Extraction Tasks}, 
      author={Ihor Stepanov and Mykhailo Shtopko},
      year={2024},
      eprint={2406.12925},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2406.12925}, 
}
```