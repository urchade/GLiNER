# 👑 GLiNER: Generalist and Lightweight Model for Named Entity Recognition

GLiNER is a Named Entity Recognition (NER) model capable of identifying any entity type using a bidirectional transformer encoder (BERT-like). It provides a practical alternative to traditional NER models, which are limited to predefined entities, and Large Language Models (LLMs) that, despite their flexibility, are costly and large for resource-constrained scenarios.

<p align="center">
    <a href="https://pypi.org/project/gliner/" target="_blank">
        <img alt="Python" src="https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54" />
        <img alt="Version" src="https://img.shields.io/pypi/v/gliner?style=for-the-badge&color=3670A0">
    </a>
</p>

<p align="center">
    <a href="https://arxiv.org/abs/2311.08526">📄 Paper</a>
    <span>&nbsp;&nbsp;•&nbsp;&nbsp;</span>
    <a href="https://discord.gg/Y2yVxpSQnG">📢 Discord</a>
    <span>&nbsp;&nbsp;•&nbsp;&nbsp;</span>
    <a href="https://huggingface.co/spaces/urchade/gliner_mediumv2.1">🤗 Demo</a>
    <span>&nbsp;&nbsp;•&nbsp;&nbsp;</span>
    <a href="https://huggingface.co/models?library=gliner&sort=trending">🤗 Available models</a>
    <span>&nbsp;&nbsp;•&nbsp;&nbsp;</span>
    <a href="https://colab.research.google.com/drive/1mhalKWzmfSTqMnR0wQBZvt9-ktTsATHB?usp=sharing">
        <img align="center" src="https://colab.research.google.com/assets/colab-badge.svg" />
    </a>
</p>

## Example Notebooks

Explore various examples including finetuning, ONNX conversion, and synthetic data generation. 

- [Example Notebooks](https://github.com/urchade/GLiNER/tree/main/examples)
- Finetune on Colab &nbsp;[<img align="center" src="https://colab.research.google.com/assets/colab-badge.svg" />](https://colab.research.google.com/drive/1HNKd74cmfS9tGvWrKeIjSxBt01QQS7bq?usp=sharing)
## 🛠 Installation & Usage

### Installation
```bash
!pip install gliner
```

### Usage
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
# Most GLiNER models should work best when entity types are in lower case or title case
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
## Support and funding

This project has been supported and funded by **FI Group** and **Laboratoire Informatique de Paris Nord**.

Over the past 20 years, [FI Group](https://fr.fi-group.com) has become a specialist in public funding strategies for R&D&I² (Research and Development, Innovation and Investment). FI Group's consultants, all engineers or PhDs, support customers from R&D through to the production of their innovations.

<p align="center">
  <img src="logo/FI Group.png" alt="FI Group" width="200"/>
</p>

We also extend our heartfelt gratitude to the open-source community for their invaluable contributions, which have been instrumental in the success of this project.


