> [!IMPORTANT]
> **🚀 GLiNER2 is Now Available from [Fastino Labs](https://github.com/fastino-ai)!** A unified multi-task model for NER, Text Classification & Structured Data Extraction. Check out [fastino-ai/GLiNER2 →](https://github.com/fastino-ai/GLiNER2)

# 👑 GLiNER: Generalist and Lightweight Model for Named Entity Recognition

---

<div align="center">
    <div>
        <a href="https://clickpy.clickhouse.com/dashboard/gliner"><img src="https://static.pepy.tech/badge/gliner" alt="GLiNER Downloads"></a>
        <a href="https://arxiv.org/abs/2311.08526"><img src="https://img.shields.io/badge/arXiv-2311.08526-b31b1b.svg" alt="GLiNER Paper"></a>
        <a href="https://discord.gg/Y2yVxpSQnG"><img alt="GLiNER Discord" src="https://img.shields.io/discord/1089800235347353640?logo=discord&logoColor=white&label=Discord&color=blue"></a>
        <a href="https://github.com/urchade/GLiNER"><img alt="GLiNER GitHub stars" src="https://img.shields.io/github/stars/urchade/GLiNER?style=social"></a>
        <a href="https://github.com/urchade/GLiNER/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/github/license/urchade/GLiNER?color=blue"></a>
        <br>
        <a href="https://colab.research.google.com/drive/1mhalKWzmfSTqMnR0wQBZvt9-ktTsATHB?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open GLiNER In Colab"></a>
        <a href="https://huggingface.co/spaces/urchade/gliner_mediumv2.1"><img src="https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-sm.svg" alt="Open GLiNER In HF Spaces"></a>
        <a href="https://huggingface.co/models?library=gliner&sort=trending"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-yellow" alt="HuggingFace Models"></a>
    </div>
    <br>
</div>

GLiNER is a framework for training and deploying Named Entity Recognition (NER) models that can identify any entity type using bidirectional transformer encoders (BERT-like). Beyond standard NER, GLiNER supports multiple tasks including joint entity and relation extraction through specialized architectures. It provides a practical alternative to both traditional NER models, which are limited to predefined entity types, and Large Language Models (LLMs), which offer flexibility but require significant computational resources.


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
model = GLiNER.from_pretrained("urchade/gliner_medium-v2.1")

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
## 🌟 Maintainers

<div align="center">
  <table>
    <tr>
      <td align="center">
        <strong>Urchade Zaratiana</strong><br>
        <em>Member of technical staff at Fastino</em><br>
        <a href="https://www.linkedin.com/in/urchade-zaratiana/"><img src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn" /></a>
      </td>
      <td align="center">
        <strong>Ihor Stepanov</strong><br>
        <em>Co-Founder at Knowledgator</em><br>
        <a href="https://www.linkedin.com/in/ihor-stepanov/"><img src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn" /></a>
      </td>
    </tr>
  </table>
</div>

## 👨‍💻 Model Authors
The model authors are:
* [Urchade Zaratiana](https://huggingface.co/urchade)
* [Ihor Stepanov](https://github.com/Ingvarstep)
* Nadi Tomeh
* Pierre Holat
* Thierry Charnois

## 📚 Citations

If you find GLiNER useful in your research, please consider citing our papers:

```bibtex
@inproceedings{zaratiana-etal-2024-gliner,
    title = "{GL}i{NER}: Generalist Model for Named Entity Recognition using Bidirectional Transformer",
    author = "Zaratiana, Urchade  and
      Tomeh, Nadi  and
      Holat, Pierre  and
      Charnois, Thierry",
    editor = "Duh, Kevin  and
      Gomez, Helena  and
      Bethard, Steven",
    booktitle = "Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)",
    month = jun,
    year = "2024",
    address = "Mexico City, Mexico",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.naacl-long.300",
    doi = "10.18653/v1/2024.naacl-long.300",
    pages = "5364--5376",
    abstract = "Named Entity Recognition (NER) is essential in various Natural Language Processing (NLP) applications. Traditional NER models are effective but limited to a set of predefined entity types. In contrast, Large Language Models (LLMs) can extract arbitrary entities through natural language instructions, offering greater flexibility. However, their size and cost, particularly for those accessed via APIs like ChatGPT, make them impractical in resource-limited scenarios. In this paper, we introduce a compact NER model trained to identify any type of entity. Leveraging a bidirectional transformer encoder, our model, GLiNER, facilitates parallel entity extraction, an advantage over the slow sequential token generation of LLMs. Through comprehensive testing, GLiNER demonstrate strong performance, outperforming both ChatGPT and fine-tuned LLMs in zero-shot evaluations on various NER benchmarks.",
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
## Support and funding

This project has been supported and funded by **F.initiatives** and **Laboratoire Informatique de Paris Nord**.

F.initiatives has been an expert in public funding strategies for R&D, Innovation, and Investments (R&D&I) for over 20 years. With a team of more than 200 qualified consultants, F.initiatives guides its clients at every stage of developing their public funding strategy: from structuring their projects to submitting their aid application, while ensuring the translation of their industrial and technological challenges to public funders. Through its continuous commitment to excellence and integrity, F.initiatives relies on the synergy between methods and tools to offer tailored, high-quality, and secure support.

<p align="center">
  <img src="logo/FI_COMPLET_CW.png" alt="FI Group" width="300"/>
</p>

We also extend our heartfelt gratitude to the open-source community for their invaluable contributions, which have been instrumental in the success of this project.