# GLiNER


## Overview

GLiNER is a Named Entity Recognition (NER) model capable of identifying any entity type using a bidirectional transformer encoder (BERT-like). It provides a practical alternative to traditional NER models, which are limited to predefined entities, and Large Language Models (LLMs) that, despite their flexibility, are costly and large for resource-constrained scenarios.

- Detail about the model: https://arxiv.org/abs/2311.08526
- Google colab demo: https://colab.research.google.com/drive/1mhalKWzmfSTqMnR0wQBZvt9-ktTsATHB?usp=sharing

## Usage
```python
import torch, re
from save_load import load_model

model = load_model("gliner_base.pt", model_name="microsoft/deberta-v3-base")
model = model.eval()

def extract_entities(model, text, labels, threshold=0.5):
    def tokenize_text(text):
        return re.findall(r'\w+(?:[-_]\w+)*|\S', text.replace("\n", ""))

    tokens = tokenize_text(text)
    input_x = {"tokenized_text": tokens, "ner": None}
    x = model.collate_fn([input_x], labels)
    output = model.predict(x, flat_ner=True, threshold=threshold)

    result_dict = {}

    for start, end, ent_type in output[0]:
        span_text = " ".join(tokens[start:end+1]).replace(" ' ", "'")
        result_dict[span_text] = ent_type

    return result_dict

text = """
Cristiano Ronaldo dos Santos Aveiro (Portuguese pronunciation: [kɾiʃˈtjɐnu ʁɔˈnaldu]; born 5 February 1985) is a Portuguese professional footballer who plays as a forward for and captains both Saudi Pro League club Al Nassr and the Portugal national team. Widely regarded as one of the greatest players of all time, Ronaldo has won five Ballon d'Or awards,[note 3] a record three UEFA Men's Player of the Year Awards, and four European Golden Shoes, the most by a European player. He has won 33 trophies in his career, including seven league titles, five UEFA Champions Leagues, the UEFA European Championship and the UEFA Nations League. Ronaldo holds the records for most appearances (183), goals (140) and assists (42) in the Champions League, goals in the European Championship (14), international goals (128) and international appearances (205). He is one of the few players to have made over 1,200 professional career appearances, the most by an outfield player, and has scored over 850 official senior career goals for club and country, making him the top goalscorer of all time.
"""

labels = ["person", "award", "date", "competitions", "teams"]

result = extract_entities(model, text, labels, threshold=0.5)

for k, v in result.items():
    print(f"{k}, {v}")
```

```
- Cristiano Ronaldo dos Santos Aveiro, person
- 5 February 1985, date
- Al Nassr, teams
- Portugal national team, teams
- Ballon d'Or, award
- UEFA Men's Player of the Year Awards, award
- European Golden Shoes, award
- UEFA Champions Leagues, competitions
- UEFA European Championship, competitions
- UEFA Nations League, competitions
- Champions League, competitions
- European Championship, competitions
```

## Named Entity Recognition benchmark result
![GLiNER Logo](image.png)


## Ressources
- [Pretrained Weights are available on Huggingface](https://huggingface.co/urchade)
- [Training Data](https://drive.google.com/file/d/1MKDx73hzm9sFByJMBJhHqEuBeJzW5TsL/view?usp=sharing)
- [Evaluation Data](https://drive.google.com/file/d/1T-5IbocGka35I7X3CE6yKe5N_Xg2lVKT/view)

## Citation
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
