# GLiNER: Generalist Model for Named Entity Recognition using Bidirectional Transformer

![GLiNER Logo](image.png)

## Overview
Named Entity Recognition (NER) is essential in various Natural Language Processing (NLP) applications. Traditional NER models are effective but limited to a set of predefined entity types. In contrast, Large Language Models (LLMs) can extract arbitrary entities through natural language instructions, offering greater flexibility. However, their size and cost, particularly for those accessed via APIs like ChatGPT, make them impractical in resource-limited scenarios. In this paper, we introduce a compact NER model trained to identify any type of entity. Leveraging a bidirectional transformer encoder, our model, GLiNER, facilitates parallel entity extraction, an advantage over the slow sequential token generation of LLMs. Through comprehensive testing, GLiNER demonstrate strong performance, outperforming both ChatGPT and fine-tuned LLMs in zero-shot evaluations on various NER benchmarks. 

## Google colab demo:
- Demo link: https://colab.research.google.com/drive/1mhalKWzmfSTqMnR0wQBZvt9-ktTsATHB?usp=sharing

## Pretrained Weight
- Download the pre-trained weights for GLiNER using the following link: [Pretrained Weight](https://drive.google.com/file/d/100aMdyzk5EC6Rl2kzLmLvMKbHz3Btt34/view?usp=sharing)

## Training Data
- Find the training data used for training GLiNER here: [Training Data](https://drive.google.com/file/d/1MKDx73hzm9sFByJMBJhHqEuBeJzW5TsL/view?usp=sharing)

## Evaluation Data
- Evaluate the performance of GLiNER using the provided evaluation data (provided by instruction_IE): [Evaluation Data](https://drive.google.com/file/d/1T-5IbocGka35I7X3CE6yKe5N_Xg2lVKT/view)

## Paper
- Arxiv link: https://arxiv.org/abs/2311.08526

## Contact
If you have any questions or need further assistance please raise an issue.

We hope GLiNER proves to be a valuable resource for your Named Entity Recognition tasks. Thank you for your interest in our project!

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
