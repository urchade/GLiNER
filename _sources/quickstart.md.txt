# Quickstart

Welcome to the GLiNER Framework Quickstart Guide! This document will help you get started with the basics of using GLiNER.


## Installation

To install GLiNER, run the following command:

```bash
pip install gliner
```

## Basic Usage

Here is a simple example to get started:

```python
from gliner import GLiNER

model = GLiNER.from_pretrained("knowledgator/gliner-multitask-large-v0.5")

text = """
Microsoft was founded by Bill Gates and Paul Allen on April 4, 1975 to develop and sell BASIC interpreters for the Altair 8800. During his career at Microsoft, Gates held the positions of chairman, chief executive officer, president and chief software architect, while also being the largest individual shareholder until May 2014.
"""

labels = ["founder", "computer", "software", "position", "date"]

entities = model.predict_entities(text, labels)

for entity in entities:
    print(entity["text"], "=>", entity["label"])

```
<details>
    <summary>Expected Output</summary>
    ```bash
    Bill Gates => founder
    Paul Allen => founder
    April 4, 1975 => date
    BASIC interpreters => software
    Altair 8800 => computer
    chairman => position
    chief executive officer => position
    president => position
    chief software architect => position
    largest individual shareholder => position
    May 2014 => date
    ```
</details>

## Next Steps

- Check out the **Examples** for more use cases.

Happy coding!