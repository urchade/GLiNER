from datasets import load_dataset
import re
import ast
import json
from tqdm import tqdm


def tokenize_text(text):
    """Tokenizes the input text into a list of tokens."""
    return re.findall(r'\w+(?:[-_]\w+)*|\S', text)


def process_entities(dataset):
    """Processes entities in the dataset to extract tokenized text and named entity spans."""
    all_data = []
    for el in tqdm(dataset["entity"]):
        try:
            tokenized_text = tokenize_text(el["input"])
            parsed_output = ast.literal_eval(el["output"])
            entity_texts, entity_types = zip(*[i.split(" <> ") for i in parsed_output])

            entity_spans = []
            for j, entity_text in enumerate(entity_texts):
                entity_tokens = tokenize_text(entity_text)
                matches = []
                for i in range(len(tokenized_text) - len(entity_tokens) + 1):
                    if " ".join(tokenized_text[i:i + len(entity_tokens)]).lower() == " ".join(entity_tokens).lower():
                        matches.append((i, i + len(entity_tokens) - 1, entity_types[j]))
                if matches:
                    entity_spans.extend(matches)

        except Exception as e:
            continue

        all_data.append({"tokenized_text": tokenized_text, "ner": entity_spans})
    return all_data


def save_data_to_file(data, filepath):
    """Saves the processed data to a JSON file."""
    with open(filepath, 'w') as f:
        json.dump(data, f)


if __name__ == "__main__":
    dataset = load_dataset("numind/NuNER")
    processed_data = process_entities(dataset)

    save_data_to_file(processed_data, 'nuner_train.json')

    print("dataset size:", len(processed_data))