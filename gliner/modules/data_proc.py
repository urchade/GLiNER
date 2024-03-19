import json
from tqdm import tqdm
# ast.literal_eval
import ast, re

path = 'train.json'

with open(path, 'r') as f:
    data = json.load(f)

def tokenize_text(text):
    return re.findall(r'\w+(?:[-_]\w+)*|\S', text)

def extract_entity_spans(entry):
    text = ""
    len_start = len("What describes ")
    len_end = len(" in the text?")
    entity_types = []
    entity_texts = []

    for c in entry['conversations']:
        if c['from'] == 'human' and c['value'].startswith('Text: '):
            text = c['value'][len('Text: '):]
            tokenized_text = tokenize_text(text)

        if c['from'] == 'human' and c['value'].startswith('What describes '):

            c_type = c['value'][len_start:-len_end]
            c_type = c_type.replace(' ', '_')
            entity_types.append(c_type)

        elif c['from'] == 'gpt' and c['value'].startswith('['):
            if c['value'] == '[]':
                entity_types = entity_types[:-1]
                continue

            texts_ents = ast.literal_eval(c['value'])
            # replace space to _ in texts_ents
            entity_texts.extend(texts_ents)
            num_repeat = len(texts_ents) - 1
            entity_types.extend([entity_types[-1]] * num_repeat)

    entity_spans = []
    for j, entity_text in enumerate(entity_texts):
        entity_tokens = tokenize_text(entity_text)
        matches = []
        for i in range(len(tokenized_text) - len(entity_tokens) + 1):
            if " ".join(tokenized_text[i:i + len(entity_tokens)]).lower() == " ".join(entity_tokens).lower():
                matches.append((i, i + len(entity_tokens) - 1, entity_types[j]))
        if matches:
            entity_spans.extend(matches)

    return entity_spans, tokenized_text

# Usage:
# Replace 'entry' with the specific entry from your JSON data
entry = data[17818]  # For example, taking the first entry
entity_spans, tokenized_text = extract_entity_spans(entry)
print("Entity Spans:", entity_spans)
#print("Tokenized Text:", tokenized_text)

# create a dict: {"tokenized_text": tokenized_text, "entity_spans": entity_spans}

all_data = []

for entry in tqdm(data):
    entity_spans, tokenized_text = extract_entity_spans(entry)
    all_data.append({"tokenized_text": tokenized_text, "ner": entity_spans})


with open('train_instruct.json', 'w') as f:
    json.dump(all_data, f)

