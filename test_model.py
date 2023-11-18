import json
from model import GLiNER

with open('/Users/urchadezaratiana/Documents/remote-server/instruct_ner/train_instruct.json', 'r') as f:
    data = json.load(f)


config_dict = {
    "hidden_size": 768,
    "max_width": 10,
    "model_name": "bert-base-uncased",
    "fine_tune": True,
    "subtoken_pooling": "first",
    "span_mode": "endpoints",
    "dropout": 0.1,
    "sample_rate": 0.1,
}

from argparse import Namespace
config_dict = Namespace(**config_dict)

model = GLiNER(config_dict)

loader = model.create_dataloader(data, batch_size=2, entity_types=None)

iter_loader = iter(loader)

x = next(iter_loader)

o = model.predict(x)

print(o)
