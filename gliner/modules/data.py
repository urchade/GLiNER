import random
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import List, Tuple, Dict

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader


# Abstract base class for handling data processing
class BaseData(ABC):
    def __init__(self, config):
        self.config = config

    @staticmethod
    def get_dict(spans: List[Tuple[int, int, str]], classes_to_id: Dict[str, int]) -> Dict[Tuple[int, int], int]:
        dict_tag = defaultdict(int)
        for span in spans:
            if span[2] in classes_to_id:
                dict_tag[(span[0], span[1])] = classes_to_id[span[2]]
        return dict_tag

    @abstractmethod
    def preprocess_spans(self, tokens: List[str], ner: List[Tuple[int, int, str]],
                         classes_to_id: Dict[str, int]) -> Dict:
        raise NotImplementedError("Subclasses should implement this method")

    @staticmethod
    def get_negatives(batch_list: List[Dict], sampled_neg: int = 5) -> List[str]:
        ent_types = []
        for b in batch_list:
            types = set([el[-1] for el in b['ner']])
            ent_types.extend(list(types))
        ent_types = list(set(ent_types))
        random.shuffle(ent_types)
        return ent_types[:sampled_neg]

    def batch_generate_class_mappings(self, batch_list: List[Dict]) -> Tuple[
        List[Dict[str, int]], List[Dict[int, str]]]:
        negs = self.get_negatives(batch_list, 100)
        class_to_ids = []
        id_to_classes = []
        for b in batch_list:
            random.shuffle(negs)
            max_neg_type_ratio = int(self.config.max_neg_type_ratio)
            neg_type_ratio = random.randint(0, max_neg_type_ratio) if max_neg_type_ratio else 0
            negs_i = negs[:len(b["ner"]) * neg_type_ratio] if neg_type_ratio else []

            types = list(set([el[-1] for el in b["ner"]] + negs_i))
            random.shuffle(types)
            types = types[:int(self.config.max_types)]

            if "label" in b:
                types = sorted(b["label"])

            class_to_id = {k: v for v, k in enumerate(types, start=1)}
            id_to_class = {k: v for v, k in class_to_id.items()}
            class_to_ids.append(class_to_id)
            id_to_classes.append(id_to_class)

        return class_to_ids, id_to_classes

    def collate_fn(self, batch_list: List[Dict], entity_types: List[str] = None) -> Dict:
        if entity_types is None:
            class_to_ids, id_to_classes = self.batch_generate_class_mappings(batch_list)
            batch = [self.preprocess_spans(b["tokenized_text"], b["ner"], class_to_ids[i]) for i, b in
                     enumerate(batch_list)]
        else:
            class_to_ids = {k: v for v, k in enumerate(entity_types, start=1)}
            id_to_classes = {k: v for v, k in class_to_ids.items()}
            batch = [self.preprocess_spans(b["tokenized_text"], b["ner"], class_to_ids) for b in batch_list]

        return self.create_batch_dict(batch, class_to_ids, id_to_classes)

    @abstractmethod
    def create_batch_dict(self, batch: List[Dict], class_to_ids: List[Dict[str, int]],
                          id_to_classes: List[Dict[int, str]]) -> Dict:
        raise NotImplementedError("Subclasses should implement this method")

    def create_dataloader(self, data, entity_types=None, **kwargs) -> DataLoader:
        return DataLoader(data, collate_fn=lambda x: self.collate_fn(x, entity_types), **kwargs)


# Implementation of BaseData for a specific dataset
class SpanData(BaseData):
    def preprocess_spans(self, tokens, ner, classes_to_id):
        if len(tokens) == 0:
            tokens = ["[PAD]"]
        max_len = self.config.max_len
        if len(tokens) > max_len:
            warnings.warn(f"Sentence of length {len(tokens)} has been truncated to {max_len}")
            tokens = tokens[:max_len]

        spans_idx = [(i, i + j) for i in range(len(tokens)) for j in range(self.config.max_width)]
        dict_lab = self.get_dict(ner, classes_to_id) if ner else defaultdict(int)
        span_label = torch.LongTensor([dict_lab[i] for i in spans_idx])
        spans_idx = torch.LongTensor(spans_idx)
        valid_span_mask = spans_idx[:, 1] > len(tokens) - 1
        span_label = span_label.masked_fill(valid_span_mask, -1)

        return {
            "tokens": tokens,
            "span_idx": spans_idx,
            "span_label": span_label,
            "seq_length": len(tokens),
            "entities": ner,
        }

    def create_batch_dict(self, batch, class_to_ids, id_to_classes):
        tokens = [el["tokens"] for el in batch]
        entities = [el["entities"] for el in batch]
        span_idx = pad_sequence([b["span_idx"] for b in batch], batch_first=True, padding_value=0)
        span_label = pad_sequence([el["span_label"] for el in batch], batch_first=True, padding_value=-1)
        seq_length = torch.LongTensor([el["seq_length"] for el in batch])
        span_mask = span_label != -1

        return {
            "seq_length": seq_length,
            "span_idx": span_idx,
            "tokens": tokens,
            "span_mask": span_mask,
            "span_label": span_label,
            "entities": entities,
            "classes_to_id": class_to_ids,
            "id_to_classes": id_to_classes,
        }


class TokenData(BaseData):
    def preprocess_spans(self, tokens, ner, classes_to_id):
        # Ensure there is always a token list, even if it's empty
        if len(tokens) == 0:
            tokens = ["[PAD]"]

        # Limit the length of tokens based on configuration maximum length
        max_len = self.config.max_len
        if len(tokens) > max_len:
            warnings.warn(f"Sentence of length {len(tokens)} has been truncated to {max_len}")
            tokens = tokens[:max_len]

        # Generate entity IDs based on the NER spans provided and their classes
        try: # 'NoneType' object is not iterable
            entities_id = [[i, j, classes_to_id[k]] for i, j, k in ner if k in classes_to_id]
        except TypeError:
            entities_id = []

        return {
            'tokens': tokens,
            'seq_length': len(tokens),
            'entities': ner,
            'entities_id': entities_id
        }

    def create_batch_dict(self, batch, class_to_ids, id_to_classes):
        # Extract relevant data from batch for batch processing
        tokens = [el["tokens"] for el in batch]
        seq_length = torch.LongTensor([el["seq_length"] for el in batch])
        entities = [el["entities"] for el in batch]
        entities_id = [el["entities_id"] for el in batch]

        # Assemble and return the batch dictionary
        return {
            "tokens": tokens,
            "seq_length": seq_length,
            "entities": entities,
            "entities_id": entities_id,
            "classes_to_id": class_to_ids,
            "id_to_classes": id_to_classes,
        }


# if __name__ == "__main__":
#     # config
#     from gliner.model import load_config_as_namespace
#
#     config = load_config_as_namespace("/Users/urchadezaratiana/Documents/remote-server/TokenGLiNER/configs/config_van.yaml")
#
#     from gliner.modules.base import InstructBase
#
#     import json
#     with open("/Users/urchadezaratiana/Documents/remote-server/instruct_ner/nuner_train.json", 'r') as f:
#         data = json.load(f)[:5]
#
#     mo_base = InstructBase(config)
#
#     loader_base = mo_base.create_dataloader(data, batch_size=2, shuffle=False)
#
#     x_1 = next(iter(loader_base))
#
#     mo_gliner = GLiNERData(config)
#
#     loader_gliner = mo_gliner.create_dataloader(data, batch_size=2, shuffle=False)
#
#     x_2 = next(iter(loader_gliner))
#
#     #print(x_1)
#
#     #print(x_2)
#
#     # is equal ? iterate over the keys and compare the values
#     for key in x_1.keys():
#         print(x_1[key])
#         print(x_2[key])
#         print("Above are the values for key: ", key)
#
#
#     # show for token gliner
#
#     token_gliner = TokenGLiNERData(config)
#
#     loader_token_gliner = token_gliner.create_dataloader(data, batch_size=2, shuffle=False)
#
#     x_3 = next(iter(loader_token_gliner))
#
#     for key in x_3.keys():
#         print(key)
#         print(x_3[key])
#         print("Above are the values for key: ", key)
