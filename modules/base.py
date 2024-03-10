from collections import defaultdict
from typing import List, Tuple, Dict

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import random


class InstructBase(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.max_width = config.max_width
        self.base_config = config

    def get_dict(self, spans, classes_to_id):
        dict_tag = defaultdict(int)
        for span in spans:
            if span[2] in classes_to_id:
                dict_tag[(span[0], span[1])] = classes_to_id[span[2]]
        return dict_tag

    def preprocess_spans(self, tokens, ner, classes_to_id):

        max_len = self.base_config.max_len

        if len(tokens) > max_len:
            length = max_len
            tokens = tokens[:max_len]
        else:
            length = len(tokens)

        spans_idx = []
        for i in range(length):
            spans_idx.extend([(i, i + j) for j in range(self.max_width)])

        dict_lab = self.get_dict(ner, classes_to_id) if ner else defaultdict(int)

        # 0 for null labels
        span_label = torch.LongTensor([dict_lab[i] for i in spans_idx])
        spans_idx = torch.LongTensor(spans_idx)

        # mask for valid spans
        valid_span_mask = spans_idx[:, 1] > length - 1

        # mask invalid positions
        span_label = span_label.masked_fill(valid_span_mask, -1)

        return {
            'tokens': tokens,
            'span_idx': spans_idx,
            'span_label': span_label,
            'seq_length': length,
            'entities': ner,
        }

    def collate_fn(self, batch_list, entity_types=None):
        # batch_list: list of dict containing tokens, ner
        if entity_types is None:
            negs = self.get_negatives(batch_list, 100)
            class_to_ids = []
            id_to_classes = []
            for b in batch_list:
                # negs = b["negative"]
                random.shuffle(negs)

                # negs = negs[:sampled_neg]
                max_neg_type_ratio = int(self.base_config.max_neg_type_ratio)

                if max_neg_type_ratio == 0:
                    # no negatives
                    neg_type_ratio = 0
                else:
                    neg_type_ratio = random.randint(0, max_neg_type_ratio)

                if neg_type_ratio == 0:
                    # no negatives
                    negs_i = []
                else:
                    negs_i = negs[:len(b['ner']) * neg_type_ratio]

                # this is the list of all possible entity types (positive and negative)
                types = list(set([el[-1] for el in b['ner']] + negs_i))

                # shuffle (every epoch)
                random.shuffle(types)

                if len(types) != 0:
                    # prob of higher number shoul
                    # random drop
                    if self.base_config.random_drop:
                        num_ents = random.randint(1, len(types))
                        types = types[:num_ents]

                # maximum number of entities types
                types = types[:int(self.base_config.max_types)]

                # supervised training
                if "label" in b:
                    types = sorted(b["label"])

                class_to_id = {k: v for v, k in enumerate(types, start=1)}
                id_to_class = {k: v for v, k in class_to_id.items()}
                class_to_ids.append(class_to_id)
                id_to_classes.append(id_to_class)

            batch = [
                self.preprocess_spans(b["tokenized_text"], b["ner"], class_to_ids[i]) for i, b in enumerate(batch_list)
            ]

        else:
            class_to_ids = {k: v for v, k in enumerate(entity_types, start=1)}
            id_to_classes = {k: v for v, k in class_to_ids.items()}
            batch = [
                self.preprocess_spans(b["tokenized_text"], b["ner"], class_to_ids) for b in batch_list
            ]

        span_idx = pad_sequence(
            [b['span_idx'] for b in batch], batch_first=True, padding_value=0
        )

        span_label = pad_sequence(
            [el['span_label'] for el in batch], batch_first=True, padding_value=-1
        )

        return {
            'seq_length': torch.LongTensor([el['seq_length'] for el in batch]),
            'span_idx': span_idx,
            'tokens': [el['tokens'] for el in batch],
            'span_mask': span_label != -1,
            'span_label': span_label,
            'entities': [el['entities'] for el in batch],
            'classes_to_id': class_to_ids,
            'id_to_classes': id_to_classes,
        }

    @staticmethod
    def get_negatives(batch_list, sampled_neg=5):
        ent_types = []
        for b in batch_list:
            types = set([el[-1] for el in b['ner']])
            ent_types.extend(list(types))
        ent_types = list(set(ent_types))
        # sample negatives
        random.shuffle(ent_types)
        return ent_types[:sampled_neg]

    def create_dataloader(self, data, entity_types=None, **kwargs):
        return DataLoader(data, collate_fn=lambda x: self.collate_fn(x, entity_types), **kwargs)

    def set_sampling_params(self, max_types, shuffle_types, random_drop, max_neg_type_ratio, max_len):
        """
        Sets sampling parameters on the given model.

        Parameters:
        - model: The model object to update.
        - max_types: Maximum types parameter.
        - shuffle_types: Boolean indicating whether to shuffle types.
        - random_drop: Boolean indicating whether to randomly drop elements.
        - max_neg_type_ratio: Maximum negative type ratio.
        - max_len: Maximum length parameter.
        """
        self.base_config.max_types = max_types
        self.base_config.shuffle_types = shuffle_types
        self.base_config.random_drop = random_drop
        self.base_config.max_neg_type_ratio = max_neg_type_ratio
        self.base_config.max_len = max_len
