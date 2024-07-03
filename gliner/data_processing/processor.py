import random
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import List, Tuple, Dict
from concurrent.futures import ProcessPoolExecutor

import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

from .utils import pad_2d_tensor

# Abstract base class for handling data processing
class BaseProcessor(ABC):
    def __init__(self, config, tokenizer, words_splitter, preprocess_text=False):
        self.config = config
        self.transformer_tokenizer = tokenizer

        self.words_splitter = words_splitter
        self.ent_token = config.ent_token
        self.sep_token = config.sep_token

        self.preprocess_text = preprocess_text

        # Check if the tokenizer has unk_token and pad_token
        self._check_and_set_special_tokens()

    def _check_and_set_special_tokens(self):
        # Check for unk_token
        if self.transformer_tokenizer.unk_token is None:
            default_unk_token = '[UNK]'
            warnings.warn(
                f"The tokenizer is missing an 'unk_token'. Setting default '{default_unk_token}'.",
                UserWarning
            )
            self.transformer_tokenizer.unk_token = default_unk_token

        # Check for pad_token
        if self.transformer_tokenizer.pad_token is None:
            default_pad_token = '[PAD]'
            warnings.warn(
                f"The tokenizer is missing a 'pad_token'. Setting default '{default_pad_token}'.",
                UserWarning
            )
            self.transformer_tokenizer.pad_token = default_pad_token

    @staticmethod
    def get_dict(spans: List[Tuple[int, int, str]], classes_to_id: Dict[str, int]) -> Dict[Tuple[int, int], int]:
        dict_tag = defaultdict(int)
        for span in spans:
            if span[2] in classes_to_id:
                dict_tag[(span[0], span[1])] = classes_to_id[span[2]]
        return dict_tag

    @abstractmethod
    def preprocess_example(self, tokens: List[str], ner: List[Tuple[int, int, str]],
                         classes_to_id: Dict[str, int]) -> Dict:
        raise NotImplementedError("Subclasses should implement this method")

    @abstractmethod
    def create_labels(self) -> torch.Tensor:
        raise NotImplementedError("Subclasses should implement this method")
    
    @abstractmethod
    def tokenize_and_prepare_labels(self):
        pass

    @staticmethod
    def get_negatives(batch_list: List[Dict], sampled_neg: int = 5) -> List[str]:
        ent_types = []
        for b in batch_list:
            types = set([el[-1] for el in b['ner']])
            ent_types.extend(list(types))
        ent_types = list(set(ent_types))
        random.shuffle(ent_types)
        return ent_types[:sampled_neg]

    def prepare_text(self, text):
        new_text = []
        for token in text:
            if not token.strip():
                new_text.append(self.transformer_tokenizer.pad_token)
            else:
                redecoded = self.transformer_tokenizer.decode(
                                    self.transformer_tokenizer.encode(token), 
                                                    skip_special_tokens=True)
                if token!=redecoded:
                    new_text.append(self.transformer_tokenizer.unk_token)
                else:
                    new_text.append(token)
        return new_text
    
    def prepare_texts(self, texts):
        texts = [self.prepare_text(text) for text in texts]
        return texts

    def tokenize_inputs(self, texts, entities):
        input_texts = []
        prompt_lengths = []
        for id, text in enumerate(texts):
            input_text = []
            if type(entities)==dict:
                entities_=entities
            else:
                entities_=entities[id]
            for ent in entities_:
                input_text.append(self.ent_token)
                input_text.append(ent)
            input_text.append(self.sep_token)
            prompt_length = len(input_text)
            prompt_lengths.append(prompt_length)
            input_text.extend(text)
            input_texts.append(input_text)

        if self.preprocess_text:
            input_texts = self.prepare_texts(input_texts)
            
        tokenized_inputs = self.transformer_tokenizer(input_texts, is_split_into_words = True, return_tensors='pt',
                                                                                truncation=True, padding="longest")
        words_masks = []
        for id in range(len(texts)):
            prompt_length = prompt_lengths[id]
            words_mask = []
            prev_word_id=None
            words_count=0
            for word_id in tokenized_inputs.word_ids(id):
                if word_id is None:
                    words_mask.append(0)
                elif word_id != prev_word_id:
                    if words_count<prompt_length:
                        words_mask.append(0)
                    else:
                        masking_word_id = word_id-prompt_length+1
                        words_mask.append(masking_word_id)
                    words_count+=1
                else:
                    words_mask.append(0)
                prev_word_id = word_id
            words_masks.append(words_mask)
        tokenized_inputs['words_mask'] = torch.tensor(words_masks)
        return tokenized_inputs

    def batch_generate_class_mappings(self, batch_list: List[Dict], negatives: List[str]=None) -> Tuple[
        List[Dict[str, int]], List[Dict[int, str]]]:
        if negatives is None:
            negatives = self.get_negatives(batch_list, 100)
        class_to_ids = []
        id_to_classes = []
        for b in batch_list:
            max_neg_type_ratio = int(self.config.max_neg_type_ratio)
            neg_type_ratio = random.randint(0, max_neg_type_ratio) if max_neg_type_ratio else 0
            
            if "negatives" in b: # manually setting negative types
                negs_i = b["negatives"]
            else: # in-batch negative types
                negs_i = negatives[:len(b["ner"]) * neg_type_ratio] if neg_type_ratio else []

            types = list(set([el[-1] for el in b["ner"]] + negs_i))
            random.shuffle(types)
            types = types[:int(self.config.max_types)]

            if "label" in b: # labels are predefined
                types = b["label"]

            class_to_id = {k: v for v, k in enumerate(types, start=1)}
            id_to_class = {k: v for v, k in class_to_id.items()}
            class_to_ids.append(class_to_id)
            id_to_classes.append(id_to_class)

        return class_to_ids, id_to_classes

    def collate_raw_batch(self, batch_list: List[Dict], entity_types: List[str] = None, negatives: List[str]=None) -> Dict:
        if entity_types is None:
            class_to_ids, id_to_classes = self.batch_generate_class_mappings(batch_list, negatives)
            batch = [self.preprocess_example(b["tokenized_text"], b["ner"], class_to_ids[i]) for i, b in
                     enumerate(batch_list)]
        else:
            class_to_ids = {k: v for v, k in enumerate(entity_types, start=1)}
            id_to_classes = {k: v for v, k in class_to_ids.items()}
            batch = [self.preprocess_example(b["tokenized_text"], b["ner"], class_to_ids) for b in batch_list]

        return self.create_batch_dict(batch, class_to_ids, id_to_classes)

    def collate_fn(self, batch, prepare_labels=True):
        model_input_batch = self.tokenize_and_prepare_labels(batch, prepare_labels)
        return model_input_batch
    
    @abstractmethod
    def create_batch_dict(self, batch: List[Dict], class_to_ids: List[Dict[str, int]],
                          id_to_classes: List[Dict[int, str]]) -> Dict:
        raise NotImplementedError("Subclasses should implement this method")

    def create_dataloader(self, data, entity_types=None, **kwargs) -> DataLoader:
        return DataLoader(data, collate_fn=lambda x: self.collate_fn(x, entity_types), **kwargs)


# Implementation of BaseData for a specific dataset
class SpanProcessor(BaseProcessor):    
    def preprocess_example(self, tokens, ner, classes_to_id):
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
        seq_length = torch.LongTensor([el["seq_length"] for el in batch]).unsqueeze(-1)
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


    def create_labels(self, batch):
        labels_batch = []
        for id in range(len(batch['tokens'])):
            tokens = batch['tokens'][id]
            classes_to_id = batch['classes_to_id'][id]
            ner = batch['entities'][id]
            num_classes = len(classes_to_id)
            spans_idx = [(i, i + j) for i in range(len(tokens)) for j in range(self.config.max_width)]
            dict_lab = self.get_dict(ner, classes_to_id) if ner else defaultdict(int)
            span_label = torch.LongTensor([dict_lab[i] for i in spans_idx])
            spans_idx = torch.LongTensor(spans_idx)
            valid_span_mask = spans_idx[:, 1] > len(tokens) - 1
            span_label = span_label.masked_fill(valid_span_mask, 0)
            labels_one_hot = F.one_hot(span_label, num_classes + 1).float()
            labels_one_hot = labels_one_hot[:, 1:]
            labels_batch.append(labels_one_hot)
        
        # Convert the list of tensors to a single tensor
        if len(labels_batch) > 1:
            labels_batch = pad_2d_tensor(labels_batch)
        else:
            labels_batch = labels_batch[0]

        return labels_batch
    
    def tokenize_and_prepare_labels(self, batch, prepare_labels):
        tokenized_input = self.tokenize_inputs(batch['tokens'], batch['classes_to_id'])
        if prepare_labels:
            labels = self.create_labels(batch)
            tokenized_input['labels'] = labels
        return tokenized_input
    
class TokenProcessor(BaseProcessor):
    def preprocess_example(self, tokens, ner, classes_to_id):
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


        example = {
            'tokens': tokens,
            'seq_length': len(tokens),
            'entities': ner,
            'entities_id': entities_id
        }
        return example

    def create_batch_dict(self, batch, class_to_ids, id_to_classes):
        # Extract relevant data from batch for batch processing
        tokens = [el["tokens"] for el in batch]
        seq_length = torch.LongTensor([el["seq_length"] for el in batch]).unsqueeze(-1)
        entities = [el["entities"] for el in batch]
        entities_id = [el["entities_id"] for el in batch]

        # Assemble and return the batch dictionary
        batch_dict = {
            "tokens": tokens,
            "seq_length": seq_length,
            "entities": entities,
            "entities_id": entities_id,
            "classes_to_id": class_to_ids,
            "id_to_classes": id_to_classes,
        }

        return batch_dict

    def create_labels(self, entities_id, batch_size, seq_len, num_classes):
        word_labels = torch.zeros(
            3, batch_size, seq_len, num_classes, dtype=torch.float
        )
        # get batch_nums and span_pos
        for i, element in enumerate(entities_id):
            for ent in element:
                st, ed, sp_label = ent
                sp_label = sp_label - 1

                # prevent indexing errors
                if st >= seq_len or ed >= seq_len:
                    continue

                word_labels[0, i, st, sp_label] = 1  # start
                word_labels[1, i, ed, sp_label] = 1  # end
                word_labels[2, i, st:ed + 1, sp_label] = 1  # inside
        return word_labels

    def tokenize_and_prepare_labels(self, batch, prepare_labels):
        batch_size = len(batch['tokens'])
        seq_len = batch['seq_length'].max()
        num_classes = max([len(cid) for cid in batch['classes_to_id']])

        tokenized_input = self.tokenize_inputs(batch['tokens'], batch['classes_to_id'])
        
        if prepare_labels:
            labels = self.create_labels(batch['entities_id'], batch_size, seq_len, num_classes)
            tokenized_input['labels'] = labels
        return tokenized_input
