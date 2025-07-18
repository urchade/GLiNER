import random
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import List, Tuple, Dict, Union
from concurrent.futures import ProcessPoolExecutor

import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

from .utils import pad_2d_tensor

# Abstract base class for handling data processing
class BaseProcessor(ABC):
    def __init__(self, config, tokenizer, words_splitter, labels_tokenizer = None, preprocess_text=False):
        self.config = config
        self.transformer_tokenizer = tokenizer
        self.labels_tokenizer = labels_tokenizer

        self.words_splitter = words_splitter
        self.ent_token = config.ent_token
        self.sep_token = config.sep_token
        self.rel_token = config.rel_token

        self.relations_layer = config.relations_layer

        self.preprocess_text = preprocess_text

        # Check if the tokenizer has unk_token and pad_token
        self._check_and_set_special_tokens(self.transformer_tokenizer)
        if self.labels_tokenizer:
            self._check_and_set_special_tokens(self.labels_tokenizer)

    def _check_and_set_special_tokens(self, tokenizer):
        # Check for unk_token
        if tokenizer.unk_token is None:
            default_unk_token = '[UNK]'
            warnings.warn(
                f"The tokenizer is missing an 'unk_token'. Setting default '{default_unk_token}'.",
                UserWarning
            )
            tokenizer.unk_token = default_unk_token

        # Check for pad_token
        if tokenizer.pad_token is None:
            default_pad_token = '[PAD]'
            warnings.warn(
                f"The tokenizer is missing a 'pad_token'. Setting default '{default_pad_token}'.",
                UserWarning
            )
            tokenizer.pad_token = default_pad_token

    @staticmethod
    def get_dict(spans: List[Tuple[int, int, str]], classes_to_id: Dict[str, int]) -> Dict[Tuple[int, int], int]:
        dict_tag = defaultdict(int)
        for span in spans:
            if span[2] in classes_to_id:
                dict_tag[(span[0], span[1])] = classes_to_id[span[2]]
        return dict_tag

    @staticmethod
    def get_rel_dict(rels: List[Tuple[int, int, str]], classes_to_id: Dict[str, int]) -> Dict[Tuple[int, int], int]:
        dict_tag = defaultdict(int)
        for rel in rels:
            if rel[1] in classes_to_id:
                dict_tag[(rel[0], rel[2])] = classes_to_id[rel[1]]
        return dict_tag
    
    @abstractmethod
    def preprocess_example(self, tokens: List[str], ner: List[Tuple[int, int, str]],
                         classes_to_id: Dict[str, int], relations = None, rel_classes_to_id = None) -> Dict:
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

    @staticmethod
    def get_negatives_rels(batch_list: List[Dict], sampled_neg: int = 5) -> List[str]:
        rel_types = []
        for b in batch_list:
            if 'rel' in b:
                types = set([el[-1] for el in b['rel']])
                rel_types.extend(list(types))
        rel_types = list(set(rel_types))
        random.shuffle(rel_types)
        return rel_types[:sampled_neg]
    
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

    def prepare_inputs(self, texts, entities, relations = None):
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
            
            if relations is not None:
                if type(relations) == dict:
                    relations_ = relations
                else:
                    relations_ = relations[id]

                for rel in relations_:
                    input_text.append(self.rel_token)
                    input_text.append(rel)
                input_text.append(self.sep_token)

            prompt_length = len(input_text)
            prompt_lengths.append(prompt_length)
            input_text.extend(text)
            input_texts.append(input_text)
        return input_texts, prompt_lengths
    
    def prepare_word_mask(self, texts, tokenized_inputs, prompt_lengths = None):
        words_masks = []
        for id in range(len(texts)):
            if prompt_lengths is not None:
                prompt_length = prompt_lengths[id]
            else:
                prompt_length = 0
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
        return words_masks
    
    def tokenize_inputs(self, texts, entities, relations = None):
        input_texts, prompt_lengths = self.prepare_inputs(texts, entities, relations = None)

        if self.preprocess_text:
            input_texts = self.prepare_texts(input_texts)
            
        tokenized_inputs = self.transformer_tokenizer(input_texts, is_split_into_words = True, return_tensors='pt',
                                                                                truncation=True, padding="longest")
        words_masks = self.prepare_word_mask(texts, tokenized_inputs, prompt_lengths)
        tokenized_inputs['words_mask'] = torch.tensor(words_masks)
        return tokenized_inputs

    def batch_generate_class_mappings(self, batch_list: List[Dict], negatives: List[str]=None) -> Tuple[
        List[Dict[str, int]], List[Dict[int, str]]]:
        if negatives is None:
            negatives = self.get_negatives(batch_list, 100)
        class_to_ids = []
        id_to_classes = []

        rel_class_to_ids = []
        rel_id_to_classes = []
        has_rel = {True for b in batch_list if 'relations' in b}
        if True in has_rel:
            negatives_rels = self.get_negatives_rels(batch_list, 100)

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

            
            if self.relations_layer is not None and True in has_rel:
                if 'relations' in b:
                    negs_i = negatives_rels[:len(b["relations"]) * neg_type_ratio] if neg_type_ratio else []
                    types = list(set([el[-1] for el in b["relations"]] + negs_i))
                else:
                    types = negatives_rels

                random.shuffle(types)
                types = types[:int(self.config.max_types)]

                if "relation_label" in b: # labels are predefined
                    types = b["relation_label"]

                rel_class_to_id = {k: v for v, k in enumerate(types, start=1)}
                rel_id_to_class = {k: v for v, k in class_to_id.items()}
                rel_class_to_ids.append(rel_class_to_id)
                rel_id_to_classes.append(rel_id_to_class)
                
        return class_to_ids, id_to_classes, rel_class_to_ids, rel_id_to_class
    
    def collate_raw_batch(self, batch_list: List[Dict], entity_types: List[Union[str, List[str]]] = None, 
                        negatives: List[str] = None, class_to_ids: Dict = None, id_to_classes: Dict = None,
                        relation_types: List[Union[str, List[str]]] = None, 
                        rel_class_to_ids: Dict = None, rel_id_to_classes: Dict = None ) -> Dict:
        def build_mapping(types):
            """Create forward and reverse mapping for types."""
            types = list(dict.fromkeys(types))  # Remove duplicates, preserve order
            mapping = {k: v for v, k in enumerate(types, start=1)}
            return mapping, {v: k for k, v in mapping.items()}
        
        if entity_types is None and class_to_ids is None:
            # Generate mappings dynamically based on batch content
            class_to_ids, id_to_classes, rel_class_to_ids, rel_id_to_class = self.batch_generate_class_mappings(batch_list, negatives)
            batch = [
                self.preprocess_example(b["tokenized_text"], b["ner"], class_to_ids[i],
                                        b['relations'] if 'relations' in b else None, 
                                        rel_class_to_ids[i] if rel_class_to_ids is not None else None) 
                for i, b in enumerate(batch_list)
            ]
        else:
            if class_to_ids is None:
                # Handle cases for entity_types being a list of strings or list of lists
                if isinstance(entity_types[0], list):  # List of lists of strings
                    class_to_ids = []
                    id_to_classes = []
                    for i, types in enumerate(entity_types):
                        mapping, rev = build_mapping(types)
                        class_to_ids.append(mapping)
                        id_to_classes.append(rev)

                    if relation_types is not None:
                        rel_class_to_ids = []
                        rel_id_to_classes = []
                    
                        for i, types in enumerate(relation_types):
                            mapping, rev = build_mapping(types)
                            rel_class_to_ids.append(mapping)
                            rel_id_to_classes.append(rev)

                    batch = [
                        self.preprocess_example(b["tokenized_text"], b["ner"], class_to_ids[i],
                                                b['relations'] if 'relations' in b else None, 
                                                rel_class_to_ids[i] if rel_class_to_ids is not None else None)
                        for i, b in enumerate(batch_list)
                    ]
                else:  # Single list of strings
                    class_to_ids = {k: v for v, k in enumerate(entity_types, start=1)}
                    id_to_classes = {v: k for k, v in class_to_ids.items()}
                    if relation_types is not None:
                        rel_class_to_ids = {k: v for v, k in enumerate(relation_types, start=1)}
                        rel_id_to_classes = {v: k for k, v in class_to_ids.items()}
                    batch = [
                        self.preprocess_example(b["tokenized_text"], b["ner"], class_to_ids,
                                                b['relations'] if 'relations' in b else None, 
                                                rel_class_to_ids) 
                        for b in batch_list
                    ]
            else:
                # Use provided mappings
                batch = [
                    self.preprocess_example(b["tokenized_text"], b["ner"], class_to_ids,
                                            b['relations'] if 'relations' in b else None, 
                                            rel_class_to_ids[i] if rel_class_to_ids is not None else None) 
                    for b in batch_list
                ]
        
        return self.create_batch_dict(batch, class_to_ids, id_to_classes, rel_class_to_ids, rel_id_to_classes)


    def collate_fn(self, batch, prepare_labels=True, *args, **kwargs):
        model_input_batch = self.tokenize_and_prepare_labels(batch, prepare_labels, *args, **kwargs)
        return model_input_batch
    
    @abstractmethod
    def create_batch_dict(self, batch: List[Dict], class_to_ids: List[Dict[str, int]],
                          id_to_classes: List[Dict[int, str]],
                          rel_class_to_ids: List[Dict[str, int]] = None,
                          rel_id_to_classes: List[Dict[int, str]] = None) -> Dict:
        raise NotImplementedError("Subclasses should implement this method")

    def create_dataloader(self, data, entity_types=None, *args, **kwargs) -> DataLoader:
        return DataLoader(data, collate_fn=lambda x: self.collate_fn(x, entity_types), *args, **kwargs)


class BaseBiEncoderProcessor(BaseProcessor):
    def tokenize_inputs(self, texts, entities=None):
        if self.preprocess_text:
            texts = self.prepare_texts(texts)
            
        tokenized_inputs = self.transformer_tokenizer(texts, is_split_into_words = True, return_tensors='pt',
                                                                                truncation=True, padding="longest")

        if entities is not None:
            tokenized_labels = self.labels_tokenizer(entities, return_tensors='pt', truncation=True, padding="longest")

            tokenized_inputs['labels_input_ids'] = tokenized_labels['input_ids']
            tokenized_inputs['labels_attention_mask'] = tokenized_labels['attention_mask']

        words_masks = self.prepare_word_mask(texts, tokenized_inputs, prompt_lengths=None)
        tokenized_inputs['words_mask'] = torch.tensor(words_masks)
        return tokenized_inputs

    def batch_generate_class_mappings(self, batch_list: List[Dict], negatives: List[str]=None) -> Tuple[
        List[Dict[str, int]], List[Dict[int, str]]]:

        classes = []
        for b in batch_list:
            max_neg_type_ratio = int(self.config.max_neg_type_ratio)
            neg_type_ratio = random.randint(0, max_neg_type_ratio) if max_neg_type_ratio else 0
            
            if "negatives" in b: # manually setting negative types
                negs_i = b["negatives"]
            else: # in-batch negative types
                negs_i = []

            types = list(set([el[-1] for el in b["ner"]] + negs_i))
            

            if "label" in b: # labels are predefined
                types = b["label"]

            classes.extend(types)
        random.shuffle(classes)
        classes = list(set(classes))[:int(self.config.max_types*len(batch_list))]
        class_to_id = {k: v for v, k in enumerate(classes, start=1)}
        id_to_class = {k: v for v, k in class_to_id.items()}

        class_to_ids = [class_to_id for i in range(len(batch_list))]
        id_to_classes = [id_to_class for i in range(len(batch_list))]

        return class_to_ids, id_to_classes
    
class SpanProcessor(BaseProcessor):    
    def preprocess_example(self, tokens, ner, classes_to_id, relations = None, rel_classes_to_id = None):
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
            "relations": relations
        }

    def create_batch_dict(self, batch, class_to_ids, id_to_classes, rel_class_to_ids = None, rel_id_to_classes = None):
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
            "rel_class_to_ids": rel_class_to_ids,
            "rel_id_to_classes": rel_id_to_classes
        }

    def create_labels(self, batch):
        labels_batch = []
        for i in range(len(batch['tokens'])):
            tokens = batch['tokens'][i]
            classes_to_id = batch['classes_to_id'][i]
            ner = batch['entities'][i]
            num_classes = len(classes_to_id)

            spans_idx = [(start, start + width)
                         for start in range(len(tokens))
                         for width in range(self.config.max_width)]
            spans_idx = torch.LongTensor(spans_idx)

            span_to_index = {
                (spans_idx[idx, 0].item(), spans_idx[idx, 1].item()): idx
                for idx in range(len(spans_idx))
            }

            labels_one_hot = torch.zeros(len(spans_idx), num_classes + 1, dtype=torch.float)

            lab_flt = []
            for span in ner:
                if span[2] in classes_to_id:
                    lab_flt.append(((span[0], span[1]), classes_to_id[span[2]]))

            for span, class_id in lab_flt:
                if span in span_to_index:
                    idx = span_to_index[span]
                    labels_one_hot[idx, class_id] = 1.0

            valid_span_mask = spans_idx[:, 1] > (len(tokens) - 1)
            labels_one_hot[valid_span_mask, :] = 0.0

            labels_one_hot = labels_one_hot[:, 1:]

            labels_batch.append(labels_one_hot)

        # Convert the list of tensors to a single tensor
        if len(labels_batch) > 1:
            labels_batch = pad_2d_tensor(labels_batch)
        else:
            labels_batch = labels_batch[0]

        return labels_batch

    def create_relation_labels(self, batch):
        B = len(batch['tokens'])
        entity_label = batch['span_label']
        relation_label = batch['rel_label']
        
        entities_list = batch['entities']
        relations_list = batch['relations']

        max_En = torch.max(torch.sum(entity_label>0, dim=-1))
        max_Rn = torch.max(torch.sum(relation_label>0, dim=-1))

        rel_classes_to_id = batch['rel_class_to_ids']
        C = len(rel_classes_to_id)

        adj_matrix = torch.zeros(B, max_En, max_En, dtype=torch.float)
        rel_matrix = torch.zeros(B, max_Rn, C, dtype=torch.float)

        for i in range(B):
            seq_len = batch['seq_length'][i].item()
            entities = entities_list[i]
            
            valid_entities = [ent for ent in entities if ent[1] <= seq_len - 1]
            N = len(valid_entities)

            valid_ent_mask = [ent[1] <= seq_len - 1 for ent in entities]
            valid_ent_old_indices = [idx for idx, is_valid in enumerate(valid_ent_mask) if is_valid]
            new_ent_idx = {old: new for new, old in enumerate(valid_ent_old_indices)}

            adj = torch.zeros(N, N)
            pos_pairs = []

            rel_idx_i = batch['rel_idx'][i]
            rel_label_i = batch['rel_label'][i]

            for k in range(rel_label_i.shape[0]):
                if rel_label_i[k] > 0:
                    e1 = rel_idx_i[k, 0].item()
                    e2 = rel_idx_i[k, 1].item()
                    if e1 in new_ent_idx and e2 in new_ent_idx:
                        new_e1 = new_ent_idx[e1]
                        new_e2 = new_ent_idx[e2]
                        adj[new_e1, new_e2] = 1.0
                        class_id = rel_label_i[k].item()
                        pos_pairs.append(class_id)

            adj_matrix[i, :N, :N] = adj

            one_hots = torch.zeros(len(pos_pairs), C)
            for k, class_id in enumerate(pos_pairs):
                one_hots[k, class_id] = 1.0
            rel_matrix[i, :len(pos_pairs), :] = one_hots

        return adj_matrix, rel_matrix

    def tokenize_and_prepare_labels(self, batch, prepare_labels, *args, **kwargs):
        tokenized_input = self.tokenize_inputs(batch['tokens'], batch['classes_to_id'])
        if prepare_labels:
            labels = self.create_labels(batch)
            tokenized_input['labels'] = labels

            if self.relations_layer is not None:
                adj_matrix, rel_matrix = self.create_relation_labels(batch)
                tokenized_input['adj_matrix'] = adj_matrix
                tokenized_input['rel_matrix'] = rel_matrix

        return tokenized_input

class SpanBiEncoderProcessor(SpanProcessor, BaseBiEncoderProcessor):   
    def tokenize_and_prepare_labels(self, batch, prepare_labels, prepare_entities=True, *args, **kwargs):
        if prepare_entities:
            if type(batch['classes_to_id']) == dict:
                entities = list(batch['classes_to_id'])
            else:
                entities = list(batch['classes_to_id'][0])
        else:
            entities = None
        tokenized_input = self.tokenize_inputs(batch['tokens'], entities)
        if prepare_labels:
            labels = self.create_labels(batch)
            tokenized_input['labels'] = labels
        return tokenized_input


class TokenProcessor(BaseProcessor):
    def preprocess_example(self, tokens, ner, classes_to_id, *args, **kwargs):
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

    def create_batch_dict(self, batch, class_to_ids, id_to_classes, *args, **kwargs):
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
            batch_size, seq_len, num_classes, 3, dtype=torch.float
        )

        for i, sentence_entities in enumerate(entities_id):      
            for st, ed, sp_label in sentence_entities:           
                sp_label -= 1                                   

                # skip entities that point beyond sequence length
                if st >= seq_len or ed >= seq_len:
                    continue

                word_labels[i, st, sp_label, 0] = 1              # start token
                word_labels[i, ed, sp_label, 1] = 1              # end token
                word_labels[i, st:ed + 1, sp_label, 2] = 1       # inside tokens (inclusive)

        return word_labels

    def tokenize_and_prepare_labels(self, batch, prepare_labels, *args, **kwargs):
        batch_size = len(batch['tokens'])
        seq_len = batch['seq_length'].max()
        num_classes = max([len(cid) for cid in batch['classes_to_id']])

        tokenized_input = self.tokenize_inputs(batch['tokens'], batch['classes_to_id'])
        
        if prepare_labels:
            labels = self.create_labels(batch['entities_id'], batch_size, seq_len, num_classes)
            tokenized_input['labels'] = labels
        return tokenized_input

class TokenBiEncoderProcessor(TokenProcessor, BaseBiEncoderProcessor):   
    def tokenize_and_prepare_labels(self, batch, prepare_labels, prepare_entities=True, **kwargs):
        if prepare_entities:
            if type(batch['classes_to_id']) == dict:
                entities = list(batch['classes_to_id'])
            else:
                entities = list(batch['classes_to_id'][0])
        else:
            entities = None
        batch_size = len(batch['tokens'])
        seq_len = batch['seq_length'].max()
        num_classes = len(entities)

        tokenized_input = self.tokenize_inputs(batch['tokens'], entities)
        
        if prepare_labels:
            labels = self.create_labels(batch['entities_id'], batch_size, seq_len, num_classes)
            tokenized_input['labels'] = labels

        return tokenized_input