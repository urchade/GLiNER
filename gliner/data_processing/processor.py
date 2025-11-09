import random
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import List, Tuple, Dict, Union, Sequence, Optional
from concurrent.futures import ProcessPoolExecutor

import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

from .tokenizer import WordsSplitter
from .utils import (pad_2d_tensor, 
                    get_negatives, 
                    prepare_word_mask, 
                    make_mapping,
                    prepare_span_idx)

class BaseProcessor(ABC):
    def __init__(self, config, tokenizer, words_splitter):
        self.config = config
        self.transformer_tokenizer = tokenizer
        if words_splitter is None:
            self.words_splitter = WordsSplitter(splitter_type=config.words_splitter_type)
        else:
            self.words_splitter = words_splitter
        self.ent_token = config.ent_token
        self.sep_token = config.sep_token

        # Check if the tokenizer has unk_token and pad_token
        self._check_and_set_special_tokens(self.transformer_tokenizer)

    def _check_and_set_special_tokens(self, tokenizer):
        if tokenizer.unk_token is None:
            if hasattr(tokenizer, 'unk_token_id') and tokenizer.unk_token_id is not None:
                # Tokenizer has unk_token_id but not unk_token
                pass
            else:
                warnings.warn(
                    "Tokenizer missing 'unk_token'. This may cause issues.",
                    UserWarning
                )
        
        if tokenizer.pad_token is None:
            # Try to use eos_token as pad_token (common practice)
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                warnings.warn(
                    "Tokenizer missing 'pad_token'. Consider setting it explicitly.",
                    UserWarning
                )

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
    
    def prepare_inputs(
        self,
        texts: Sequence[Sequence[str]],
        entities: Union[Sequence[Sequence[str]], Dict[int, Sequence[str]], Sequence[str]],
        blank: Optional[str] = None,
        **kwargs
    ) -> Tuple[List[List[str]], List[int]]:
        input_texts: List[List[str]] = []
        prompt_lengths: List[int] = []

        for i, text in enumerate(texts):
            ents = self._select_entities(i, entities, blank)

            ents = self._maybe_remap_entities(ents)
            prompt: List[str] = []
            for ent in ents:
                prompt += [self.ent_token, str(ent)]

            prompt += self._extra_prompt_tokens(i, text, ents)

            prompt.append(self.sep_token)
            prompt_lengths.append(len(prompt))
            input_texts.append(prompt + list(text))
        return input_texts, prompt_lengths

    def _select_entities(
        self,
        i: int,
        entities: Union[Sequence[Sequence[str]], Dict[int, Sequence[str]], Sequence[str]],
        blank: Optional[str] = None,
    ) -> List[str]:
        if blank is not None:
            return [blank]
        if isinstance(entities, dict):
            return list(entities)
        if entities and isinstance(entities[0], (list, tuple, dict)):  # per-item lists
            return list(entities[i])  # type: ignore[index]
        if entities and isinstance(entities[0], str):            # same for all
            return list(entities)      # type: ignore[list-item]
        return []

    def _maybe_remap_entities(self, ents: Sequence[str]) -> List[str]:
        return list(ents)

    def _extra_prompt_tokens(
        self, i: int, text: Sequence[str], ents: Sequence[str]
    ) -> List[str]:
        """Default: no extras."""
        return []
    
    def prepare_word_mask(self, texts, tokenized_inputs, skip_first_words=None, token_level=False):
        return prepare_word_mask(
            texts,
            tokenized_inputs,
            skip_first_words=skip_first_words,
            token_level=token_level,
        )
    
    def tokenize_inputs(self, texts, entities, blank = None, **kwargs):

        input_texts, prompt_lengths = self.prepare_inputs(texts, entities, blank=blank, **kwargs)

        tokenized_inputs = self.transformer_tokenizer(
            input_texts,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            padding="longest",
        )
        words_masks = self.prepare_word_mask(texts, tokenized_inputs, prompt_lengths)
        tokenized_inputs["words_mask"] = torch.tensor(words_masks)

        return tokenized_inputs

    def batch_generate_class_mappings(self, batch_list: List[Dict], negatives: List[str]=None, 
                                        key: str = 'ner', sampled_neg: int = 100) -> Tuple[
        List[Dict[str, int]], List[Dict[int, str]]]:
        if negatives is None:
            negatives = get_negatives(batch_list, sampled_neg=sampled_neg, key=key)
        class_to_ids = []
        id_to_classes = []
        for b in batch_list:
            max_neg_type_ratio = int(self.config.max_neg_type_ratio)
            neg_type_ratio = random.randint(0, max_neg_type_ratio) if max_neg_type_ratio else 0
            
            if f"{key}_negatives" in b: # manually setting negative types
                negs_i = b[f"{key}_negatives"]
            else: # in-batch negative types
                negs_i = negatives[:len(b[key]) * neg_type_ratio] if neg_type_ratio else []

            if f"{key}_labels" in b: # labels are predefined
                types = b[f"{key}_labels"]
            else:
                types = list(set([el[-1] for el in b[key]] + negs_i))
                random.shuffle(types)
                types = types[:int(self.config.max_types)]
                
            class_to_id = {k: v for v, k in enumerate(types, start=1)}
            id_to_class = {k: v for v, k in class_to_id.items()}
            class_to_ids.append(class_to_id)
            id_to_classes.append(id_to_class)

        return class_to_ids, id_to_classes

    def collate_raw_batch(
        self,
        batch_list: List[Dict],
        entity_types: Optional[List[Union[str, List[str]]]] = None,
        negatives: Optional[List[str]] = None,
        class_to_ids: Optional[Union[Dict[str, int], List[Dict[str, int]]]] = None,
        id_to_classes: Optional[Union[Dict[int, str], List[Dict[int, str]]]] = None,
        key='ner'
    ) -> Dict:
        """Collate a raw batch with optional dynamic or provided label mappings."""

        if class_to_ids is None and entity_types is None:
            # Dynamically infer per-example mappings
            class_to_ids, id_to_classes = self.batch_generate_class_mappings(batch_list, negatives)
        elif class_to_ids is None:
            # Build mappings from entity_types
            if entity_types and isinstance(entity_types[0], list):
                # Per-example mappings
                built = [make_mapping(t) for t in entity_types]  # list of (fwd, rev)
                class_to_ids, id_to_classes = list(zip(*built))
                class_to_ids, id_to_classes = list(class_to_ids), list(id_to_classes)
            else:
                # Single mapping for all examples
                class_to_ids, id_to_classes = make_mapping(entity_types or [])

        if isinstance(class_to_ids, list):
            batch = [
                self.preprocess_example(b["tokenized_text"], b[key], class_to_ids[i])
                for i, b in enumerate(batch_list)
            ]
        else:
            batch = [
                self.preprocess_example(b["tokenized_text"], b[key], class_to_ids)
                for b in batch_list
            ]

        return self.create_batch_dict(batch, class_to_ids, id_to_classes)


    def collate_fn(self, batch, prepare_labels=True, *args, **kwargs):
        model_input_batch = self.tokenize_and_prepare_labels(batch, prepare_labels, *args, **kwargs)
        return model_input_batch
    
    @abstractmethod
    def create_batch_dict(self, batch: List[Dict], class_to_ids: List[Dict[str, int]],
                          id_to_classes: List[Dict[int, str]]) -> Dict:
        raise NotImplementedError("Subclasses should implement this method")

    def create_dataloader(self, data, entity_types=None, *args, **kwargs) -> DataLoader:
        return DataLoader(data, collate_fn=lambda x: self.collate_fn(x, entity_types), *args, **kwargs)

    
class UniEncoderSpanProcessor(BaseProcessor):    
    def preprocess_example(self, tokens, ner, classes_to_id):
        max_width = self.config.max_width
        num_tokens = len(tokens)
        if num_tokens == 0:
            tokens = ["[PAD]"]
        max_len = self.config.max_len
        if num_tokens > max_len:
            warnings.warn(f"Sentence of length {num_tokens} has been truncated to {max_len}")
            tokens = tokens[:max_len]
        num_tokens = len(tokens)
        spans_idx = prepare_span_idx(num_tokens, max_width)
        dict_lab = self.get_dict(ner, classes_to_id) if ner else defaultdict(int)
        span_label = torch.LongTensor([dict_lab[i] for i in spans_idx])
        spans_idx = torch.LongTensor(spans_idx)
        valid_span_mask = spans_idx[:, 1] > num_tokens - 1
        span_label = span_label.masked_fill(valid_span_mask, -1)

        return {
            "tokens": tokens,
            "span_idx": spans_idx,
            "span_label": span_label,
            "seq_length": num_tokens,
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
        for i in range(len(batch['tokens'])):
            tokens = batch['tokens'][i]
            classes_to_id = batch['classes_to_id'][i]
            ner = batch['entities'][i]
            num_classes = len(classes_to_id)
            spans_idx = torch.LongTensor(
                prepare_span_idx(len(tokens), self.config.max_width)
            )
            span_to_index = {
                (spans_idx[idx, 0].item(), spans_idx[idx, 1].item()): idx
                for idx in range(len(spans_idx))
            }
            labels_one_hot = torch.zeros(len(spans_idx), num_classes + 1, dtype=torch.float)
            end_token_idx = (len(tokens) - 1)
            span_labels_dict = {}
            for (start, end, label) in ner:
                span = (start, end)
                if label in classes_to_id and span in span_to_index:
                    idx = span_to_index[span]
                    class_id = classes_to_id[label]
                    labels_one_hot[idx, class_id] = 1.0
                    span_labels_dict[idx] = label
            valid_span_mask = spans_idx[:, 1] > end_token_idx
            labels_one_hot[valid_span_mask, :] = 0.0
            labels_one_hot = labels_one_hot[:, 1:]
            labels_batch.append(labels_one_hot)
        labels_batch = pad_2d_tensor(labels_batch) if len(labels_batch) > 1 else labels_batch[0].unsqueeze(0)
        return labels_batch
    
    def tokenize_and_prepare_labels(self, batch, prepare_labels, *args, **kwargs):
        tokenized_input = self.tokenize_inputs(batch['tokens'], batch['classes_to_id'])
        if prepare_labels:
            labels = self.create_labels(batch)
            tokenized_input['labels'] = labels

        return tokenized_input

class UniEncoderTokenProcessor(BaseProcessor):
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


class BaseBiEncoderProcessor(BaseProcessor):
    def __init__(self, config, tokenizer, words_splitter, labels_tokenizer):
        super().__init__(config, tokenizer, words_splitter)
        self.labels_tokenizer = labels_tokenizer

        # Check special tokens for additional tokenizers
        if self.labels_tokenizer:
            self._check_and_set_special_tokens(self.labels_tokenizer)

    def tokenize_inputs(self, texts, entities=None):
        tokenized_inputs = self.transformer_tokenizer(texts, is_split_into_words = True, return_tensors='pt',
                                                                                truncation=True, padding="longest")

        if entities is not None:
            tokenized_labels = self.labels_tokenizer(entities, return_tensors='pt', truncation=True, padding="longest")

            tokenized_inputs['labels_input_ids'] = tokenized_labels['input_ids']
            tokenized_inputs['labels_attention_mask'] = tokenized_labels['attention_mask']

        words_masks = self.prepare_word_mask(texts, tokenized_inputs, skip_first_words=None)
        tokenized_inputs['words_mask'] = torch.tensor(words_masks)
        return tokenized_inputs

    def batch_generate_class_mappings(self, batch_list: List[Dict], *args) -> Tuple[
        List[Dict[str, int]], List[Dict[int, str]]]:

        classes = []
        for b in batch_list:
            
            if "ner_negatives" in b: # manually setting negative types
                negs_i = b["ner_negatives"]
            else: # in-batch negative types
                negs_i = []

            types = list(set([el[-1] for el in b["ner"]] + negs_i))
            
            if "ner_label" in b: # labels are predefined
                types = b["ner_label"]

            classes.extend(types)
        random.shuffle(classes)
        classes = list(set(classes))[:int(self.config.max_types*len(batch_list))]
        class_to_id = {k: v for v, k in enumerate(classes, start=1)}
        id_to_class = {k: v for v, k in class_to_id.items()}

        class_to_ids = [class_to_id for i in range(len(batch_list))]
        id_to_classes = [id_to_class for i in range(len(batch_list))]

        return class_to_ids, id_to_classes
    
class BiEncoderSpanProcessor(UniEncoderSpanProcessor, BaseBiEncoderProcessor):
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

class BiEncoderTokenProcessor(UniEncoderTokenProcessor, BaseBiEncoderProcessor):   
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
    

class UniEncoderSpanDecoderProcessor(UniEncoderSpanProcessor):
    def __init__(self, config, tokenizer, words_splitter, decoder_tokenizer):
        super().__init__(config, tokenizer, words_splitter)
        self.decoder_tokenizer = decoder_tokenizer
        
        # Check special tokens for additional tokenizers
        if self.decoder_tokenizer:
            self._check_and_set_special_tokens(self.decoder_tokenizer)
    
    def tokenize_inputs(self, texts, entities, blank=None):
        input_texts, prompt_lengths = self.prepare_inputs(texts, entities, blank=blank)
        
        tokenized_inputs = self.transformer_tokenizer(
            input_texts,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            padding="longest",
        )
        words_masks = self.prepare_word_mask(texts, tokenized_inputs, skip_first_words=prompt_lengths)
        tokenized_inputs["words_mask"] = torch.tensor(words_masks)
        
        # Add decoder inputs if decoder tokenizer is available and mode is 'span'
        if self.config.decoder_mode == 'span':
            decoder_input_texts = [[f" {t}" if i else t for i, t in enumerate(tokens)] for tokens in input_texts]
            decoder_tokenized_inputs = self.decoder_tokenizer(
                decoder_input_texts,
                is_split_into_words=True,
                return_tensors="pt",
                truncation=True,
                padding="longest",
            )
            tokenized_inputs['decoder_input_ids'] = decoder_tokenized_inputs['input_ids']
            tokenized_inputs['decoder_attention_mask'] = decoder_tokenized_inputs['attention_mask']
            
            if self.config.full_decoder_context:
                decoder_words_masks = self.prepare_word_mask(
                    texts, decoder_tokenized_inputs, 
                    skip_first_words=prompt_lengths, token_level=True
                )
                tokenized_inputs['decoder_words_mask'] = torch.tensor(decoder_words_masks)
        
        return tokenized_inputs
    
    def create_labels(self, batch, blank=None):
        labels_batch = []
        decoder_label_strings = []
        
        for i in range(len(batch['tokens'])):
            tokens = batch['tokens'][i]
            classes_to_id = batch['classes_to_id'][i]
            ner = batch['entities'][i]
            num_classes = len(classes_to_id)
            
            spans_idx = torch.LongTensor(
                prepare_span_idx(len(tokens), self.config.max_width)
            )
            span_to_index = {
                (spans_idx[idx, 0].item(), spans_idx[idx, 1].item()): idx
                for idx in range(len(spans_idx))
            }
            
            if blank is not None:
                num_classes = 1
            
            labels_one_hot = torch.zeros(len(spans_idx), num_classes + 1, dtype=torch.float)
            end_token_idx = len(tokens) - 1
            used_spans = set()
            span_labels_dict = {}
            
            for (start, end, label) in ner:
                span = (start, end)
                if label in classes_to_id and span in span_to_index:
                    idx = span_to_index[span]
                    if self.config.decoder_mode == 'span':
                        class_id = classes_to_id[label] if blank is None else 1
                    else:
                        class_id = classes_to_id[label]
                    
                    if labels_one_hot[idx, class_id] == 0 and idx not in used_spans:
                        used_spans.add(idx)
                        if end <= end_token_idx:
                            labels_one_hot[idx, class_id] = 1.0
                            span_labels_dict[idx] = label
            
            valid_span_mask = spans_idx[:, 1] > end_token_idx
            labels_one_hot[valid_span_mask, :] = 0.0
            labels_one_hot = labels_one_hot[:, 1:]
            labels_batch.append(labels_one_hot)
            
            # Collect decoder label strings in order
            sorted_idxs = sorted(span_labels_dict.keys())
            for idx in sorted_idxs:
                decoder_label_strings.append(span_labels_dict[idx])
        
        labels_batch = pad_2d_tensor(labels_batch) if len(labels_batch) > 1 else labels_batch[0].unsqueeze(0)

        decoder_tokenized_input = None
        if self.config.decoder_mode == 'span':
            if not len(decoder_label_strings):
                decoder_label_strings = ['other']
            
            decoder_tokenized_input = self.decoder_tokenizer(
                decoder_label_strings,
                return_tensors="pt",
                truncation=True,
                padding="longest",
                add_special_tokens=True
            )
            decoder_input_ids = decoder_tokenized_input['input_ids']
            decoder_attention_mask = decoder_tokenized_input['attention_mask']
            decoder_labels = decoder_input_ids.clone()
            decoder_labels.masked_fill(~decoder_attention_mask.bool(), -100)
            decoder_tokenized_input["labels"] = decoder_labels
        
        return labels_batch, decoder_tokenized_input
    
    def tokenize_and_prepare_labels(self, batch, prepare_labels, *args, **kwargs):
        blank = None
        if random.uniform(0, 1)<self.config.blank_entity_prob and prepare_labels:
            blank = "entity"
        
        tokenized_input = self.tokenize_inputs(batch['tokens'], batch['classes_to_id'], blank)
        
        if prepare_labels:
            labels, decoder_tokenized_input = self.create_labels(batch, blank=blank)
            tokenized_input['labels'] = labels
            
            if decoder_tokenized_input is not None:
                tokenized_input['decoder_labels_ids'] = decoder_tokenized_input['input_ids']
                tokenized_input['decoder_labels_mask'] = decoder_tokenized_input['attention_mask']
                tokenized_input['decoder_labels'] = decoder_tokenized_input['labels']
        
        return tokenized_input

class RelationExtractionSpanProcessor(UniEncoderSpanProcessor):
    def __init__(self, config, tokenizer, words_splitter):
        super().__init__(config, tokenizer, words_splitter)
        self.rel_token = config.rel_token

    def preprocess_example(self, tokens, ner, classes_to_id, relations):
        max_width = self.config.max_width

        if len(tokens) == 0:
            tokens = ["[PAD]"]
        max_len = self.config.max_len
        if len(tokens) > max_len:
            warnings.warn(f"Sentence of length {len(tokens)} has been truncated to {max_len}")
            tokens = tokens[:max_len]
        num_tokens = len(tokens)
        spans_idx = prepare_span_idx(num_tokens, max_width)
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
    def create_batch_dict(self, batch, class_to_ids, id_to_classes, 
                                        rel_class_to_ids, rel_id_to_classes):
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

    def create_relation_labels(self, batch):
        B = len(batch['tokens'])
        entity_label = batch['span_label']
        relation_label = batch['rel_label']
        
        entities_list = batch['entities']

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
                # Convert 1-based class_id to 0-based index
                one_hots[k, class_id - 1] = 1.0
            rel_matrix[i, :len(pos_pairs), :] = one_hots

        return adj_matrix, rel_matrix

    def prepare_inputs(
        self,
        texts: Sequence[Sequence[str]],
        entities: Union[Sequence[Sequence[str]], Dict[int, Sequence[str]], Sequence[str]],
        blank: Optional[str] = None,
        relations: Union[Sequence[Sequence[str]], Dict[int, Sequence[str]], Sequence[str]] = None,
    ) -> Tuple[List[List[str]], List[int]]:
        input_texts: List[List[str]] = []
        prompt_lengths: List[int] = []

        for i, text in enumerate(texts):
            ents = self._select_entities(i, entities, blank)
            ents = self._maybe_remap_entities(ents)

            rels = self._select_entities(i, relations, blank)
            rels = self._maybe_remap_entities(rels)

            prompt: List[str] = []
            for ent in ents:
                prompt += [self.ent_token, str(ent)]
            prompt.append(self.sep_token)

            for rel in rels:
                prompt += [self.rel_token, str(rel)]
            prompt.append(self.sep_token)

            prompt_lengths.append(len(prompt))
            input_texts.append(prompt + list(text))

        return input_texts, prompt_lengths
    
    def tokenize_and_prepare_labels(self, batch, prepare_labels, *args, **kwargs):
        tokenized_input = self.tokenize_inputs(batch['tokens'], batch['classes_to_id'], blank=None,  
                                                                relations = batch['rel_class_to_ids'])
        if prepare_labels:
            labels = self.create_labels(batch)
            tokenized_input['labels'] = labels

            adj_matrix, rel_matrix = self.create_relation_labels(batch)
            tokenized_input['adj_matrix'] = adj_matrix
            tokenized_input['rel_matrix'] = rel_matrix

        return tokenized_input