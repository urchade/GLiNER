import random
from tqdm import tqdm 
from typing import Optional, List
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from . import TokenProcessor, SpanProcessor, WordsSplitter
from ..config import GLiNERConfig

class GLiNERDataset(Dataset):
    def __init__(self, examples, 
                        config: Optional[GLiNERConfig], 
                        tokenizer: Optional[AutoTokenizer] = None, 
                        words_splitter: Optional[WordsSplitter] = None,
                        data_processor = None, 
                        return_tokens: bool = False,
                        return_id_to_classes: bool = False,
                        return_entities: bool = False,
                        prepare_labels: bool = True,
                        entities: List[str] = None,
                        get_negatives=True):
        self._data = examples
        self.config=config
        if data_processor is not None:
            self.data_processor = data_processor
        else:
            if config.span_mode == "token_level":
                self.data_processor = TokenProcessor(config, tokenizer, words_splitter, preprocess_text=True)
            else:
                self.data_processor = SpanProcessor(config, tokenizer, words_splitter, preprocess_text=True)
        
        self.return_tokens = return_tokens
        self.return_id_to_classes = return_id_to_classes
        self.prepare_labels = prepare_labels
        self.return_entities = return_entities
        self.max_neg_type_ratio = int(self.config.max_neg_type_ratio)
        if not entities:
            self.all_entities = self._collect_all_entities()
        else:
            self.all_entities = entities
        self.get_negatives = get_negatives

    def _get_entities_from_example(self, example):
        entities = {ner[-1] for ner in example['ner']}
        return entities
    
    def _collect_all_entities(self):
        print("Collecting all entities...")
        all_entities = set()
        for example in tqdm(self._data):
            curr_entities = self._get_entities_from_example(example)
            all_entities.update(curr_entities)
        print('Total number of entity classes: ', len(all_entities))
        return all_entities

    def _get_negatives(self):
        negatives = random.sample(self.all_entities, k=50)
        random.shuffle(negatives)
        return negatives
    
    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        try:
            example = self._data[idx]
            if self.get_negatives:
                curr_negatives = self._get_negatives()
            else:
                curr_negatives = None

            raw_batch = self.data_processor.collate_raw_batch([example], entity_types = self.all_entities,
                                                                                negatives=curr_negatives)
            
            model_input = self.data_processor.collate_fn(raw_batch, prepare_labels=self.prepare_labels)
            if 'span_idx' in raw_batch:
                model_input['span_idx'] = raw_batch['span_idx']
            if 'span_mask' in raw_batch:
                model_input['span_mask'] = raw_batch['span_mask']
            if 'seq_length' in raw_batch:
                model_input['text_lengths'] = raw_batch['seq_length']
            if self.return_tokens:
                model_input['tokens'] = raw_batch['tokens'][0]
            if self.return_id_to_classes:
                model_input['id_to_classes'] = raw_batch['id_to_classes']
            if self.return_entities:
                model_input['entities'] = raw_batch['entities'][0]
            return model_input
        except Exception as e:
            print(f"Skipping getting item due to error: {e}")
            return None