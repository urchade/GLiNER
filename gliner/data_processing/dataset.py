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
                        entities = None,
                        get_negatives:bool=True):
        self._data = examples
        self.config=config
        if data_processor is not None:
            self.data_processor = data_processor
        else:
            if config.span_mode == "token_level":
                self.data_processor = TokenProcessor(config, tokenizer, words_splitter, preprocess_text=True)
            else:
                self.data_processor = SpanProcessor(config, tokenizer, words_splitter, preprocess_text=True)
        
        self.max_neg_type_ratio = int(self.config.max_neg_type_ratio)
        self.get_negatives = get_negatives
        if not entities:
            self.all_entities = self._collect_all_entities()
        else:
            self.all_entities = entities
        self.max_negatives = min(50, len(self.all_entities))

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
        return list(all_entities)

    def _get_negatives(self):
        negatives = random.sample(self.all_entities, k=self.max_negatives)
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

            raw_batch = self.data_processor.collate_raw_batch([example], negatives = curr_negatives)
            
            model_input = self.data_processor.collate_fn(raw_batch, prepare_labels=True)
            if 'span_idx' in raw_batch:
                model_input['span_idx'] = raw_batch['span_idx']
            if 'span_mask' in raw_batch:
                model_input['span_mask'] = raw_batch['span_mask']
            if 'seq_length' in raw_batch:
                model_input['text_lengths'] = raw_batch['seq_length']
            return model_input
        except Exception as e:
            print(f"Skipping getting item due to error: {e}")
            return None