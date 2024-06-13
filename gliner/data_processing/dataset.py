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
                        entities: List[str] = None):
        self._data = examples
        self.config=config
        if data_processor is not None:
            self.data_processor = data_processor
        else:
            if config.span_mode == "token_level":
                self.data_processor = TokenProcessor(config, tokenizer, words_splitter)
            else:
                self.data_processor = SpanProcessor(config, tokenizer, words_splitter)
        
        self.return_tokens = return_tokens
        self.return_id_to_classes = return_id_to_classes
        self.prepare_labels = prepare_labels
        self.return_entities = return_entities
        self.entities = entities

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        try:
            example = self._data[idx]
            raw_batch = self.data_processor.collate_raw_batch([example], entity_types=self.entities)
            
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