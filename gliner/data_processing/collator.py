import torch
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from .processor import SpanProcessor, TokenProcessor
from .utils import pad_2d_tensor

class DataCollator:
    def __init__(self, config, tokenizer=None, words_splitter=None, data_processor=None, 
                        return_tokens: bool = False,
                        return_id_to_classes: bool = False,
                        return_entities: bool = False,
                        prepare_labels: bool = False,
                        entity_types = None):
        self.config=config
        if data_processor is None:
            if config.span_mode == "token_level":
                self.data_processor = TokenProcessor(config, tokenizer, words_splitter)
            else:
                self.data_processor = SpanProcessor(config, tokenizer, words_splitter)
        else:
            self.data_processor = data_processor
        self.prepare_labels = prepare_labels
        self.return_tokens = return_tokens
        self.return_id_to_classes = return_id_to_classes
        self.return_entities = return_entities
        self.entity_types = entity_types

    def __call__(self, input_x):
        raw_batch = self.data_processor.collate_raw_batch(input_x, entity_types = self.entity_types)
        
        model_input = self.data_processor.collate_fn(raw_batch, prepare_labels=self.prepare_labels)
        model_input.update({"span_idx": raw_batch['span_idx'] if 'span_idx' in raw_batch else None, 
                            "span_mask": raw_batch["span_mask"] if 'span_mask' in raw_batch else None,
                            "text_lengths": raw_batch['seq_length']})
        if self.return_tokens:
            model_input['tokens'] = raw_batch['tokens']
        if self.return_id_to_classes:
            model_input['id_to_classes'] = raw_batch['id_to_classes']
        if self.return_entities:
            model_input['entities'] = raw_batch['entities']
        return model_input

class DataCollatorWithPadding:
    def __init__(self, config=None):
        """
        Initialize the DataCollator with configs.
                """
        self.config = config

    def __call__(self, batch):
        if not batch:
            raise ValueError("Batch cannot be empty")
        batch = [item for item in batch if item is not None]
        # Extract all keys from the first item
        keys = batch[0].keys()

        # Create a dictionary to hold padded data
        padded_batch = {key: [] for key in keys}

        for key in keys:
            if key in {'tokens', 'id_to_classes', 'entities'}:
                padded_batch[key] = [item[key] for item in batch]
                continue
            # Collect data for the current key
            key_data = [item[key].squeeze(0) for item in batch]

            if isinstance(key_data[0], torch.Tensor):
                if key_data[0].dim() == 1:
                    # For 1D tensors, use pad_sequence
                    if key == 'span_label':
                        span_label = pad_sequence(key_data, batch_first=True, padding_value=-1)
                        span_mask = span_label != -1
                        padded_batch[key] = span_mask
                    else:
                        padded_batch[key] = pad_sequence(key_data, batch_first=True)
                elif key_data[0].dim() == 2: # span_idx case
                    padded_batch[key] = self._pad_2d_tensor(key_data)
                elif key == 'labels' and self.config.span_mode == 'token_level':
                    padded_batch[key] = self.pad_token_labels(key_data)
                else:
                    raise TypeError(f"Unsuported amount of dimension for key '{key}'")
            elif isinstance(key_data[0], list):
                # Pad list-like data
                max_length = max(len(seq) for seq in key_data)
                padded_batch[key] = torch.tensor(
                    [seq + [0] * (max_length - len(seq)) for seq in key_data],
                    dtype=torch.float32
                ).to(self.device)
            elif isinstance(key_data[0], (int, float)):
                # Directly convert numeric data to tensors
                padded_batch[key] = torch.tensor(key_data, dtype=torch.float32).to(self.device)
            else:
                raise TypeError(f"Unsupported data type for key '{key}': {type(key_data[0])}")

        return padded_batch
    
    def _pad_2d_tensor(self, key_data):
        padded_tensors = pad_2d_tensor(key_data)
        return padded_tensors

    def pad_token_labels(self, key_data):
        if not key_data:
            raise ValueError("The input list 'key_data' should not be empty.")

        # Determine the maximum sequence length and number of classes
        max_seq_len = max(tensor.shape[2] for tensor in key_data)
        max_num_classes = max(tensor.shape[3] for tensor in key_data)
        
        padded_tensors = []

        for tensor in key_data:
            current_seq_len = tensor.shape[2]
            current_num_classes = tensor.shape[3]

            seq_padding = max_seq_len - current_seq_len
            class_padding = max_num_classes - current_num_classes

            # Pad tensor to the maximum sequence length and number of classes
            padded_tensor = F.pad(tensor, (0, class_padding, 0, seq_padding), mode='constant', value=0)
            padded_tensors.append(padded_tensor)
        
        # Concatenate the tensors along the batch dimension
        concatenated_labels = torch.cat(padded_tensors, dim=1)
        
        return concatenated_labels