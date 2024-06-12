import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"
import argparse
import random
import json

from transformers import AutoTokenizer
from torch.utils.data import Dataset
import torch

from gliner import GLiNERConfig, GLiNER
from gliner.training import Trainer, TrainingArguments
from gliner.data_processing.collator import DataCollatorWithPadding
from gliner.utils import load_config_as_namespace
from gliner.data_processing import TokenProcessor, SpanProcessor, WordsSplitter

class GLiNERDataset(Dataset):
    def __init__(self, examples, config, tokenizer, words_splitter):
        self._data = examples
        self.config=config
        if config.span_mode == "token_level":
            self.data_processor = TokenProcessor(config, tokenizer, words_splitter)
        else:
            self.data_processor = SpanProcessor(config, tokenizer, words_splitter)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        example = self._data[idx]
        raw_batch = self.data_processor.collate_raw_batch([example])
        
        model_input = self.data_processor.collate_fn(raw_batch, prepare_labels=True)
        if 'span_idx' in raw_batch:
            model_input['span_idx'] = raw_batch['span_idx']
        if 'span_mask' in raw_batch:
            model_input['span_mask'] = raw_batch['span_mask']
        if 'seq_length' in raw_batch:
            model_input['text_lengths'] = raw_batch['seq_length']
        return model_input


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default= "config.yaml")
    parser.add_argument('--log_dir', type=str, default = 'models/')
    parser.add_argument('--compile_model', type=bool, default = False)
    args = parser.parse_args()

    config = load_config_as_namespace(args.config)
    config.log_dir = args.log_dir

    model_config = GLiNERConfig(**vars(config))

    with open(config.train_data, 'r') as f:
        data = json.load(f)

    print('Dataset size:', len(data))
    #shuffle
    random.shuffle(data)    
    print('Dataset is shuffled...')

    train_data = data[:int(len(data)*0.9)]
    test_data = data[int(len(data)*0.9):]

    print('Dataset is splitted...')


    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name)
    model_config.class_token_index=len(tokenizer)
    tokenizer.add_tokens([model_config.ent_token, model_config.sep_token])
    model_config.vocab_size = len(tokenizer)
    
    words_splitter = WordsSplitter(model_config.words_splitter_type)

    train_dataset = GLiNERDataset(train_data, model_config, tokenizer, words_splitter)
    test_dataset = GLiNERDataset(test_data, model_config, tokenizer, words_splitter)

    data_collator = DataCollatorWithPadding(model_config)

    model = GLiNER(model_config, tokenizer=tokenizer, words_splitter=words_splitter)
    model.resize_token_embeddings([model_config.ent_token, model_config.sep_token], 
                                  set_class_token_index = False,
                                  add_tokens_to_tokenizer=False)

    if args.compile_model:
        torch.set_float32_matmul_precision('high')
        model.compile()

    training_args = TrainingArguments(
        output_dir=config.log_dir,
        learning_rate=float(config.lr_encoder),
        weight_decay=float(config.weight_decay_encoder),
        others_lr=float(config.lr_others),
        others_weight_decay=float(config.weight_decay_other),
        lr_scheduler_type=config.scheduler_type,
        warmup_ratio=config.warmup_ratio,
        per_device_train_batch_size=config.train_batch_size,
        per_device_eval_batch_size=config.train_batch_size,
        max_steps=config.num_steps,
        evaluation_strategy="epoch",
        save_steps = config.eval_every,
        save_total_limit=config.save_total_limit,
        dataloader_num_workers = 8,
        use_cpu = False,
        report_to="none",
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    trainer.train()