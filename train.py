import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"
import argparse
import random
import json

from transformers import AutoTokenizer
import torch

from gliner import GLiNERConfig, GLiNER
from gliner.training import Trainer, TrainingArguments
from gliner.data_processing.collator import DataCollatorWithPadding, DataCollator
from gliner.utils import load_config_as_namespace
from gliner.data_processing import WordsSplitter, GLiNERDataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default= "configs/config.yaml")
    parser.add_argument('--log_dir', type=str, default = 'models/')
    parser.add_argument('--compile_model', type=bool, default = False)
    parser.add_argument('--freeze_text_encoder', type=bool, default = False)
    parser.add_argument('--freeze_decoder', type=bool, default = False)
    parser.add_argument('--freeze_labels_encoder', type=bool, default = False)
    parser.add_argument('--new_data_schema', type=bool, default = False)
    args = parser.parse_args()
    
    config = load_config_as_namespace(args.config)
    config.log_dir = args.log_dir

    with open(config.train_data, 'r') as f:
        data = [item for item in json.load(f) if len(item['tokenized_text']) and len(item['ner'])]

    print('Dataset size:', len(data))
    #shuffle
    random.shuffle(data)    
    print('Dataset is shuffled...')

    train_data = data[:int(len(data)*0.9)]
    test_data = data[int(len(data)*0.9):]

    print('Dataset is splitted...')


    if config.prev_path is not None:
        tokenizer = AutoTokenizer.from_pretrained(config.prev_path, add_prefix_space=True)
        model = GLiNER.from_pretrained(config.prev_path)
        model_config = model.config
    else:
        model_config = GLiNERConfig(**vars(config))
        tokenizer = AutoTokenizer.from_pretrained(model_config.model_name, add_prefix_space=True)
    
        words_splitter = WordsSplitter(model_config.words_splitter_type)

        model = GLiNER(model_config, tokenizer=tokenizer, words_splitter=words_splitter)

        if not config.labels_encoder:
            model_config.class_token_index=len(tokenizer)
            tokenizer.add_tokens([model_config.ent_token, model_config.sep_token], special_tokens=True)
            model_config.vocab_size = len(tokenizer)
            model.resize_token_embeddings([model_config.ent_token, model_config.sep_token], 
                                        set_class_token_index = False,
                                        add_tokens_to_tokenizer=False)

    if args.compile_model:
        torch.set_float32_matmul_precision('high')
        model.compile_for_training()
        
    if args.freeze_text_encoder:
        model.model.token_rep_layer.bert_layer.model.requires_grad_(False)
    else:
        model.model.token_rep_layer.bert_layer.model.requires_grad_(True)

    if model.config.labels_encoder is not None:
        if args.freeze_labels_encoder:
            model.model.token_rep_layer.labels_encoder.model.requires_grad_(False)
        else:
            model.model.token_rep_layer.labels_encoder.model.requires_grad_(True)

    if model.config.labels_decoder is not None:
        if args.freeze_decoder:
            model.model.decoder.decoder_layer.model.requires_grad_(False)
        else:
            model.model.decoder.decoder_layer.model.requires_grad_(True)

    if args.new_data_schema:
        train_dataset = GLiNERDataset(train_data, model_config, tokenizer, words_splitter)
        test_dataset = GLiNERDataset(test_data, model_config, tokenizer, words_splitter)
        data_collator = DataCollatorWithPadding(model_config)
    else:
        train_dataset = train_data
        test_dataset = test_data
        data_collator = DataCollator(model.config, data_processor=model.data_processor, prepare_labels=True)

    training_args = TrainingArguments(
        output_dir=config.log_dir,
        learning_rate=float(config.lr_encoder),
        weight_decay=float(config.weight_decay_encoder),
        others_lr=float(config.lr_others),
        others_weight_decay=float(config.weight_decay_other),
        focal_loss_gamma=config.loss_gamma,
        focal_loss_alpha=config.loss_alpha,
        focal_loss_prob_margin = config.loss_prob_margin,
        loss_reduction=config.loss_reduction,
        negatives=config.negatives,
        masking=config.masking,
        lr_scheduler_type=config.scheduler_type,
        warmup_ratio=config.warmup_ratio,
        per_device_train_batch_size=config.train_batch_size,
        per_device_eval_batch_size=config.train_batch_size,
        max_grad_norm=config.max_grad_norm,
        max_steps=config.num_steps,
        save_steps = config.eval_every,
        save_total_limit=config.save_total_limit,
        dataloader_num_workers = 1,
        logging_steps=10,
        use_cpu = False,
        report_to="none",
        bf16=True,
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
