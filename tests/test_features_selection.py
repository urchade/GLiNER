import pytest
import torch
from transformers import AutoTokenizer
from gliner import GLiNERConfig
from gliner.modeling.base import extract_prompt_features_and_word_embeddings
from gliner.data_processing import SpanProcessor, WordsSplitter

class TestFeaturesExtractor:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.config = GLiNERConfig()
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.config.class_token_index=len(self.tokenizer)
        self.tokenizer.add_tokens([self.config.ent_token, self.config.sep_token])
        self.splitter = WordsSplitter()
        self.base_tokens = [['Hello', 'world', '!']]
        self.tokens_with_missed = [['Hello', '', 'world', '']]
        self.labels = ['world']
        self.processor = SpanProcessor(self.config, self.tokenizer, self.splitter)

    def test_base_extraction(self):
        input_x = [{"tokenized_text": tk, "ner": None} for tk in self.base_tokens]
        raw_batch = self.processor.collate_raw_batch(input_x, self.labels)
        model_input = self.processor.collate_fn(raw_batch, prepare_labels=False)
        model_input['text_lengths'] = raw_batch['seq_length']
        token_embeds = torch.rand(model_input['words_mask'].shape + (self.config.hidden_size,))
        
        (prompts_embedding,
         prompts_embedding_mask,
         words_embedding,
         mask) = extract_prompt_features_and_word_embeddings(self.config, token_embeds, **model_input)
        
        assert prompts_embedding_mask.shape == (1, 1)
        assert prompts_embedding.shape == (1, 1, self.config.hidden_size)
        assert words_embedding.shape == (1, len(self.base_tokens[0]), self.config.hidden_size)
        
    def test_extraction_with_missed_tokens(self):
        input_x = [{"tokenized_text": tk, "ner": None} for tk in self.tokens_with_missed]
        raw_batch = self.processor.collate_raw_batch(input_x, self.labels)
        model_input = self.processor.collate_fn(raw_batch, prepare_labels=False)
        model_input['text_lengths'] = raw_batch['seq_length']
        token_embeds = torch.rand(model_input['words_mask'].shape + (self.config.hidden_size,))
        
        (prompts_embedding,
         prompts_embedding_mask,
         words_embedding,
         mask) = extract_prompt_features_and_word_embeddings(self.config, token_embeds, **model_input)
        
        assert prompts_embedding_mask.shape == (1, 1)
        assert prompts_embedding.shape == (1, 1, self.config.hidden_size)
        assert words_embedding.shape == (1, len(self.tokens_with_missed[0]), self.config.hidden_size)
        
