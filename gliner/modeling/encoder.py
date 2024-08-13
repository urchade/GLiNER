import warnings
from pathlib import Path

import torch
from torch import nn
from transformers import AutoModel, AutoConfig

from ..utils import is_module_available, MissedPackageException

IS_LLM2VEC = is_module_available('llm2vec')
IS_PEFT = is_module_available('peft')


if IS_LLM2VEC:
    from llm2vec.models import MistralBiModel, LlamaBiModel, GemmaBiModel, Qwen2BiModel
    DECODER_MODEL_MAPPING = {
        "MistralConfig": MistralBiModel,
        "LlamaConfig": LlamaBiModel,
        "GemmaConfig": GemmaBiModel,
        "Qwen2Config": Qwen2BiModel
    }
else:
    DECODER_MODEL_MAPPING = {}

if IS_PEFT:
    from peft import LoraConfig, get_peft_model

class Transformer(nn.Module):
    def __init__(self, config, from_pretrained):
        super().__init__()
        if config.encoder_config is not None:
            encoder_config = config.encoder_config
        else:
            encoder_config = AutoConfig.from_pretrained(config.model_name)
            if config.vocab_size!=-1:
                encoder_config.vocab_size = config.vocab_size

        config_name = encoder_config.__class__.__name__

        if config_name in DECODER_MODEL_MAPPING:
            if not IS_LLM2VEC:
                raise MissedPackageException(f"The llm2vec package must be installed to use this decoder model: {config_name}")
            else:
                print('Loading decoder model using LLM2Vec...')
                ModelClass = DECODER_MODEL_MAPPING[config_name]
            decoder = True
        else:
            decoder = False
            ModelClass = AutoModel

        if from_pretrained:
            self.model = ModelClass.from_pretrained(config.model_name, trust_remote_code=True)
        else:
            if not decoder:
                self.model = ModelClass.from_config(encoder_config, trust_remote_code=True)
            else:
                self.model = ModelClass(encoder_config)

        adapter_config_file = Path(config.model_name) / "adapter_config.json"

        if adapter_config_file.exists():
            if not IS_PEFT:
                warnings.warn(f"Adapter configs were detected, if you want to apply them you need to install peft package.")
            else:
                adapter_config = LoraConfig.from_pretrained(config.model_name)
                self.model = get_peft_model(self.model, adapter_config)
            
    def forward(self, *args, **kwargs):
        output = self.model(*args, **kwargs)
        return output[0]
    
class Encoder(nn.Module):
    def __init__(self, config, from_pretrained: bool = False):
        super().__init__()

        self.bert_layer = Transformer( #transformer_model
            config, from_pretrained,
        )

        bert_hidden_size = self.bert_layer.model.config.hidden_size

        if config.hidden_size != bert_hidden_size:
            self.projection = nn.Linear(bert_hidden_size, config.hidden_size)

    def resize_token_embeddings(self, new_num_tokens, pad_to_multiple_of=None):
        return self.bert_layer.model.resize_token_embeddings(new_num_tokens, 
                                                                            pad_to_multiple_of)
    def forward(self, *args, **kwargs) -> torch.Tensor:
        token_embeddings = self.bert_layer(*args, **kwargs)
        if hasattr(self, "projection"):
            token_embeddings = self.projection(token_embeddings)

        return token_embeddings
