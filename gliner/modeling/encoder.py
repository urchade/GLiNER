import warnings
from pathlib import Path

import torch
from torch import nn
from transformers import AutoModel, AutoConfig

from .layers import LayersFuser
from ..utils import is_module_available, MissedPackageException

IS_LLM2VEC = is_module_available('llm2vec')
IS_PEFT = is_module_available('peft')
IS_TURBOT5 = is_module_available('turbot5')

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

if IS_TURBOT5:
    from turbot5.model.modeling import T5EncoderModel
else:
    from transformers import T5EncoderModel

if IS_PEFT:
    from peft import LoraConfig, get_peft_model

class Transformer(nn.Module):
    def __init__(self, model_name, config, from_pretrained=False, labels_encoder = False):
        super().__init__()
        if labels_encoder:
            encoder_config = config.labels_encoder_config
        else:
            encoder_config = config.encoder_config
        if encoder_config is None:
            encoder_config = AutoConfig.from_pretrained(model_name)
            if config.vocab_size!=-1:
                encoder_config.vocab_size = config.vocab_size

        config_name = encoder_config.__class__.__name__

        if config_name in DECODER_MODEL_MAPPING:
            if not IS_LLM2VEC:
                raise MissedPackageException(f"The llm2vec package must be installed to use this decoder model: {config_name}")
            else:
                print('Loading decoder model using LLM2Vec...')
                ModelClass = DECODER_MODEL_MAPPING[config_name]
            custom = True
            kwargs = {}
        elif config_name in {'T5Config', 'MT5Config'}:
            custom = True
            ModelClass = T5EncoderModel
            if IS_TURBOT5:
                kwargs = {"attention_type": 'flash'}
            else:
                kwargs = {}
        else:
            custom = False
            ModelClass = AutoModel

        if from_pretrained:
            self.model = ModelClass.from_pretrained(model_name, trust_remote_code=True)
        else:
            if not custom:
                self.model = ModelClass.from_config(encoder_config, trust_remote_code=True)
            else:
                self.model = ModelClass(encoder_config, **kwargs)

        adapter_config_file = Path(model_name) / "adapter_config.json"

        if adapter_config_file.exists():
            if not IS_PEFT:
                warnings.warn(f"Adapter configs were detected, if you want to apply them you need to install peft package.")
            else:
                adapter_config = LoraConfig.from_pretrained(model_name)
                self.model = get_peft_model(self.model, adapter_config)

        if config.fuse_layers:
            self.layers_fuser = LayersFuser(encoder_config.num_hidden_layers,
                                                        encoder_config.hidden_size)

        if labels_encoder:
            config.labels_encoder_config = encoder_config
        else:
            config.encoder_config = encoder_config

        self.config = config

    def forward(self, *args, **kwargs):
        if self.config.fuse_layers:
            output_hidden_states = True
        else:
            output_hidden_states = False
        output = self.model(*args, output_hidden_states = output_hidden_states, 
                                            return_dict = True,  **kwargs)
        if self.config.fuse_layers:
            encoder_layer = self.layers_fuser(output.hidden_states)
        else:
            encoder_layer = output[0]

        return encoder_layer
    
class Encoder(nn.Module):
    def __init__(self, config, from_pretrained: bool = False):
        super().__init__()

        self.bert_layer = Transformer( #transformer_model
            config.model_name, config, from_pretrained,
        )

        bert_hidden_size = self.bert_layer.model.config.hidden_size

        if config.hidden_size != bert_hidden_size:
            self.projection = nn.Linear(bert_hidden_size, config.hidden_size)

    def resize_token_embeddings(self, new_num_tokens, pad_to_multiple_of=None):
        return self.bert_layer.model.resize_token_embeddings(new_num_tokens, 
                                                                pad_to_multiple_of)

    def get_input_embeddings(self):
        return self.bert_layer.model.get_input_embeddings()
    
    def encode_text(self, input_ids, attention_mask, *args, **kwargs):
        token_embeddings = self.bert_layer(input_ids, attention_mask, *args, **kwargs)
        if hasattr(self, "projection"):
            token_embeddings = self.projection(token_embeddings)
        return token_embeddings
    
    def forward(self, *args, **kwargs) -> torch.Tensor:
        token_embeddings = self.encode_text(*args, **kwargs)
        return token_embeddings

class BiEncoder(Encoder):
    def __init__(self, config, from_pretrained: bool = False):
        super().__init__(config, from_pretrained)
        if config.labels_encoder is not None:
            self.labels_encoder = Transformer( #transformer_model
                config.labels_encoder, config, from_pretrained, True
            )
            le_hidden_size = self.labels_encoder.model.config.hidden_size

            if config.hidden_size != le_hidden_size:
                self.labels_projection = nn.Linear(le_hidden_size, config.hidden_size)
    
    def mean_pooling(self, token_embeddings, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def encode_labels(self, input_ids, attention_mask, *args, **kwargs):
        labels_embeddings = self.labels_encoder(input_ids, attention_mask, *args, **kwargs)
        if hasattr(self, "labels_projection"):
            labels_embeddings = self.labels_projection(labels_embeddings)
        labels_embeddings = self.mean_pooling(labels_embeddings, attention_mask)
        return labels_embeddings

    def forward(self, input_ids, attention_mask, 
                    labels_input_ids = None, labels_attention_mask=None, 
                                            *args, **kwargs) -> torch.Tensor:
        token_embeddings = self.encode_text(input_ids, attention_mask, *args, **kwargs)

        labels_embeddings = self.encode_labels(labels_input_ids, labels_attention_mask, *args, **kwargs)
        return token_embeddings, labels_embeddings