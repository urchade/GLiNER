import warnings
from pathlib import Path

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoConfig

from ..utils import is_module_available, MissedPackageException
from typing import Optional, Union

IS_PEFT = is_module_available('peft')

if IS_PEFT:
    from peft import LoraConfig, get_peft_model

class DecoderTransformer(nn.Module):
    def __init__(
        self, 
        model_name, 
        config, 
        from_pretrained=False, 
        cache_dir:Optional[Union[str, Path]] = None
    ):
        super().__init__()
        decoder_config = config.labels_decoder_config
        if decoder_config is None:
            decoder_config = AutoConfig.from_pretrained(model_name, cache_dir=cache_dir)

        kwargs = {}
        custom = False
        ModelClass = AutoModelForCausalLM

        if from_pretrained:
            self.model = ModelClass.from_pretrained(model_name, trust_remote_code=True)
        else:
            if not custom:
                self.model = ModelClass.from_config(decoder_config, trust_remote_code=True)
            else:
                self.model = ModelClass(decoder_config, **kwargs)

        adapter_config_file = Path(model_name) / "adapter_config.json"

        if adapter_config_file.exists():
            if not IS_PEFT:
                warnings.warn(f"Adapter configs were detected, if you want to apply them you need to install peft package.")
            else:
                adapter_config = LoraConfig.from_pretrained(model_name)
                self.model = get_peft_model(self.model, adapter_config)

        self.config = config

    def forward(self, *args, **kwargs):
        output = self.model(*args, **kwargs)
        encoder_layer = output[0]
        return encoder_layer
    
class Decoder(nn.Module):
    def __init__(self, config, from_pretrained: bool = False, cache_dir: Optional[Union[str, Path]]= None):
        super().__init__()

        self.decoder_layer = DecoderTransformer( #transformer_model
            config.labels_decoder, config, from_pretrained, cache_dir = cache_dir
        )

        self.decoder_hidden_size = self.decoder_layer.model.config.hidden_size

    def ids_to_embeds(
        self, 
        input_ids: torch.LongTensor
    ) -> torch.FloatTensor:
        input_ids = input_ids.to(self.decoder_layer.model.device)
        embedding_layer = self.decoder_layer.model.get_input_embeddings()
        return embedding_layer(input_ids)
    
    def generate_from_embeds(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **gen_kwargs
    ):
        if attention_mask is None:
            # create one that covers the provided prefix
            attention_mask = torch.ones(
                inputs_embeds.size()[:2], dtype=torch.long, device=inputs_embeds.device
            )
        return self.decoder_layer.model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **gen_kwargs,
        )
        
    def generate(self, *args, **kwargs):
        if "inputs_embeds" in kwargs:
            return self.generate_from_embeds(*args, **kwargs)
        else:
            return self.decoder_layer.model.generate(*args, **kwargs) 
    
    def forward(self, *args, **kwargs) -> torch.Tensor:
        decoded_embeddings = self.decoder_layer(*args, **kwargs)
        return decoded_embeddings