import warnings
from pathlib import Path

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoConfig

from ..utils import is_module_available, MissedPackageException
from typing import Optional, Union

from ..decoding.trie import LabelsTrie

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
    
    @torch.inference_mode()
    def generate_from_embeds(
        self,
        inputs_embeds: torch.Tensor,                 # (B, L0, D)
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 32,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        temperature: float = 1.0,
        do_sample: bool = False,
        labels_trie: Optional[LabelsTrie] = None,
        **kwargs
    ):
        model = self.decoder_layer.model
        device, (B, L0, _) = inputs_embeds.device, inputs_embeds.shape
        cfg = model.config

        eos_token_id = eos_token_id or cfg.eos_token_id
        pad_token_id = pad_token_id or cfg.pad_token_id or eos_token_id

        # prefix mask
        if attention_mask is None:
            attention_mask = torch.ones(B, L0, dtype=torch.long, device=device)

        out = model(inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    use_cache=True)
        past_key_values = out.past_key_values
        next_logits = out.logits[:, -1]                         # (B, V)

        unfinished = torch.ones(B, dtype=torch.bool, device=device)
        generated = [[] for _ in range(B)]

        for _ in range(max_new_tokens):
            if labels_trie is not None:
                V = next_logits.shape[1]
                mask_tensor = torch.full((B, V), -float('inf'), device=device)
                for b in range(B):
                    if unfinished[b]:
                        current_seq = generated[b]  # Tokens generated so far
                        
                        allowed_tokens = labels_trie.get(current_seq)

                        if len(allowed_tokens) == 0:
                            allowed_tokens = [eos_token_id]
                        
                        mask_tensor[b, allowed_tokens] = 0
                    else:
                         mask_tensor[b, :] = 0
                next_logits = next_logits + mask_tensor

            if temperature != 1.0:
                next_logits = next_logits / temperature

            if do_sample:
                probs = torch.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)      # (B, 1)
            
            else:
                next_token = next_logits.argmax(dim=-1, keepdim=True)     # (B, 1)

            for b in range(B):
                if unfinished[b]:
                    generated[b].append(next_token[b, 0].item())

            eos_hit = next_token.squeeze() == eos_token_id
            unfinished = unfinished & ~eos_hit
            if not unfinished.any():         
                break

            next_token = next_token.masked_fill(~unfinished.unsqueeze(1), pad_token_id)

            attention_mask = torch.cat(
                [attention_mask, torch.ones(B, 1, dtype=torch.long, device=device)],
                dim=1,
            )

            out = model(input_ids=next_token,
                        attention_mask=attention_mask,
                        past_key_values=past_key_values,
                        use_cache=True)
            past_key_values = out.past_key_values
            next_logits = out.logits[:, -1]

        max_len = max(len(seq) for seq in generated)
        out_ids = torch.full((B, max_len), pad_token_id, dtype=torch.long, device=device)
        for b, seq in enumerate(generated):
            if seq:
                out_ids[b, :len(seq)] = torch.tensor(seq, device=device)

        return out_ids 
    
    def generate_from_embeds_base(
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