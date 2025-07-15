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
    
    @torch.inference_mode()
    def generate_from_embeds(
        self,
        inputs_embeds: torch.Tensor,            # (B, L₀, D)
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 32,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        temperature: float = 1.0,
        do_sample: bool = False,
        **kwargs
    ) -> torch.LongTensor:
        """
        Greedy / sampled decoding that starts from *input embeddings*.

        Returns
        -------
        torch.LongTensor
            Final token ids of shape (B, L₀ + max_generated) with left-padded
            `pad_token_id` (if given).  Generated tokens **exclude** the prefix
            part – exactly like HF `generate`.
        """
        model = self.decoder_layer.model
        device      = inputs_embeds.device
        B, L0, D    = inputs_embeds.shape
        cfg         = model.config

        eos_token_id = (
            eos_token_id
            if eos_token_id is not None
            else getattr(cfg, "eos_token_id", None)
        )
        pad_token_id = (
            pad_token_id
            if pad_token_id is not None
            else getattr(cfg, "pad_token_id", eos_token_id)
        )

        # Attention mask for the prefix
        if attention_mask is None:
            attention_mask = torch.ones(B, L0, dtype=torch.long, device=device)

        # Storage for generated ids (we *do not* know the prefix ids!)
        generated_ids: list[list[int]] = [[] for _ in range(B)]

        # 1st forward pass: supply the whole prefix as `inputs_embeds`
        outputs = model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=True,               # enables fast incremental decoding
        )
        past_key_values = outputs.past_key_values     # cache for speed
        next_logits     = outputs.logits[:, -1]       # (B, V)
        step            = 0

        while step < max_new_tokens:
            if temperature and temperature != 1.0:
                next_logits = next_logits / temperature

            if do_sample:
                # Sample from softmax
                probs       = torch.nn.functional.softmax(next_logits, dim=-1)
                next_token  = torch.multinomial(probs, num_samples=1)  # (B, 1)
            else:
                # Greedy
                next_token  = torch.argmax(next_logits, dim=-1, keepdim=True)  # (B, 1)

            for b in range(B):
                generated_ids[b].append(next_token[b, 0].item())

            # Early-stop if *all* sequences ended
            if eos_token_id is not None:
                if torch.all(next_token.squeeze() == eos_token_id):
                    break

            # Update masks & run the *next* incremental step
            if attention_mask is not None:
                attention_mask = torch.cat(
                    [attention_mask, torch.ones(B, 1, dtype=torch.long, device=device)],
                    dim=1,
                )

            outputs = model(
                input_ids=next_token,           # (B, 1)
                attention_mask=attention_mask,
                use_cache=True,
                past_key_values=past_key_values,
            )
            past_key_values = outputs.past_key_values
            next_logits     = outputs.logits[:, -1]
            step += 1

        max_gen_len = max(len(seq) for seq in generated_ids)
        if pad_token_id is None:                         # fall-back
            pad_token_id = eos_token_id if eos_token_id is not None else 0

        final_out = torch.full(
            (B, max_gen_len), pad_token_id, dtype=torch.long, device=device
        )

        for b, seq in enumerate(generated_ids):
            if seq:                                      # may be empty!
                final_out[b, : len(seq)] = torch.tensor(
                    seq, dtype=torch.long, device=device
                )

        return final_out
    
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