import warnings
from pathlib import Path

import torch
from torch import nn
from transformers import AutoModel, AutoConfig
from transformers.modeling_outputs import BaseModelOutput

from .layers import LayersFuser
from ..utils import is_module_available, MissedPackageException
from ..infer_packing import InferencePackingConfig, pack_requests, unpack_spans
from typing import Optional, Union, List

IS_LLM2VEC = is_module_available('llm2vec')
IS_PEFT = is_module_available('peft')
IS_TURBOT5 = is_module_available('turbot5')
IS_FLASHDEBERTA = is_module_available('flashdeberta')

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

if IS_FLASHDEBERTA:
    from flashdeberta import FlashDebertaV2Model as DebertaV2Model
else:
    from transformers import DebertaV2Model

if IS_PEFT:
    from peft import LoraConfig, get_peft_model

class Transformer(nn.Module):
    def __init__(
        self, 
        model_name, 
        config, 
        from_pretrained=False, 
        labels_encoder = False, 
        cache_dir:Optional[Union[str, Path]] = None
    ):
        super().__init__()
        if labels_encoder:
            encoder_config = config.labels_encoder_config
        else:
            encoder_config = config.encoder_config
        if encoder_config is None:
            encoder_config = AutoConfig.from_pretrained(model_name, cache_dir=cache_dir)
            if config.vocab_size!=-1:
                encoder_config.vocab_size = config.vocab_size

        if config._attn_implementation is not None and not labels_encoder:
            encoder_config._attn_implementation = config._attn_implementation

        config_name = encoder_config.__class__.__name__

        kwargs = {}
        if config_name in DECODER_MODEL_MAPPING:
            if not IS_LLM2VEC:
                raise MissedPackageException(f"The llm2vec package must be installed to use this decoder model: {config_name}")
            else:
                print('Loading decoder model using LLM2Vec...')
                ModelClass = DECODER_MODEL_MAPPING[config_name]
            custom = True
        elif config_name in {'T5Config', 'MT5Config'}:
            custom = True
            ModelClass = T5EncoderModel
            if IS_TURBOT5:
                kwargs = {"attention_type": 'flash'}
        elif config_name in {'DebertaV2Config'}:
            custom = True
            ModelClass = DebertaV2Model
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
        pair_attention_mask = kwargs.pop("pair_attention_mask", None)
        base_attention_mask = kwargs.pop("attention_mask", None)

        args = list(args)
        input_ids = kwargs.get("input_ids")
        if input_ids is None and args:
            input_ids = args[0]
            args = args[1:]
        args = tuple(args)

        kwargs.setdefault("output_attentions", False)
        if self.config.fuse_layers:
            kwargs["output_hidden_states"] = True
        else:
            kwargs.setdefault("output_hidden_states", False)
        kwargs.setdefault("return_dict", True)

        if pair_attention_mask is not None:
            mask_info = self._prepare_pair_attention_masks(
                pair_attention_mask,
                base_attention_mask,
                input_ids,
                kwargs.get("inputs_embeds"),
            )

            model_kwargs = dict(kwargs)
            model_name = self.model.__class__.__name__

            if model_name in {"DebertaV2Model", "DebertaModel"}:
                output = self._forward_deberta(
                    input_ids=input_ids,
                    model_kwargs=model_kwargs,
                    mask_info=mask_info,
                )
            elif model_name == "ModernBertModel":
                output = self._forward_modernbert(
                    input_ids=input_ids,
                    model_kwargs=model_kwargs,
                    mask_info=mask_info,
                )
            elif model_name in {"T5EncoderModel", "MT5EncoderModel", "T5Model"}:
                output = self._forward_t5(
                    input_ids=input_ids,
                    model_kwargs=model_kwargs,
                    mask_info=mask_info,
                )
            else:
                model_kwargs.pop("packing_config", None)
                model_kwargs["attention_mask"] = mask_info["extended_mask"]
                output = self.model(*args, **model_kwargs)
        else:
            if base_attention_mask is not None:
                kwargs["attention_mask"] = base_attention_mask
            output = self.model(*args, **kwargs)

        if self.config.fuse_layers:
            encoder_layer = self.layers_fuser(output.hidden_states)
        else:
            encoder_layer = output[0]

        return encoder_layer

    def _get_model_dtype(self) -> torch.dtype:
        try:
            return next(self.model.parameters()).dtype
        except StopIteration:
            return torch.float32

    def _prepare_pair_attention_masks(
        self,
        pair_attention_mask: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        input_ids: Optional[torch.Tensor],
        inputs_embeds: Optional[torch.Tensor],
    ) -> dict:
        device = pair_attention_mask.device
        if input_ids is not None:
            device = input_ids.device
        elif inputs_embeds is not None:
            device = inputs_embeds.device

        pair_mask_bool = pair_attention_mask.to(device=device, dtype=torch.bool)

        token_mask_bool = pair_mask_bool.any(dim=-1)
        if attention_mask is not None:
            token_mask_bool = token_mask_bool & attention_mask.to(device=device, dtype=torch.bool)

        seq_len = pair_mask_bool.size(-1)
        if seq_len:
            identity = torch.eye(seq_len, device=device, dtype=torch.bool).unsqueeze(0)
            token_diag = token_mask_bool.unsqueeze(-1)
            pair_mask_bool = pair_mask_bool | (identity & token_diag)

        active = token_mask_bool.unsqueeze(-1) & token_mask_bool.unsqueeze(-2)
        pair_mask_bool = pair_mask_bool & active

        if attention_mask is not None:
            token_mask = token_mask_bool.to(attention_mask.dtype)
        else:
            token_mask = token_mask_bool.to(dtype=torch.float32)

        mask_dtype = self._get_model_dtype()
        neg_inf = torch.finfo(mask_dtype).min
        extended_mask = torch.zeros(
            pair_mask_bool.shape, dtype=mask_dtype, device=device
        ).masked_fill(~pair_mask_bool, neg_inf).unsqueeze(1)

        inactive = ~token_mask_bool
        if inactive.any():
            extended_mask = extended_mask.masked_fill(
                inactive.unsqueeze(1).unsqueeze(-1),
                torch.tensor(0.0, dtype=mask_dtype, device=device),
            )

        return {
            "token_mask": token_mask,
            "token_mask_bool": token_mask_bool,
            "extended_mask": extended_mask,
            "block_mask": pair_mask_bool,
        }

    def _forward_deberta(
        self,
        input_ids: Optional[torch.Tensor],
        model_kwargs: dict,
        mask_info: dict,
    ) -> BaseModelOutput:
        inputs_embeds = model_kwargs.pop("inputs_embeds", None)
        token_type_ids = model_kwargs.pop("token_type_ids", None)
        position_ids = model_kwargs.pop("position_ids", None)
        output_attentions = model_kwargs.pop("output_attentions")
        produce_hidden = model_kwargs.pop("output_hidden_states")
        return_dict = model_kwargs.pop("return_dict")

        if input_ids is None and inputs_embeds is None:
            raise ValueError("Either input_ids or inputs_embeds must be provided for packed attention")
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Cannot supply both input_ids and inputs_embeds")

        if token_type_ids is None:
            ref = inputs_embeds if inputs_embeds is not None else input_ids
            shape = ref.size()[:-1] if inputs_embeds is not None else ref.size()
            token_type_ids = torch.zeros(shape, dtype=torch.long, device=ref.device)

        embedding_output = self.model.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            mask=mask_info["token_mask"],
            inputs_embeds=inputs_embeds,
        )

        encoder_outputs = self.model.encoder(
            embedding_output,
            mask_info["block_mask"],
            output_hidden_states=True,
            output_attentions=output_attentions,
            return_dict=True,
        )

        encoded_layers = list(encoder_outputs.hidden_states)

        if getattr(self.model, "z_steps", 0) > 1:
            hidden_states = encoded_layers[-2]
            layers = [self.model.encoder.layer[-1] for _ in range(self.model.z_steps)]
            query_states = encoded_layers[-1]
            rel_embeddings = self.model.encoder.get_rel_embedding()
            attention_mask = self.model.encoder.get_attention_mask(mask_info["block_mask"])
            rel_pos = self.model.encoder.get_rel_pos(embedding_output)
            for layer in layers[1:]:
                query_states = layer(
                    hidden_states,
                    attention_mask,
                    output_attentions=False,
                    query_states=query_states,
                    relative_pos=rel_pos,
                    rel_embeddings=rel_embeddings,
                )
                encoded_layers.append(query_states)

        sequence_output = encoded_layers[-1]
        hidden_states_tuple = tuple(encoded_layers) if produce_hidden else None
        attentions = encoder_outputs.attentions if output_attentions else None

        if not return_dict:
            result = (sequence_output,)
            if hidden_states_tuple is not None:
                result += (hidden_states_tuple,)
            if attentions is not None:
                result += (attentions,)
            return result

        return BaseModelOutput(
            last_hidden_state=sequence_output,
            hidden_states=hidden_states_tuple,
            attentions=attentions,
        )

    def _forward_modernbert(
        self,
        input_ids: Optional[torch.Tensor],
        model_kwargs: dict,
        mask_info: dict,
    ) -> BaseModelOutput:
        inputs_embeds = model_kwargs.pop("inputs_embeds", None)
        position_ids = model_kwargs.pop("position_ids", None)
        indices = model_kwargs.pop("indices", None)
        cu_seqlens = model_kwargs.pop("cu_seqlens", None)
        max_seqlen = model_kwargs.pop("max_seqlen", None)
        batch_size = model_kwargs.pop("batch_size", None)
        seq_len = model_kwargs.pop("seq_len", None)
        output_attentions = model_kwargs.pop("output_attentions")
        output_hidden_states = model_kwargs.pop("output_hidden_states")
        return_dict = model_kwargs.pop("return_dict")

        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("ModernBERT requires exactly one of input_ids or inputs_embeds")

        token_mask_bool = mask_info["token_mask_bool"].to(torch.bool)
        if batch_size is None or seq_len is None:
            ref = inputs_embeds if inputs_embeds is not None else input_ids
            batch_size, seq_len = ref.shape[:2]
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if position_ids is None:
            position_ids = torch.arange(seq_len, device=device).unsqueeze(0)

        base_attention_mask = token_mask_bool

        original_impl = self.model.config._attn_implementation
        if original_impl == "flash_attention_2":
            self.model.config._attn_implementation = "eager"

        self.model._maybe_set_compile()

        global_attention_mask, sliding_window_mask = self.model._update_attention_mask(
            base_attention_mask,
            output_attentions=output_attentions,
        )

        block = mask_info["block_mask"].unsqueeze(1)
        neg_inf = torch.finfo(global_attention_mask.dtype).min
        global_attention_mask = global_attention_mask.masked_fill(~block, neg_inf)
        sliding_window_mask = sliding_window_mask.masked_fill(~block, neg_inf)

        hidden_states = self.model.embeddings(input_ids=input_ids, inputs_embeds=inputs_embeds)

        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for encoder_layer in self.model.layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = encoder_layer(
                hidden_states,
                attention_mask=global_attention_mask,
                sliding_window_mask=sliding_window_mask,
                position_ids=position_ids,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                output_attentions=output_attentions,
            )
            hidden_states = layer_outputs[0]
            if output_attentions and len(layer_outputs) > 1:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        hidden_states = self.model.final_norm(hidden_states)

        if original_impl == "flash_attention_2":
            self.model.config._attn_implementation = original_impl

        if not return_dict:
            return tuple(
                v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None
            )

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )

    def _forward_t5(
        self,
        input_ids: Optional[torch.Tensor],
        model_kwargs: dict,
        mask_info: dict,
    ) -> BaseModelOutput:
        stack = self.model.encoder

        kw_input_ids = model_kwargs.pop("input_ids", None)
        if input_ids is None:
            input_ids = kw_input_ids
        elif kw_input_ids is not None:
            input_ids = kw_input_ids

        inputs_embeds = model_kwargs.pop("inputs_embeds", None)
        head_mask = model_kwargs.pop("head_mask", None)
        past_key_values = model_kwargs.pop("past_key_values", None)
        use_cache = model_kwargs.pop("use_cache", stack.config.use_cache)
        output_attentions = model_kwargs.pop("output_attentions")
        output_hidden_states = model_kwargs.pop("output_hidden_states")
        return_dict = model_kwargs.pop("return_dict")
        cache_position = model_kwargs.pop("cache_position", None)

        if model_kwargs:
            raise ValueError(f"Unsupported kwargs for T5 forward: {list(model_kwargs.keys())}")

        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("Either input_ids or inputs_embeds must be provided")
            inputs_embeds = stack.embed_tokens(input_ids)
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        batch_size, seq_length = input_shape
        device = inputs_embeds.device

        if cache_position is None:
            cache_position = torch.arange(seq_length, device=device)

        block_mask = mask_info["block_mask"].to(device=device, dtype=torch.bool)

        dtype = inputs_embeds.dtype
        neg_inf = torch.finfo(dtype).min
        causal_mask = torch.zeros(block_mask.shape, dtype=dtype, device=device)
        causal_mask = causal_mask.masked_fill(~block_mask, neg_inf).unsqueeze(1)

        head_mask = stack.get_head_mask(head_mask, stack.config.num_layers)

        hidden_states = stack.dropout(inputs_embeds)

        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        position_bias = None

        for idx, layer_module in enumerate(stack.block):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[idx] if head_mask is not None else None

            layer_outputs = layer_module(
                hidden_states,
                attention_mask=causal_mask,
                position_bias=position_bias,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                encoder_decoder_position_bias=None,
                layer_head_mask=layer_head_mask,
                cross_attn_layer_head_mask=None,
                past_key_values=None if not use_cache else past_key_values,
                use_cache=False,
                output_attentions=output_attentions,
                return_dict=True,
                cache_position=cache_position,
            )

            hidden_states = layer_outputs[0]
            position_bias = layer_outputs[1]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[2],)

        hidden_states = stack.final_layer_norm(hidden_states)
        hidden_states = stack.dropout(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            result = (hidden_states,)
            if output_hidden_states:
                result += (all_hidden_states,)
            if output_attentions:
                result += (all_attentions,)
            return result

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )
    
class Encoder(nn.Module):
    def __init__(self, config, from_pretrained: bool = False, cache_dir: Optional[Union[str, Path]]= None):
        super().__init__()

        self.bert_layer = Transformer( #transformer_model
            config.model_name, config, from_pretrained, cache_dir = cache_dir
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
        packing_config: Optional[InferencePackingConfig] = kwargs.pop("packing_config", None)
        pair_attention_mask = kwargs.pop("pair_attention_mask", None)

        if (
            packing_config is not None
            and not self.training
            and isinstance(input_ids, torch.Tensor)
            and isinstance(attention_mask, torch.Tensor)
            and input_ids.dim() == 2
        ):
            token_embeddings = self._encode_with_packing(
                input_ids,
                attention_mask,
                packing_config,
                pair_attention_mask,
                *args,
                **kwargs,
            )
        else:
            bert_kwargs = dict(kwargs)
            if attention_mask is not None:
                bert_kwargs["attention_mask"] = attention_mask
            if pair_attention_mask is not None:
                bert_kwargs["pair_attention_mask"] = pair_attention_mask
            token_embeddings = self.bert_layer(
                input_ids=input_ids,
                **bert_kwargs,
            )

        if hasattr(self, "projection"):
            token_embeddings = self.projection(token_embeddings)
        return token_embeddings

    def _encode_with_packing(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        packing_config: InferencePackingConfig,
        pair_attention_mask: Optional[torch.Tensor],
        *args,
        **kwargs,
    ) -> torch.Tensor:
        lengths = attention_mask.sum(dim=-1).tolist()
        seq_len = int(input_ids.size(1))
        if not lengths or all(int(l) == seq_len for l in lengths):
            bert_kwargs = dict(kwargs)
            bert_kwargs["attention_mask"] = attention_mask
            if pair_attention_mask is not None:
                bert_kwargs["pair_attention_mask"] = pair_attention_mask
            return self.bert_layer(input_ids=input_ids, **bert_kwargs)

        requests = []
        for row, length in zip(input_ids, lengths):
            length = int(length)
            if length <= 0:
                requests.append({"input_ids": []})
            else:
                requests.append({"input_ids": row[:length].tolist()})

        pad_token_id = self.bert_layer.model.config.pad_token_id
        if pad_token_id is None:
            pad_token_id = 0

        packed = pack_requests(requests, packing_config, pad_token_id)

        device = input_ids.device
        packed_ids = packed.input_ids.to(device=device)
        packed_mask = packed.pair_attention_mask.to(device=device)
        packed_fallback = packed.attention_mask.to(device=device)

        attn_to_use = packed_mask if packed_mask.numel() else packed_fallback
        bert_kwargs = dict(kwargs)

        if packed_mask.numel():
            bert_kwargs["attention_mask"] = packed_fallback
            bert_kwargs["pair_attention_mask"] = packed_mask
        else:
            bert_kwargs["attention_mask"] = attn_to_use

        token_embeddings = self.bert_layer(
            input_ids=packed_ids,
            **bert_kwargs,
        )

        unpacked: List[torch.Tensor] = unpack_spans(token_embeddings, packed)
        hidden_size = token_embeddings.size(-1)
        batch, seq = input_ids.size()
        output = token_embeddings.new_zeros(batch, seq, hidden_size)
        for idx, target in enumerate(unpacked):
            tgt_len = int(target.size(0))
            if tgt_len == 0:
                continue
            output[idx, :tgt_len] = target
        return output
    
    def forward(self, *args, **kwargs) -> torch.Tensor:
        token_embeddings = self.encode_text(*args, **kwargs)
        return token_embeddings

class BiEncoder(Encoder):
    def __init__(self, config, from_pretrained: bool = False, cache_dir:Optional[Union[str, Path]] = None):
        super().__init__(config, from_pretrained)
        if config.labels_encoder is not None:
            self.labels_encoder = Transformer( #transformer_model
                config.labels_encoder, config, from_pretrained, True, cache_dir=cache_dir
            )
            le_hidden_size = self.labels_encoder.model.config.hidden_size

            if config.hidden_size != le_hidden_size:
                self.labels_projection = nn.Linear(le_hidden_size, config.hidden_size)
    
    def mean_pooling(self, token_embeddings, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def encode_labels(self, input_ids, attention_mask, *args, **kwargs):
        label_kwargs = dict(kwargs)
        label_kwargs.pop("packing_config", None)
        label_kwargs.pop("pair_attention_mask", None)
        labels_embeddings = self.labels_encoder(input_ids, attention_mask, *args, **label_kwargs)
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
