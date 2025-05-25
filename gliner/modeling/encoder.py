import warnings
from pathlib import Path
from typing import Optional, Union, Tuple

import torch
from torch import nn
from transformers import AutoModel, AutoConfig

from .layers import LayersFuser
from ..utils import is_module_available, MissedPackageException
from ..multimodal.layers import SpatialEmbeddings, PatchEmbeddings

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
    
class VisionTransformer(nn.Module):
    def __init__(
        self, 
        model_name, 
        config, 
        from_pretrained=False, 
        cache_dir:Optional[Union[str, Path]] = None
    ):
        super().__init__()
        encoder_config = config.vision_encoder_config
        if encoder_config is None:
            encoder_config = AutoConfig.from_pretrained(model_name, cache_dir=cache_dir)

        config_name = encoder_config.__class__.__name__

        kwargs = {}
        if config_name in {'CLIPConfig'}:
            from transformers import CLIPVisionModel
            ModelClass = CLIPVisionModel
        elif config_name in {'Owlv2Config'}:
            from transformers import Owlv2VisionModel
            ModelClass = Owlv2VisionModel
        #TODO:
        # elif config_name in {'CustomConfig'}:
        #     ModelClass = PathEmbeddings
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

        if config.fuse_layers:
            self.layers_fuser = LayersFuser(encoder_config.num_hidden_layers,
                                                        encoder_config.hidden_size)

        config.vision_encoder_config = encoder_config

        self.config = config

    def forward(self, pixel_values, *args, **kwargs):
        if "output_hidden_states" not in kwargs:
            if self.config.fuse_layers:
                output_hidden_states = True
            else:
                output_hidden_states = False
            return_hidden_states = False
        else:
            output_hidden_states = kwargs.pop('output_hidden_states')
            return_hidden_states = True
        output = self.model(pixel_values, *args, output_hidden_states = output_hidden_states, 
                                            return_dict = True,  **kwargs)
        if return_hidden_states:
            return output.hidden_states
        else:
            if self.config.fuse_layers:
                encoder_layer = self.layers_fuser(output.hidden_states)
            else:
                encoder_layer = output[0]

            return encoder_layer
    
class Encoder(nn.Module):
    def __init__(self, config, from_pretrained: bool = False, cache_dir: Optional[Union[str, Path]]= None):
        super().__init__()
        self.config = config

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
        token_embeddings = self.bert_layer(input_ids, attention_mask, *args, **kwargs)
        if hasattr(self, "projection"):
            token_embeddings = self.projection(token_embeddings)
        return token_embeddings
    
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


class LayoutVLEncoder(Encoder):
    def __init__(self, config, from_pretrained: bool = False, cache_dir:Optional[Union[str, Path]] = None):
        super().__init__(config, from_pretrained)
        if config.vision_encoder is not None:
            self.vision_encoder = VisionTransformer( #transformer_model
                config.vision_encoder, config, from_pretrained, cache_dir=cache_dir
            )
            ve_hidden_size = self.vision_encoder.model.config.hidden_size
            bert_hidden_size = self.bert_layer.model.config.hidden_size

            if bert_hidden_size != ve_hidden_size:
                self.vision_projection = nn.Linear(ve_hidden_size, bert_hidden_size)
        self.spatial_embeddings = SpatialEmbeddings(config)
        
        self.pad_token_id = self.config.encoder_config.pad_token_id

    def mean_pooling(self, token_embeddings, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def encode_image(self, pixel_values, *args, **kwargs):
        vision_embeddings = self.vision_encoder(pixel_values, *args, **kwargs)
        if hasattr(self, "vision_projection"):
            vision_embeddings = self.vision_projection(vision_embeddings)
        return vision_embeddings

    def _merge_input_ids_with_image_features(self, image_features, inputs_embeds, 
                                                        input_ids, attention_mask, 
                                                                bbox, visual_bbox):
        # return inputs_embeds, attention_mask, None
        num_images, num_image_patches, embed_dim = image_features.shape
        batch_size, sequence_length = input_ids.shape
        left_padding = not torch.sum(input_ids[:, -1] == torch.tensor(self.pad_token_id))
        # 1. Create a mask to know where special image tokens are
        special_image_token_mask = input_ids == self.config.image_token_index
        num_special_image_tokens = torch.sum(special_image_token_mask, dim=-1)
        # Compute the maximum embed dimension
        max_embed_dim = (num_special_image_tokens.max() * (num_image_patches - 1)) + sequence_length
        batch_indices, non_image_indices = torch.where(input_ids != self.config.image_token_index)

        # 2. Compute the positions where text should be written
        # Calculate new positions for text tokens in merged image-text sequence.
        # `special_image_token_mask` identifies image tokens. Each image token will be replaced by `nb_text_tokens_per_images - 1` text tokens.
        # `torch.cumsum` computes how each image token shifts subsequent text token positions.
        # - 1 to adjust for zero-based indexing, as `cumsum` inherently increases indices by one.
        new_token_positions = torch.cumsum((special_image_token_mask * (num_image_patches - 1) + 1), -1) - 1
        nb_image_pad = max_embed_dim - 1 - new_token_positions[:, -1]
        if left_padding:
            new_token_positions += nb_image_pad[:, None]  # offset for left padding
        text_to_overwrite = new_token_positions[batch_indices, non_image_indices]

        # 3. Create the full embedding, already padded to the maximum position
        final_embedding = torch.zeros(
            batch_size, max_embed_dim, embed_dim, dtype=inputs_embeds.dtype, device=inputs_embeds.device
        )
        final_attention_mask = torch.zeros(
            batch_size, max_embed_dim, dtype=attention_mask.dtype, device=inputs_embeds.device
        )

        final_bbox = torch.zeros(
                    batch_size, max_embed_dim, 4, dtype=attention_mask.dtype, device=inputs_embeds.device
                )
        
        # In case the Vision model or the Language model has been offloaded to CPU, we need to manually
        # set the corresponding tensors into their correct target device.
        target_device = inputs_embeds.device
        batch_indices, non_image_indices, text_to_overwrite = (
            batch_indices.to(target_device),
            non_image_indices.to(target_device),
            text_to_overwrite.to(target_device),
        )
        attention_mask = attention_mask.to(target_device)

        # 4. Fill the embeddings based on the mask. If we have ["hey" "<image>", "how", "are"]
        # we need to index copy on [0, 577, 578, 579] for the text and [1:576] for the image features
        final_embedding[batch_indices, text_to_overwrite] = inputs_embeds[batch_indices, non_image_indices]
        final_attention_mask[batch_indices, text_to_overwrite] = attention_mask[batch_indices, non_image_indices]

        # 5. Fill the embeddings corresponding to the images. Anything that is still zeros needs filling
        image_to_overwrite = torch.all(final_embedding == 0, dim=-1)
        image_to_overwrite &= image_to_overwrite.cumsum(-1) - 1 >= nb_image_pad[:, None].to(target_device)

        if image_to_overwrite.sum() != image_features.shape[:-1].numel():
            raise ValueError(
                f"The input provided to the model are wrong. The number of image tokens is {torch.sum(special_image_token_mask)} while"
                f" the number of image given to the model is {num_images}. This prevents correct indexing and breaks batch generation."
            )

        final_embedding[image_to_overwrite] = image_features.contiguous().reshape(-1, embed_dim).to(target_device)
        final_attention_mask |= image_to_overwrite

        if bbox is not None:

            final_bbox[batch_indices, text_to_overwrite] = bbox[batch_indices, non_image_indices]

            if visual_bbox is not None:
                 final_bbox[image_to_overwrite] = visual_bbox.contiguous().reshape(-1, 4).to(target_device)

        # 6. Mask out the embedding at padding positions, as we later use the past_key_value value to determine the non-attended tokens.
        batch_indices, pad_indices = torch.where(input_ids == self.pad_token_id)
        indices_to_mask = new_token_positions[batch_indices, pad_indices]

        final_embedding[batch_indices, indices_to_mask] = 0
        
        return final_embedding, final_attention_mask, final_bbox
    
    def prepare_inputs_embeds(self, input_ids, attention_mask, pixel_values, image_outputs, vision_feature_layer, 
                                    vision_feature_select_strategy, bbox, visual_bbox, **kwargs):
        inputs_embeds = self.get_input_embeddings()(input_ids)

        if pixel_values is not None and input_ids.shape[1] != 1:
            if self.config.use_patch_embeddings:
                selected_image_feature = self.vision_encoder(pixel_values)
            else:
                vision_hidden_states = self.vision_encoder(pixel_values, output_hidden_states = True)
                selected_image_feature = vision_hidden_states[vision_feature_layer]

            if vision_feature_select_strategy == "default":
                selected_image_feature = selected_image_feature[:, 1:]
            elif vision_feature_select_strategy == "full":
                selected_image_feature = selected_image_feature
            else:
                raise ValueError(
                    f"Unexpected select feature strategy: {self.config.vision_feature_select_strategy}"
                )

            image_features = self.vision_projection(selected_image_feature)

            inputs_embeds, attention_mask, bbox = self._merge_input_ids_with_image_features(
                image_features, inputs_embeds, input_ids, attention_mask, bbox, visual_bbox
            )

        if hasattr(self, "spatial_embeddings"):
            spatial_embeddings_ = self.spatial_embeddings(bbox)
            inputs_embeds+=spatial_embeddings_

        return inputs_embeds, attention_mask, bbox, image_outputs
    
    def forward(self, input_ids: torch.LongTensor = None,
                        pixel_values: torch.FloatTensor = None,
                        attention_mask: Optional[torch.Tensor] = None,
                        bbox: Optional[torch.LongTensor] = None,
                        images_bbox: Optional[torch.LongTensor] = None,
                        image_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
                        inputs_embeds: Optional[torch.FloatTensor] = None,
                        vision_feature_layer: Optional[int] = None,
                        vision_feature_select_strategy: Optional[str] = None,
                        **kwargs):

        vision_feature_layer = (
            vision_feature_layer if vision_feature_layer is not None else self.config.vision_feature_layer
        )
        vision_feature_select_strategy = (
            vision_feature_select_strategy
            if vision_feature_select_strategy is not None
            else self.config.vision_feature_select_strategy
        )

        if attention_mask is None:
            attention_mask = torch.ones(input_ids.shape[:2], dtype=input_ids.dtype, device=input_ids.device)

        if inputs_embeds is None:
            inputs_embeds, attention_mask, bbox, image_outputs = self.prepare_inputs_embeds(input_ids, attention_mask, pixel_values, image_outputs, vision_feature_layer, 
                                    vision_feature_select_strategy, bbox, images_bbox)

        token_embeddings = self.bert_layer(
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                **kwargs)
        
        if hasattr(self, "projection"):
            token_embeddings = self.projection(token_embeddings)
        
        return token_embeddings, attention_mask