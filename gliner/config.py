from typing import Optional
from transformers import PretrainedConfig
from transformers.models.auto import CONFIG_MAPPING


class BaseGLiNERConfig(PretrainedConfig):
    """Base configuration class for all GLiNER models."""
    model_type = "gliner"
    is_composition = True
    
    def __init__(self, 
                 model_name: str = "microsoft/deberta-v3-small",
                 name: str = "gliner",
                 max_width: int = 12,
                 hidden_size: int = 512,
                 dropout: float = 0.4,
                 fine_tune: bool = True,
                 subtoken_pooling: str = "first",
                 span_mode: str = "markerV0",
                 post_fusion_schema: str = '',
                 num_post_fusion_layers: int = 1, 
                 vocab_size: int = -1,
                 max_neg_type_ratio: int = 1,
                 max_types: int = 25,
                 max_len: int = 384,
                 words_splitter_type: str = "whitespace",
                 has_rnn: bool = True,
                 fuse_layers: bool = False,
                 embed_ent_token: bool = True,
                 class_token_index: int = -1,
                 encoder_config: Optional[dict] = None,
                 ent_token: str = "<<ENT>>",
                 sep_token: str = "<<SEP>>",
                 _attn_implementation: str = None,
                 **kwargs):
        super().__init__(**kwargs)
        
        if isinstance(encoder_config, dict):
            encoder_config["model_type"] = (encoder_config["model_type"] 
                                            if "model_type" in encoder_config 
                                            else "deberta-v2")
            encoder_config = CONFIG_MAPPING[encoder_config["model_type"]](**encoder_config)
        self.encoder_config = encoder_config
        
        self.model_name = model_name
        self.name = name
        self.max_width = max_width
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.fine_tune = fine_tune
        self.subtoken_pooling = subtoken_pooling
        self.span_mode = span_mode
        self.post_fusion_schema = post_fusion_schema
        self.num_post_fusion_layers = num_post_fusion_layers
        self.vocab_size = vocab_size
        self.max_neg_type_ratio = max_neg_type_ratio
        self.max_types = max_types
        self.max_len = max_len
        self.words_splitter_type = words_splitter_type
        self.has_rnn = has_rnn
        self.fuse_layers = fuse_layers
        self.class_token_index = class_token_index
        self.embed_ent_token = embed_ent_token
        self.ent_token = ent_token
        self.sep_token = sep_token
        self._attn_implementation = _attn_implementation


class UniEncoderConfig(BaseGLiNERConfig):
    """Base configuration for uni-encoder GLiNER models."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class UniEncoderSpanConfig(UniEncoderConfig):
    """Configuration for uni-encoder span-based GLiNER model."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.span_mode == 'token-level':
            raise ValueError("UniEncoderSpanConfig requires span_mode != 'token-level'")
    
    @property
    def model_type(self):
        return "uni-encoder-span"


class UniEncoderTokenConfig(UniEncoderConfig):
    """Configuration for uni-encoder token-based GLiNER model."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.span_mode = 'token-level'
    
    @property
    def model_type(self):
        return "uni-encoder-token"


class UniEncoderSpanDecoderConfig(UniEncoderConfig):
    """Configuration for uni-encoder span model with decoder for label generation."""
    
    def __init__(self, 
                 labels_decoder: str = None,
                 decoder_mode: str = None,
                 full_decoder_context: bool = True,
                 labels_decoder_config: Optional[dict] = None,
                 **kwargs):
        super().__init__(**kwargs)
        
        if isinstance(labels_decoder_config, dict):
            labels_decoder_config["model_type"] = (labels_decoder_config["model_type"] 
                                                   if "model_type" in labels_decoder_config 
                                                   else "gpt2")
            labels_decoder_config = CONFIG_MAPPING[labels_decoder_config["model_type"]](**labels_decoder_config)
        self.labels_decoder_config = labels_decoder_config
        
        self.labels_decoder = labels_decoder
        self.decoder_mode = decoder_mode  # 'prompt' or 'span'
        self.full_decoder_context = full_decoder_context
        
        if self.span_mode == 'token-level':
            raise ValueError("UniEncoderSpanDecoderConfig requires span_mode != 'token-level'")
    
    @property
    def model_type(self):
        return "encoder-decoder"


class UniEncoderSpanRelexConfig(UniEncoderConfig):
    """Configuration for uni-encoder span model with relation extraction."""
    
    def __init__(self, 
                 relations_layer: str = None,
                 triples_layer: str = None,
                 embed_rel_token: bool = True,
                 rel_token_index: int = -1,
                 rel_token: str = "<<REL>>",
                 **kwargs):
        super().__init__(**kwargs)
        
        self.relations_layer = relations_layer
        self.triples_layer = triples_layer
        self.embed_rel_token = embed_rel_token
        self.rel_token_index = rel_token_index
        self.rel_token = rel_token
        
        if self.span_mode == 'token-level':
            raise ValueError("UniEncoderSpanRelexConfig requires span_mode != 'token-level'")
    
    @property
    def model_type(self):
        return "uni-encoder-span-relex"


class BiEncoderConfig(BaseGLiNERConfig):
    """Base configuration for bi-encoder GLiNER models."""
    
    def __init__(self, 
                 labels_encoder: str = None,
                 labels_encoder_config: Optional[dict] = None,
                 **kwargs):
        super().__init__(**kwargs)
        
        if isinstance(labels_encoder_config, dict):
            labels_encoder_config["model_type"] = (labels_encoder_config["model_type"] 
                                                   if "model_type" in labels_encoder_config 
                                                   else "deberta-v2")
            labels_encoder_config = CONFIG_MAPPING[labels_encoder_config["model_type"]](**labels_encoder_config)
        self.labels_encoder_config = labels_encoder_config
        
        self.labels_encoder = labels_encoder


class BiEncoderSpanConfig(BiEncoderConfig):
    """Configuration for bi-encoder span-based GLiNER model."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.span_mode == 'token-level':
            raise ValueError("BiEncoderSpanConfig requires span_mode != 'token-level'")
    
    @property
    def model_type(self):
        return "bi-encoder-span"


class BiEncoderTokenConfig(BiEncoderConfig):
    """Configuration for bi-encoder token-based GLiNER model."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.span_mode = 'token-level'
    
    @property
    def model_type(self):
        return "bi-encoder-token"


# Legacy GLiNERConfig for backward compatibility
class GLiNERConfig(BaseGLiNERConfig):
    """Legacy configuration class that auto-detects model type."""
    
    def __init__(self, 
                 labels_encoder: str = None,
                 labels_decoder: str = None,
                 decoder_mode: str = None,
                 full_decoder_context: bool = True,
                 labels_encoder_config: Optional[dict] = None,
                 labels_decoder_config: Optional[dict] = None,
                 relations_layer: str = None,
                 triples_layer: str = None,
                 embed_rel_token: bool = True,
                 rel_token_index: int = -1,
                 rel_token: str = "<<REL>>",
                 **kwargs):
        super().__init__(**kwargs)
        
        # Labels encoder config
        if isinstance(labels_encoder_config, dict):
            labels_encoder_config["model_type"] = (labels_encoder_config["model_type"] 
                                                   if "model_type" in labels_encoder_config 
                                                   else "deberta-v2")
            labels_encoder_config = CONFIG_MAPPING[labels_encoder_config["model_type"]](**labels_encoder_config)
        self.labels_encoder_config = labels_encoder_config
        
        # Labels decoder config
        if isinstance(labels_decoder_config, dict):
            labels_decoder_config["model_type"] = (labels_decoder_config["model_type"] 
                                                   if "model_type" in labels_decoder_config 
                                                   else "gpt2")
            labels_decoder_config = CONFIG_MAPPING[labels_decoder_config["model_type"]](**labels_decoder_config)
        self.labels_decoder_config = labels_decoder_config
        
        self.labels_encoder = labels_encoder
        self.labels_decoder = labels_decoder
        self.decoder_mode = decoder_mode
        self.full_decoder_context = full_decoder_context
        
        # Relation extraction config
        self.relations_layer = relations_layer
        self.triples_layer = triples_layer
        self.embed_rel_token = embed_rel_token
        self.rel_token_index = rel_token_index
        self.rel_token = rel_token
    
    @property
    def model_type(self):
        """Auto-detect model type based on configuration."""
        if self.labels_decoder:
            return "encoder-decoder"
        elif self.labels_encoder:
            return 'bi-encoder'
        elif self.span_mode == 'token-level':
            return 'uni-encoder-token'
        else:
            return 'uni-encoder-span'


# Register all configurations
CONFIG_MAPPING.update({
    "gliner": GLiNERConfig,
    "gliner_base": BaseGLiNERConfig,
    "gliner_uni_encoder": UniEncoderConfig,
    "gliner_uni_encoder_span": UniEncoderSpanConfig,
    "gliner_uni_encoder_token": UniEncoderTokenConfig,
    "gliner_uni_encoder_span_decoder": UniEncoderSpanDecoderConfig,
    "gliner_uni_encoder_span_relex": UniEncoderSpanRelexConfig,
    "gliner_bi_encoder": BiEncoderConfig,
    "gliner_bi_encoder_span": BiEncoderSpanConfig,
    "gliner_bi_encoder_token": BiEncoderTokenConfig,
})