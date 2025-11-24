from typing import Optional

from transformers import PretrainedConfig
from transformers.models.auto import CONFIG_MAPPING


class BaseGLiNERConfig(PretrainedConfig):
    """Base configuration class for all GLiNER models."""

    is_composition = True
    model_type = None

    def __init__(
        self,
        model_name: str = "microsoft/deberta-v3-small",
        name: str = "gliner",
        max_width: int = 12,
        hidden_size: int = 512,
        dropout: float = 0.4,
        fine_tune: bool = True,
        subtoken_pooling: str = "first",
        span_mode: str = "markerV0",
        post_fusion_schema: str = "",
        num_post_fusion_layers: int = 1,
        vocab_size: int = -1,
        max_neg_type_ratio: int = 1,
        max_types: int = 25,
        max_len: int = 384,
        words_splitter_type: str = "whitespace",
        num_rnn_layers: int = 1,
        fuse_layers: bool = False,
        embed_ent_token: bool = True,
        class_token_index: int = -1,
        encoder_config: Optional[dict] = None,
        ent_token: str = "<<ENT>>",
        sep_token: str = "<<SEP>>",
        _attn_implementation: Optional[str] = None,
        **kwargs,
    ):
        """Initialize BaseGLiNERConfig.

        Args:
            model_name (str, optional): Name of the pretrained encoder model.
                Defaults to "microsoft/deberta-v3-small".
            name (str, optional): Name identifier for the GLiNER model. Defaults to "gliner".
            max_width (int, optional): Maximum span width for entity detection. Defaults to 12.
            hidden_size (int, optional): Dimension of hidden representations. Defaults to 512.
            dropout (float, optional): Dropout probability. Defaults to 0.4.
            fine_tune (bool, optional): Whether to fine-tune the encoder. Defaults to True.
            subtoken_pooling (str, optional): Subtoken pooling strategy. Defaults to "first".
            span_mode (str, optional): Span representation mode. Defaults to "markerV0".
            post_fusion_schema (str, optional): Post-fusion processing schema. Defaults to ''.
            num_post_fusion_layers (int, optional): Number of post-fusion layers. Defaults to 1.
            vocab_size (int, optional): Vocabulary size. Defaults to -1.
            max_neg_type_ratio (int, optional): Max ratio of negative to positive types. Defaults to 1.
            max_types (int, optional): Maximum number of entity types. Defaults to 25.
            max_len (int, optional): Maximum sequence length. Defaults to 384.
            words_splitter_type (str, optional): Word splitter type. Defaults to "whitespace".
            num_rnn_layers (int, optional): Number of LSTM layers, if less then 1, then LSTM is not used.
            fuse_layers (bool, optional): Whether to fuse layers. Defaults to False.
            embed_ent_token (bool, optional): Whether to embed entity tokens. Defaults to True.
            class_token_index (int, optional): Index of class token. Defaults to -1.
            encoder_config (dict, optional): Encoder configuration dict. Defaults to None.
            ent_token (str, optional): Entity marker token. Defaults to "<<ENT>>".
            sep_token (str, optional): Separator token. Defaults to "<<SEP>>".
            _attn_implementation (str, optional): Attention implementation. Defaults to None.
            **kwargs: Additional keyword arguments passed to parent class.
        """
        super().__init__(**kwargs)

        if isinstance(encoder_config, dict):
            encoder_config["model_type"] = encoder_config.get("model_type", "deberta-v2")

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
        self.num_rnn_layers = num_rnn_layers
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
        if self.span_mode == "token_level":
            raise ValueError("UniEncoderSpanConfig requires span_mode != 'token_level'")

        self.model_type = "gliner_uni_encoder_span"


class UniEncoderTokenConfig(UniEncoderConfig):
    """Configuration for uni-encoder token-based GLiNER model."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.span_mode = "token_level"
        self.model_type = "gliner_uni_encoder_token"


class UniEncoderSpanDecoderConfig(UniEncoderConfig):
    """Configuration for uni-encoder span model with decoder for label generation."""

    def __init__(
        self,
        labels_decoder: Optional[str] = None,
        decoder_mode: Optional[str] = None,
        full_decoder_context: bool = True,
        blank_entity_prob: float = 0.1,
        labels_decoder_config: Optional[dict] = None,
        decoder_loss_coef=0.5,
        span_loss_coef=0.5,
        **kwargs,
    ):
        """Initialize UniEncoderSpanDecoderConfig.

        Args:
            labels_decoder (str, optional): Name/path of the decoder model. Defaults to None.
            decoder_mode (str, optional): Mode for decoder ('prompt' or 'span'). Defaults to None.
            full_decoder_context (bool, optional): Use full context in decoder. Defaults to True.
            blank_entity_prob (float, optional): Probability of blank entities. Defaults to 0.1.
            labels_decoder_config (dict, optional): Decoder config dict. Defaults to None.
            decoder_loss_coef (float, optional): Decoder loss coefficient. Defaults to 0.5.
            span_loss_coef (float, optional): Span loss coefficient. Defaults to 0.5.
            **kwargs: Additional keyword arguments passed to UniEncoderConfig.

        Raises:
            ValueError: If span_mode is 'token-level', which is incompatible with this config.
        """
        super().__init__(**kwargs)

        if isinstance(labels_decoder_config, dict):
            labels_decoder_config["model_type"] = labels_decoder_config.get("model_type", "gpt2")

            labels_decoder_config = CONFIG_MAPPING[labels_decoder_config["model_type"]](**labels_decoder_config)
        self.labels_decoder_config = labels_decoder_config
        self.blank_entity_prob = blank_entity_prob
        self.labels_decoder = labels_decoder
        self.decoder_mode = decoder_mode  # 'prompt' or 'span'
        self.full_decoder_context = full_decoder_context
        self.decoder_loss_coef = decoder_loss_coef
        self.span_loss_coef = span_loss_coef
        self.model_type = "gliner_uni_encoder_span_decoder"
        if self.span_mode == "token_level":
            raise ValueError("UniEncoderSpanDecoderConfig requires span_mode != 'token_level'")


class UniEncoderSpanRelexConfig(UniEncoderConfig):
    """Configuration for uni-encoder span model with relation extraction."""

    def __init__(
        self,
        relations_layer: Optional[str] = None,
        triples_layer: Optional[str] = None,
        embed_rel_token: bool = True,
        rel_token_index: int = -1,
        rel_token: str = "<<REL>>",
        span_loss_coef=1.0,
        adjacency_loss_coef=1.0,
        relation_loss_coef=1.0,
        **kwargs,
    ):
        """Initialize UniEncoderSpanRelexConfig.

        Args:
            relations_layer (str, optional): Name of relations layer,
                see gliner.modeling.multitask.relations_layers.py. Defaults to None.
            triples_layer (str, optional): Name of triples layer,
                see gliner.modeling.multitask.triples_layers.py. Defaults to None.
            embed_rel_token (bool, optional): Whether to embed relation tokens. Defaults to True.
            rel_token_index (int, optional): Index of relation token. Defaults to -1.
            rel_token (str, optional): Relation marker token. Defaults to "<<REL>>".
            span_loss_coef (float, optional): Span representaton loss coefficient. Defaults to 1.0.
            adjacency_loss_coef (float, optional): Adjacency modeling loss coefficient. Defaults to 1.0.
            relation_loss_coef (float, optional): Relation representaton loss coefficient. Defaults to 1.0.
            **kwargs: Additional keyword arguments passed to UniEncoderConfig.

        Raises:
            ValueError: If span_mode is 'token_level', which is incompatible with this config.
        """
        super().__init__(**kwargs)

        self.relations_layer = relations_layer
        self.triples_layer = triples_layer
        self.embed_rel_token = embed_rel_token
        self.rel_token_index = rel_token_index
        self.rel_token = rel_token
        self.span_loss_coef = span_loss_coef
        self.adjacency_loss_coef = adjacency_loss_coef
        self.relation_loss_coef = relation_loss_coef
        self.model_type = "gliner_uni_encoder_span_relex"
        if self.span_mode == "token_level":
            raise ValueError("UniEncoderSpanRelexConfig requires span_mode != 'token_level'")


class BiEncoderConfig(BaseGLiNERConfig):
    """Base configuration for bi-encoder GLiNER models."""

    def __init__(self, labels_encoder: Optional[str] = None, labels_encoder_config: Optional[dict] = None, **kwargs):
        """Initialize BiEncoderConfig.

        Args:
            labels_encoder (str, optional): Name/path of labels encoder model. Defaults to None.
            labels_encoder_config (dict, optional): Labels encoder config dict. Defaults to None.
            **kwargs: Additional keyword arguments passed to BaseGLiNERConfig.
        """
        super().__init__(**kwargs)

        if isinstance(labels_encoder_config, dict):
            labels_encoder_config["model_type"] = labels_encoder_config.get("model_type", "deberta-v2")

            labels_encoder_config = CONFIG_MAPPING[labels_encoder_config["model_type"]](**labels_encoder_config)
        self.labels_encoder_config = labels_encoder_config

        self.labels_encoder = labels_encoder


class BiEncoderSpanConfig(BiEncoderConfig):
    """Configuration for bi-encoder span-based GLiNER model."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.span_mode == "token_level":
            raise ValueError("BiEncoderSpanConfig requires span_mode != 'token_level'")
        self.model_type = "gliner_bi_encoder_span"


class BiEncoderTokenConfig(BiEncoderConfig):
    """Configuration for bi-encoder token-based GLiNER model."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.span_mode = "token_level"
        self.model_type = "gliner_bi_encoder_token"


class GLiNERConfig(BaseGLiNERConfig):
    """Legacy configuration class that auto-detects model type.

    This class provides backward compatibility by automatically determining the
    appropriate model type based on the provided configuration parameters.

    Attributes:
        labels_encoder (str): Name of the encoder for entity labels (bi-encoder).
        labels_decoder (str): Name of the decoder for label generation.
        relations_layer (str): Layer configuration for relation extraction.
    """

    def __init__(
        self,
        labels_encoder: Optional[str] = None,
        labels_decoder: Optional[str] = None,
        relations_layer: Optional[str] = None,
        **kwargs,
    ):
        """Initialize GLiNERConfig.

        Args:
            labels_encoder (str, optional): Labels encoder for bi-encoder models. Defaults to None.
            labels_decoder (str, optional): Decoder for label generation. Defaults to None.
            relations_layer (str, optional): Relations layer for relation extraction. Defaults to None.
            **kwargs: Additional keyword arguments passed to BaseGLiNERConfig.
        """
        super().__init__(**kwargs)

        self.labels_encoder = labels_encoder
        self.labels_decoder = labels_decoder
        self.relations_layer = relations_layer

    @property
    def model_type(self):
        """Auto-detect model type based on configuration."""
        if self.labels_decoder:
            return "gliner_uni_encoder_span_decoder"
        elif self.labels_encoder:
            return "gliner_bi_encoder_span" if self.span_mode != "token-level" else "gliner_bi_encoder_token"
        elif self.relations_layer is not None:
            return "gliner_uni_encoder_span_relex"
        elif self.span_mode == "token-level":
            return "gliner_uni_encoder_token"
        else:
            return "gliner_uni_encoder_span"


# Register all configurations
CONFIG_MAPPING.update(
    {
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
    }
)
