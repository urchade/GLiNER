from typing import Union, Optional

from transformers import DebertaV2Config, ModernBertConfig, PretrainedConfig
from transformers.models.auto import CONFIG_MAPPING

SUBTOKEN_POOLING_MODES = ("first", "last", "mean", "max")
CONTEXT_ENCODER_MODEL_TYPES = ("deberta-v2", "modernbert", "rnn")


class RNNEncoderConfig(PretrainedConfig):
    """Configuration for an RNN that contextualizes existing hidden states."""

    model_type = "rnn"

    def __init__(
        self,
        hidden_size: int = 512,
        num_hidden_layers: int = 1,
        dropout: float = 0.0,
        bidirectional: bool = True,
        rnn_type: str = "lstm",
        **kwargs,
    ):
        super().__init__(**kwargs)
        rnn_type = rnn_type.lower()
        if rnn_type != "lstm":
            raise ValueError("RNN context encoders currently support only rnn_type='lstm'")
        if hidden_size < 1:
            raise ValueError("RNN encoder hidden_size must be positive")
        if bidirectional and hidden_size % 2:
            raise ValueError("Bidirectional RNN encoder hidden_size must be even")
        if num_hidden_layers < 1:
            raise ValueError("RNN encoder num_hidden_layers must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("RNN encoder dropout must be in the range [0, 1)")

        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.rnn_type = rnn_type


ContextEncoderConfig = Union[DebertaV2Config, ModernBertConfig, RNNEncoderConfig]


def _default_num_attention_heads(hidden_size: int) -> int:
    """Choose a practical attention-head count that divides ``hidden_size``."""
    num_heads = max(1, hidden_size // 64)
    while hidden_size % num_heads:
        num_heads -= 1
    return num_heads


def normalize_context_encoder_config(
    encoder_config: Optional[Union[dict, ContextEncoderConfig]],
    *,
    hidden_size: int,
    dropout: float,
    default_num_hidden_layers: int = 1,
) -> Optional[ContextEncoderConfig]:
    """Normalize configs shared by span and streaming-label context encoders."""
    if encoder_config is None:
        return None

    if isinstance(encoder_config, dict):
        values = encoder_config.copy()
        model_type = values.pop("model_type", None)
        if model_type not in CONTEXT_ENCODER_MODEL_TYPES:
            supported = ", ".join(CONTEXT_ENCODER_MODEL_TYPES)
            raise ValueError(f"Unknown context encoder model_type {model_type!r}. Expected one of: {supported}")

        encoder_hidden_size = values.setdefault("hidden_size", hidden_size)
        values.setdefault("num_hidden_layers", default_num_hidden_layers)

        if model_type == "rnn":
            values.setdefault("dropout", dropout)
            encoder_config = RNNEncoderConfig(**values)
        else:
            values.setdefault("num_attention_heads", _default_num_attention_heads(encoder_hidden_size))
            values.setdefault("intermediate_size", encoder_hidden_size * 4)
            if model_type == "deberta-v2":
                values.setdefault("hidden_dropout_prob", dropout)
                values.setdefault("attention_probs_dropout_prob", dropout)
                values.setdefault("relative_attention", True)
                values.setdefault("pos_att_type", ["p2c", "c2p"])
                values.setdefault("max_relative_positions", 512)
                encoder_config = DebertaV2Config(**values)
            else:
                # ModernBERT is called with ``inputs_embeds``. Keep its unused
                # token table tiny instead of allocating the default 50k rows.
                values.setdefault("vocab_size", 1)
                values.setdefault("pad_token_id", 0)
                values.setdefault("bos_token_id", 0)
                values.setdefault("eos_token_id", 0)
                values.setdefault("cls_token_id", 0)
                values.setdefault("sep_token_id", 0)
                values.setdefault("attention_dropout", dropout)
                values.setdefault("embedding_dropout", dropout)
                values.setdefault("mlp_dropout", dropout)
                encoder_config = ModernBertConfig(**values)
    elif not isinstance(encoder_config, (DebertaV2Config, ModernBertConfig, RNNEncoderConfig)):
        supported = ", ".join(config.__name__ for config in (DebertaV2Config, ModernBertConfig, RNNEncoderConfig))
        raise TypeError(f"Context encoder config must be a dict or one of: {supported}")

    if encoder_config.hidden_size < 1:
        raise ValueError("Context encoder hidden_size must be positive")
    if encoder_config.num_hidden_layers < 1:
        raise ValueError("Context encoder num_hidden_layers must be positive")

    if isinstance(encoder_config, (DebertaV2Config, ModernBertConfig)):
        num_heads = encoder_config.num_attention_heads
        if num_heads < 1:
            raise ValueError("Context encoder num_attention_heads must be positive")
        if encoder_config.hidden_size % num_heads:
            raise ValueError("Context encoder hidden_size must be divisible by num_attention_heads")
        if encoder_config.intermediate_size < 1:
            raise ValueError("Context encoder intermediate_size must be positive")

    return encoder_config


class BaseGLiNERConfig(PretrainedConfig):
    """Base configuration class for all GLiNER models."""

    is_composition = True
    model_type = None

    # Keep positional compatibility for downstream callers of this public config API.
    def __init__(  # noqa: PLR0917, RUF100
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
        span_encoder_config: Optional[dict] = None,
        ent_token: str = "<<ENT>>",
        sep_token: str = "<<SEP>>",
        _attn_implementation: Optional[str] = None,
        token_loss_coef: float = 1.0,
        span_loss_coef: float = 1.0,
        represent_spans: bool = False,
        neg_spans_ratio: float = 1.0,
        precomputed_prompts_mode: Optional[bool] = None,
        id_to_classes: Optional[dict] = None,
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
            subtoken_pooling (str, optional): Subtoken pooling strategy. One of
                "first", "last", "mean", or "max". Defaults to "first".
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
            span_encoder_config (dict, optional): Optional hidden-state encoder
                applied before span representations. Defaults to None.
            ent_token (str, optional): Entity marker token. Defaults to "<<ENT>>".
            sep_token (str, optional): Separator token. Defaults to "<<SEP>>".
            _attn_implementation (str, optional): Attention implementation. Defaults to None.
            token_loss_coef (float, optional): Token loss coefficient. Defaults to 1.0.
            span_loss_coef (float, optional): Span loss coefficient. Defaults to 1.0.
            represent_spans (bool, optional): Whether to represent spans. Defaults to False.
            neg_spans_ratio (float, optional): Ratio of negative spans. Defaults to 1.0.
            precomputed_prompts_mode (Optional[bool]): Whether to use precomputed prompts. Defaults to None.
            id_to_classes (Optional[dict]): Mapping from class IDs to class names. Defaults to None.
            **kwargs: Additional keyword arguments passed to parent class.
        """
        super().__init__(**kwargs)

        if isinstance(encoder_config, dict):
            encoder_config["model_type"] = encoder_config.get("model_type", "deberta-v2")

            encoder_config = CONFIG_MAPPING[encoder_config["model_type"]](**encoder_config)
        self.encoder_config = encoder_config
        self.span_encoder_config = normalize_context_encoder_config(
            span_encoder_config,
            hidden_size=hidden_size,
            dropout=dropout,
        )

        self.model_name = model_name
        self.name = name
        self.max_width = max_width
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.fine_tune = fine_tune
        if subtoken_pooling not in SUBTOKEN_POOLING_MODES:
            supported = ", ".join(SUBTOKEN_POOLING_MODES)
            raise ValueError(f"Unknown subtoken pooling strategy {subtoken_pooling!r}. Expected one of: {supported}")
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
        self.token_loss_coef = token_loss_coef
        self.span_loss_coef = span_loss_coef
        self.represent_spans = represent_spans
        self.neg_spans_ratio = neg_spans_ratio
        self.precomputed_prompts_mode = precomputed_prompts_mode
        self.id_to_classes = id_to_classes


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
        self.model_type = "gliner_uni_encoder_span_decoder"


class UniEncoderTokenDecoderConfig(UniEncoderSpanDecoderConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.span_mode = "token_level"
        self.model_type = "gliner_encoder_token_decoder"
        self.represent_spans = True  # hardcoded to True for token decoder


class StreamingSpanConfig(UniEncoderConfig):
    """Configuration for span NER backed directly by a causal decoder.

    Unlike :class:`UniEncoderSpanDecoderConfig`, the decoder is the text
    backbone itself rather than an auxiliary label-generation head.  A
    dedicated model type keeps the two architectures unambiguous when models
    are saved and loaded through :class:`GLiNER`.  During cached inference,
    ``right_context_width`` controls how many following words may revise a
    span; when omitted it defaults to ``max_width``.  Set it to zero for the
    previous append-only behavior.
    """

    model_type = "gliner_streaming_span"

    def __init__(
        self,
        model_name: Optional[str] = None,
        decoder_config: Optional[dict] = None,
        labels_encoder_config: Optional[dict] = None,
        span_encoder_config: Optional[dict] = None,
        label_token: str = "<<LABEL>>",
        sep_token_index: int = -1,
        max_cache_length: Optional[int] = None,
        right_context_width: Optional[int] = None,
        labels_decoder: Optional[str] = None,
        labels_decoder_config: Optional[dict] = None,
        **kwargs,
    ):
        # ``labels_decoder`` used to identify this architecture's only
        # backbone.  Consume the old fields when loading an existing checkpoint
        # but do not retain or re-serialize them: ``model_name`` and
        # ``decoder_config`` are the canonical fields for StreamingSpan models.
        default_model_name = "microsoft/deberta-v3-small"
        if labels_decoder is not None and (model_name is None or model_name == default_model_name):
            model_name = labels_decoder
        if model_name is None:
            model_name = default_model_name
        if decoder_config is None:
            decoder_config = labels_decoder_config
        if isinstance(decoder_config, dict):
            decoder_config = decoder_config.copy()
            decoder_config["model_type"] = decoder_config.get("model_type", "gpt2")
            decoder_config = CONFIG_MAPPING[decoder_config["model_type"]](**decoder_config)

        # Decoder streaming is causal by default. Optional word context lives
        # in the span representation layer, so do not also create the legacy
        # model-level BiLSTM unless a caller explicitly requests it.
        kwargs.setdefault("num_rnn_layers", 0)
        kwargs.setdefault("span_mode", "markerV2")
        super().__init__(model_name=model_name, span_encoder_config=span_encoder_config, **kwargs)

        # The default label scorer remains a compact, independent DeBERTa-v2
        # encoder, but all supported context encoders use the same contract.
        if labels_encoder_config is None:
            labels_encoder_config = {"model_type": "deberta-v2", "num_hidden_layers": 2}
        labels_encoder_config = normalize_context_encoder_config(
            labels_encoder_config,
            hidden_size=self.hidden_size,
            dropout=self.dropout,
            default_num_hidden_layers=2,
        )
        if max_cache_length is not None and max_cache_length < 1:
            raise ValueError("max_cache_length must be positive when provided")
        if right_context_width is None:
            right_context_width = self.max_width
        if right_context_width < 0:
            raise ValueError("right_context_width must be non-negative")

        self.decoder_config = decoder_config
        self.labels_encoder_config = labels_encoder_config
        self.label_token = label_token
        self.sep_token_index = sep_token_index
        self.max_cache_length = max_cache_length
        self.right_context_width = right_context_width
        self.model_type = "gliner_streaming_span"


class UniEncoderRelexConfig(UniEncoderConfig):
    def __init__(
        self,
        relations_layer: Optional[str] = None,
        triples_layer: Optional[str] = None,
        embed_rel_token: bool = True,
        rel_token_index: int = -1,
        rel_token: str = "<<REL>>",
        adjacency_loss_coef=1.0,
        relation_loss_coef=1.0,
        augment_data_prob=0.5,
        augment_ent_drop_prob=(0.0, 1.0),
        augment_rel_drop_prob=(0.0, 0.3),
        augment_add_other_prob=0.5,
        rel_id_to_classes: Optional[dict] = None,
        **kwargs,
    ):
        """Initialize UniEncoderRelexConfig.

        Args:
            relations_layer (str, optional): Name of relations layer,
                see gliner.modeling.multitask.relations_layers.py. Defaults to None.
                Use "none" to enable single-step relation extraction that scores all
                entity pair combinations directly without adjacency filtering.
            triples_layer (str, optional): Name of triples layer,
                see gliner.modeling.multitask.triples_layers.py. Defaults to None.
            embed_rel_token (bool, optional): Whether to embed relation tokens. Defaults to True.
            rel_token_index (int, optional): Index of relation token. Defaults to -1.
            rel_token (str, optional): Relation marker token. Defaults to "<<REL>>".
            adjacency_loss_coef (float, optional): Adjacency modeling loss coefficient. Defaults to 1.0.
            relation_loss_coef (float, optional): Relation representaton loss coefficient. Defaults to 1.0.
            augment_data_prob (float, optional): Probability of applying data augmentation
                to an example. Defaults to 0.0 (disabled).
            augment_ent_drop_prob (tuple, optional): Range (min, max) from which to sample
                the per-type entity drop probability. Defaults to (0.0, 0.4).
            augment_rel_drop_prob (tuple, optional): Range (min, max) from which to sample
                the per-type relation drop probability. Defaults to (0.0, 0.4).
            augment_add_other_prob (float, optional): Probability of adding "other" relation to a pair with no relation.
            rel_id_to_classes (Optional[dict]): Mapping from relation class IDs to class names. Defaults to None.
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
        self.adjacency_loss_coef = adjacency_loss_coef
        self.relation_loss_coef = relation_loss_coef
        self.augment_data_prob = augment_data_prob
        self.augment_ent_drop_prob = tuple(augment_ent_drop_prob)
        self.augment_rel_drop_prob = tuple(augment_rel_drop_prob)
        self.augment_add_other_prob = augment_add_other_prob
        self.rel_id_to_classes = rel_id_to_classes


class UniEncoderSpanRelexConfig(UniEncoderRelexConfig):
    """Configuration for uni-encoder span model with relation extraction."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_type = "gliner_uni_encoder_span_relex"
        if self.span_mode == "token_level":
            raise ValueError("UniEncoderSpanRelexConfig requires span_mode != 'token_level'")


class UniEncoderTokenRelexConfig(UniEncoderRelexConfig):
    """Configuration for uni-encoder token-level model with relation extraction."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_type = "gliner_uni_encoder_token_relex"
        self.span_mode = "token_level"


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
        self.model_type = self._resolve_model_type()

    def _resolve_model_type(self):
        """Auto-detect model type based on configuration."""
        if self.labels_decoder:
            if self.span_mode == "token-level":
                return "gliner_uni_encoder_token_decoder"
            else:
                return "gliner_uni_encoder_span_decoder"
        elif self.labels_encoder:
            return "gliner_bi_encoder_span" if self.span_mode != "token-level" else "gliner_bi_encoder_token"
        elif self.relations_layer is not None:
            if self.span_mode == "token-level":
                return "gliner_uni_encoder_token_relex"
            else:
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
        "gliner_uni_encoder_token_decoder": UniEncoderTokenDecoderConfig,
        "gliner_streaming_span": StreamingSpanConfig,
        "gliner_uni_encoder_span_relex": UniEncoderSpanRelexConfig,
        "gliner_uni_encoder_token_relex": UniEncoderTokenRelexConfig,
        "gliner_bi_encoder": BiEncoderConfig,
        "gliner_bi_encoder_span": BiEncoderSpanConfig,
        "gliner_bi_encoder_token": BiEncoderTokenConfig,
    }
)
