from typing import Optional, Union, Tuple, Dict
from transformers import PretrainedConfig
from transformers.models.auto import CONFIG_MAPPING

class GLiNERConfig(PretrainedConfig):
    model_type = "gliner"
    is_composition = True
    def __init__(self, 
                 model_name: str = "microsoft/deberta-v3-small",
                 labels_encoder: str = None,
                 name: str = "span level gliner",
                 max_width: int = 12,
                 hidden_size: int = 512,
                 dropout: float = 0.4,
                 fine_tune: bool = True,
                 subtoken_pooling: str = "first",
                 span_mode: str = "markerV0",
                 post_fusion_schema: str = '', #l2l-l2t-t2t
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
                 labels_encoder_config: Optional[dict] = None,
                 ent_token = "<<ENT>>",
                 sep_token = "<<SEP>>",
                 _attn_implementation = None,
                 **kwargs):
        super().__init__(**kwargs)
        if isinstance(encoder_config, dict):
            encoder_config["model_type"] = (encoder_config["model_type"] 
                                                if "model_type" in encoder_config 
                                                else "deberta-v2")
            encoder_config = CONFIG_MAPPING[encoder_config["model_type"]](**encoder_config)
        self.encoder_config = encoder_config

        if isinstance(labels_encoder_config, dict):
            labels_encoder_config["model_type"] = (labels_encoder_config["model_type"] 
                                                if "model_type" in labels_encoder_config 
                                                else "deberta-v2")
            labels_encoder_config = CONFIG_MAPPING[labels_encoder_config["model_type"]](**labels_encoder_config)
        self.labels_encoder_config = labels_encoder_config

        self.model_name = model_name
        self.labels_encoder = labels_encoder
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

class GLiNERPDFConfig(GLiNERConfig):
    model_type = "gliner_pdf"

    def __init__(
        self,
        model_name: str = "microsoft/deberta-v3-small",
        labels_encoder: Optional[str] = None,
        name: str = "span level gliner pdf",
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
        has_rnn: bool = True,
        fuse_layers: bool = False,
        embed_ent_token: bool = True,
        class_token_index: int = -1,
        image_token_index: int = 0,
        ent_token: str = "<<ENT>>",
        sep_token: str = "<<SEP>>",
        image_token: str = "<<IMG>>",
        page_token: str = "<<PAGE>>",
        _attn_implementation=None,
        image_size: int = None,
        patch_size: int = None,
        vision_encoder: str = "google/vit-base-patch16-224",
        vision_encoder_config: Optional[Dict] = None,
        use_patch_embeddings: bool = False,
        num_channels: int = 3,
        vision_feature_layer: int = 0,
        vision_feature_select_strategy: str = "default",
        max_2d_position_embeddings: int = 1024,
        coordinate_size: int = 64,
        shape_size: int = 64,
        **kwargs
    ):
        super().__init__(
            model_name=model_name,
            labels_encoder=labels_encoder,
            name=name,
            max_width=max_width,
            hidden_size=hidden_size,
            dropout=dropout,
            fine_tune=fine_tune,
            subtoken_pooling=subtoken_pooling,
            span_mode=span_mode,
            post_fusion_schema=post_fusion_schema,
            num_post_fusion_layers=num_post_fusion_layers,
            vocab_size=vocab_size,
            max_neg_type_ratio=max_neg_type_ratio,
            max_types=max_types,
            max_len=max_len,
            words_splitter_type=words_splitter_type,
            has_rnn=has_rnn,
            fuse_layers=fuse_layers,
            embed_ent_token=embed_ent_token,
            class_token_index=class_token_index,
            ent_token=ent_token,
            sep_token=sep_token,
            _attn_implementation=_attn_implementation,
            **kwargs
        )

        self.vision_encoder = vision_encoder
        if isinstance(vision_encoder_config, dict):
            cfg = vision_encoder_config.copy()
            cfg["model_type"] = cfg.get("model_type", "layoutlmv3")
            self.vision_encoder_config = CONFIG_MAPPING[cfg["model_type"]](**cfg)
        else:
            self.vision_encoder_config = vision_encoder_config

        self.use_patch_embeddings = use_patch_embeddings
        self.patch_size = self.vision_encoder_config.patch_size if patch_size is None else patch_size
        self.image_size = self.vision_encoder_config.image_size if image_size is None else image_size
        self.num_channels = num_channels

        self.vision_feature_layer = vision_feature_layer
        self.vision_feature_select_strategy = vision_feature_select_strategy
        self.image_token = image_token
        self.image_token_index = image_token_index
        self.page_token = page_token

        self.max_2d_position_embeddings = max_2d_position_embeddings
        self.coordinate_size = coordinate_size
        self.shape_size = shape_size

        self.text_config = self.encoder_config

        if not hasattr(self.text_config, "layer_norm_epsilon"):
            eps = getattr(self.text_config, "layer_norm_eps", getattr(self.text_config, "layer_norm_epsilon", 1e-5))
            setattr(self.text_config, "layer_norm_epsilon", eps)
        if not hasattr(self.text_config, "dropout_rate"):
            dr = getattr(self.text_config, "hidden_dropout_prob", getattr(self.text_config, "dropout_rate", self.dropout))
            setattr(self.text_config, "dropout_rate", dr)


CONFIG_MAPPING.update({"gliner": GLiNERConfig, "gliner_pdf": GLiNERPDFConfig})