from typing import Optional
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

# Register the configuration
from transformers import CONFIG_MAPPING
CONFIG_MAPPING.update({"gliner": GLiNERConfig})