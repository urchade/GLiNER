import torch
from torch import nn
from transformers import AutoModel, AutoConfig

#just wraping to allow to load previously created models
class Transformer(nn.Module):
    def __init__(self, config, from_pretrained):
        super().__init__()
        if from_pretrained:
            self.model = AutoModel.from_pretrained(config.model_name)
        else:
            if config.encoder_config is None:
                encoder_config = AutoConfig.from_pretrained(config.model_name)
                if config.vocab_size!=-1:
                    encoder_config.vocab_size = config.vocab_size
   
            else:
                encoder_config = config.encoder_config 
            self.model = AutoModel.from_config(encoder_config)
    
    def forward(self, *args, **kwargs):
        output = self.model(*args, **kwargs)
        return output[0]
    
class Encoder(nn.Module):
    def __init__(self, config, from_pretrained: bool = False):
        super().__init__()

        self.bert_layer = Transformer( #transformer_model
            config, from_pretrained,
        )

        bert_hidden_size = self.bert_layer.model.config.hidden_size

        if config.hidden_size != bert_hidden_size:
            self.projection = nn.Linear(bert_hidden_size, config.hidden_size)

    def resize_token_embeddings(self, new_num_tokens, pad_to_multiple_of=None):
        return self.bert_layer.model.resize_token_embeddings(new_num_tokens, 
                                                                            pad_to_multiple_of)
    def forward(self, *args, **kwargs) -> torch.Tensor:
        token_embeddings = self.bert_layer(*args, **kwargs)
        if hasattr(self, "projection"):
            token_embeddings = self.projection(token_embeddings)

        return token_embeddings