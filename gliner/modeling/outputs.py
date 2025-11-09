from dataclasses import dataclass
from typing import Optional
import torch
from transformers.utils import ModelOutput


@dataclass
class GLiNERBaseOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    prompts_embedding: Optional[torch.FloatTensor] = None
    prompts_embedding_mask: Optional[torch.LongTensor] = None
    words_embedding: Optional[torch.FloatTensor] = None
    mask: Optional[torch.LongTensor] = None

@dataclass
class GLiNERDecoderOutput(GLiNERBaseOutput):
    decoder_loss: Optional[torch.FloatTensor] = None
    decoder_embedding: Optional[torch.FloatTensor] = None
    decoder_embedding_mask: Optional[torch.LongTensor] = None
    decoder_span_idx: Optional[torch.LongTensor] = None

@dataclass
class GLiNERRelexOutput(GLiNERBaseOutput):
    rel_idx: Optional[torch.LongTensor] = None
    rel_logits: Optional[torch.FloatTensor] = None
    rel_mask: Optional[torch.FloatTensor] = None