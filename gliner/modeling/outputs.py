from typing import Optional
from dataclasses import dataclass

import torch
from transformers.utils import ModelOutput


@dataclass
class GLiNERBaseOutput(ModelOutput):
    """Base output class for GLiNER models.

    This class contains the fundamental outputs produced by GLiNER models,
    including loss, logits, and embeddings for both prompts (entity types)
    and input words/tokens.

    Attributes:
        loss (Optional[torch.FloatTensor]): The total loss for training.
            Shape: scalar tensor.
        logits (Optional[torch.FloatTensor]): The prediction scores for
            entity spans or other outputs. Shape varies depending on the model configuration,
            typically [batch_size, num_spans, num_classes] or similar.
        prompts_embedding (Optional[torch.FloatTensor]): Embeddings for the
            entity type prompts/labels. Shape: [batch_size, num_classes, hidden_size].
        prompts_embedding_mask (Optional[torch.LongTensor]): Attention mask
            for prompt embeddings. Shape: [batch_size, num_classes].
        words_embedding (Optional[torch.FloatTensor]): Embeddings for input
            words/tokens. Shape: [batch_size, seq_len, hidden_size].
        mask (Optional[torch.LongTensor]): Attention mask for input tokens.
            Shape: [batch_size, seq_len].
    """

    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    prompts_embedding: Optional[torch.FloatTensor] = None
    prompts_embedding_mask: Optional[torch.LongTensor] = None
    words_embedding: Optional[torch.FloatTensor] = None
    mask: Optional[torch.LongTensor] = None


@dataclass
class GLiNERDecoderOutput(GLiNERBaseOutput):
    """Output class for GLiNER models with decoder components.

    Extends GLiNERBaseOutput with additional decoder-specific outputs,
    including decoder embeddings and span indices. This is typically used
    in GLiNER variants that include an explicit decoder module.

    Attributes:
        loss (Optional[torch.FloatTensor]): Inherited from GLiNERBaseOutput.
        logits (Optional[torch.FloatTensor]): Inherited from GLiNERBaseOutput.
        prompts_embedding (Optional[torch.FloatTensor]): Inherited from GLiNERBaseOutput.
        prompts_embedding_mask (Optional[torch.LongTensor]): Inherited from GLiNERBaseOutput.
        words_embedding (Optional[torch.FloatTensor]): Inherited from GLiNERBaseOutput.
        mask (Optional[torch.LongTensor]): Inherited from GLiNERBaseOutput.
        decoder_loss (Optional[torch.FloatTensor]): Loss specific to the
            decoder component. Shape: scalar tensor.
        decoder_embedding (Optional[torch.FloatTensor]): Output embeddings
            from the decoder. Shape: [batch_size, num_decoder_tokens, hidden_size].
        decoder_embedding_mask (Optional[torch.LongTensor]): Attention mask
            for decoder embeddings. Shape: [batch_size, num_decoder_tokens].
        decoder_span_idx (Optional[torch.LongTensor]): Indices of spans
            processed by the decoder. Shape: [batch_size, num_spans, 2],
            where the last dimension contains [start_idx, end_idx].
    """

    decoder_loss: Optional[torch.FloatTensor] = None
    decoder_embedding: Optional[torch.FloatTensor] = None
    decoder_embedding_mask: Optional[torch.LongTensor] = None
    decoder_span_idx: Optional[torch.LongTensor] = None


@dataclass
class GLiNERRelexOutput(GLiNERBaseOutput):
    """Output class for GLiNER models with relation extraction.

    Extends GLiNERBaseOutput with relation-specific outputs for models
    that perform both entity recognition and relation extraction (Relex).
    This enables joint modeling of entities and their relationships.

    Attributes:
        loss (Optional[torch.FloatTensor]): Inherited from GLiNERBaseOutput.
        logits (Optional[torch.FloatTensor]): Inherited from GLiNERBaseOutput.
        prompts_embedding (Optional[torch.FloatTensor]): Inherited from GLiNERBaseOutput.
        prompts_embedding_mask (Optional[torch.LongTensor]): Inherited from GLiNERBaseOutput.
        words_embedding (Optional[torch.FloatTensor]): Inherited from GLiNERBaseOutput.
        mask (Optional[torch.LongTensor]): Inherited from GLiNERBaseOutput.
        rel_idx (Optional[torch.LongTensor]): Indices of entity pairs for
            which relations are predicted. Shape: [batch_size, num_relations, 2],
            where the last dimension contains indices of the two entities.
        rel_logits (Optional[torch.FloatTensor]): Prediction scores for
            relations between entity pairs. Shape: [batch_size, num_relations, num_relation_types].
        rel_mask (Optional[torch.FloatTensor]): Mask indicating valid relation
            predictions. Shape: [batch_size, num_relations].
    """

    rel_idx: Optional[torch.LongTensor] = None
    rel_logits: Optional[torch.FloatTensor] = None
    rel_mask: Optional[torch.FloatTensor] = None
