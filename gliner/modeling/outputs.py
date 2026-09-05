from typing import Any
from dataclasses import field, dataclass

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
        span_embeddings (Optional[torch.FloatTensor]): Native span representations
            used by span-based scorers. Shape: [batch_size, seq_len, max_width, hidden_size]
            or [batch_size, num_spans, hidden_size].
    """

    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    prompts_embedding: torch.FloatTensor | None = None
    prompts_embedding_mask: torch.LongTensor | None = None
    words_embedding: torch.FloatTensor | None = None
    mask: torch.LongTensor | None = None
    span_idx: torch.LongTensor | None = None
    span_mask: torch.Tensor | None = None
    span_logits: torch.FloatTensor | None = None
    span_embeddings: torch.FloatTensor | None = field(default=None, kw_only=True)


@dataclass
class GLiNERRepresentationOutput(ModelOutput):
    """Intermediate prompt and text representations produced by GLiNER backbones."""

    prompts_embedding: torch.FloatTensor | None = None
    prompts_embedding_mask: torch.LongTensor | None = None
    words_embedding: torch.FloatTensor | None = None
    mask: torch.LongTensor | None = None
    past_key_values: Any | None = None
    past_word_embeddings: torch.FloatTensor | None = None
    past_word_mask: torch.LongTensor | None = None


@dataclass
class GLiNERStreamingSpanOutput(GLiNERBaseOutput):
    """Span-classification output with reusable decoder and word cache state."""

    past_key_values: Any | None = None
    past_word_embeddings: torch.FloatTensor | None = None
    past_word_mask: torch.LongTensor | None = None
    cached_prompts_embedding: torch.FloatTensor | None = None
    cached_prompts_mask: torch.LongTensor | None = None


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

    decoder_loss: torch.FloatTensor | None = None
    decoder_embedding: torch.FloatTensor | None = None
    decoder_embedding_mask: torch.LongTensor | None = None
    decoder_span_idx: torch.LongTensor | None = None


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
        rel_prompts_embedding (Optional[torch.FloatTensor]): Embeddings for
            relation type prompts/labels. Shape: [batch_size, num_relation_types, hidden_size].
        rel_prompts_embedding_mask (Optional[torch.LongTensor]): Attention mask
            for relation prompt embeddings. Shape: [batch_size, num_relation_types].
        relation_embeddings (Optional[torch.FloatTensor]): Projected entity-pair
            representations used by pair-based relation scorers. Shape:
            [batch_size, num_relations, hidden_size].
        relation_head_embeddings (Optional[torch.FloatTensor]): Selected head
            representations used by triple-based relation scorers. Shape:
            [batch_size, num_relations, hidden_size].
        relation_tail_embeddings (Optional[torch.FloatTensor]): Selected tail
            representations used by triple-based relation scorers. Shape:
            [batch_size, num_relations, hidden_size].
    """

    rel_idx: torch.LongTensor | None = None
    rel_logits: torch.FloatTensor | None = None
    rel_mask: torch.FloatTensor | None = None
    entity_spans: torch.LongTensor | None = None
    rel_prompts_embedding: torch.FloatTensor | None = None
    rel_prompts_embedding_mask: torch.LongTensor | None = None
    relation_embeddings: torch.FloatTensor | None = field(default=None, kw_only=True)
    relation_head_embeddings: torch.FloatTensor | None = field(default=None, kw_only=True)
    relation_tail_embeddings: torch.FloatTensor | None = field(default=None, kw_only=True)
