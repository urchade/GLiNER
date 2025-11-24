"""Base model classes for GLiNER neural network architectures.

This module provides abstract base classes and concrete implementations for various
encoder-decoder architectures used in named entity recognition (NER) and relation
extraction tasks.

Classes:
    BaseModel: Abstract base class for all models.
    BaseUniEncoderModel: Base class for uni-encoder architectures.
    UniEncoderSpanModel: Span-based NER model with uni-encoder.
    UniEncoderTokenModel: Token-based NER model with uni-encoder.
    BaseBiEncoderModel: Base class for bi-encoder architectures.
    BiEncoderSpanModel: Span-based NER model with bi-encoder.
    BiEncoderTokenModel: Token-based NER model with bi-encoder.
    UniEncoderSpanDecoderModel: Span model with decoder for label generation.
    UniEncoderSpanRelexModel: Span model with relation extraction capabilities.
"""

import warnings
from abc import ABC, abstractmethod
from typing import Any, Tuple, Union, Optional
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .utils import (
    build_entity_pairs,
    extract_prompt_features,
    extract_word_embeddings,
    extract_prompt_features_and_word_embeddings,
)
from .layers import CrossFuser, LstmSeq2SeqEncoder, create_projection_layer
from .decoder import Decoder
from .encoder import Encoder, BiEncoder
from .outputs import GLiNERBaseOutput, GLiNERRelexOutput, GLiNERDecoderOutput
from .scorers import Scorer
from .span_rep import SpanRepLayer
from .loss_functions import cross_entropy_loss, focal_loss_with_logits
from .multitask.triples_layers import TriplesScoreLayer
from .multitask.relations_layers import RelationsRepLayer


class BaseModel(ABC, nn.Module):
    """Abstract base class for all GLiNER models.

    This class defines the common interface and shared functionality for all model
    architectures. It includes methods for padding/truncating sequences and computing
    losses with various masking strategies.

    Attributes:
        data_processor: Data processor for handling input preprocessing.
        config: Model configuration object.
        from_pretrained (bool): Whether model was loaded from pretrained weights.
        cache_dir (Optional[Path]): Directory for caching pretrained models.
    """

    data_processor = None

    def __init__(
        self, config: Any, from_pretrained: bool = False, cache_dir: Optional[Union[str, Path]] = None
    ) -> None:
        """Initialize the base model.

        Args:
            config: Configuration object containing model hyperparameters.
            from_pretrained: Whether to load from pretrained weights.
            cache_dir: Directory path for caching pretrained models.
        """
        super().__init__()
        self.config = config
        self.from_pretrained = from_pretrained
        self.cache_dir = cache_dir

    @abstractmethod
    def get_representations(self) -> Tuple[torch.Tensor, ...]:
        """Get intermediate representations from the model.

        Returns:
            Tuple of tensors representing intermediate model outputs.
        """
        pass

    @staticmethod
    def _fit_length(embedding: torch.Tensor, mask: torch.Tensor, target_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Make embedding and mask exactly target_len along dimension 1.

        Pads with zeros if current length is less than target, truncates if greater.

        Args:
            embedding: Tensor of shape (B, L, D) containing embeddings.
            mask: Tensor of shape (B, L) containing attention mask.
            target_len: Desired sequence length.

        Returns:
            Tuple containing:
                - embedding: Resized embedding tensor of shape (B, target_len, D).
                - mask: Resized mask tensor of shape (B, target_len).
        """
        L = embedding.shape[1]

        if target_len == L:
            return embedding, mask

        if target_len > L:
            pad_len = target_len - L
            embedding = F.pad(embedding, (0, 0, 0, pad_len))
            mask = F.pad(mask, (0, pad_len))
        else:
            embedding = embedding[:, :target_len]
            mask = mask[:, :target_len]

        return embedding, mask

    @abstractmethod
    def forward(self, x: Any) -> Any:
        """Forward pass through the model.

        Args:
            x: Input data.

        Returns:
            Model outputs.
        """
        pass

    def _loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        alpha: float = -1.0,
        gamma: float = 0.0,
        prob_margin: float = 0.0,
        label_smoothing: float = 0.0,
        negatives: float = 1.0,
        masking: str = "none",
        normalize_prob: bool = True,
    ) -> torch.Tensor:
        """Compute loss with optional negative sampling and masking.

        This method computes focal loss and applies negative sampling to balance
        positive and negative examples in the training data.

        Args:
            logits: Predicted logits from the model.
            labels: Ground truth labels.
            alpha: Weight factor for balanced focal loss. If -1, no balancing.
            gamma: Focusing parameter for focal loss.
            prob_margin: Margin to subtract from probabilities.
            label_smoothing: Label smoothing factor.
            negatives: Probability of sampling negative examples.
            masking: Masking strategy, one of "none", "global", "label", or "span".
            normalize_prob: Whether to normalize probabilities in loss computation.

        Returns:
            Loss tensor of same shape as labels.
        """
        # Compute the loss per element using the focal loss function
        all_losses = focal_loss_with_logits(
            logits,
            labels,
            alpha=alpha,
            gamma=gamma,
            prob_margin=prob_margin,
            label_smoothing=label_smoothing,
            normalize_prob=normalize_prob,
        )

        # Create a mask of the same shape as labels:
        # For elements where labels==0, sample a Bernoulli random variable that is 1 with probability `negatives`
        # For elements where labels==1, set the mask to 1 (i.e. do not change these losses)
        if masking == "global":
            mask_neg = torch.where(labels == 0, (torch.rand_like(labels) < negatives).float(), torch.ones_like(labels))
        elif masking == "label":
            neg_proposals = (labels.sum(dim=1) == 0).unsqueeze(1).expand_as(labels)
            mask_neg = torch.where(
                neg_proposals,
                (torch.rand_like(neg_proposals.float()) < negatives).float(),
                torch.ones_like(neg_proposals.float()),
            )
        elif masking == "span":
            neg_proposals = (labels.sum(dim=2) == 0).unsqueeze(2).expand_as(labels)
            mask_neg = torch.where(
                neg_proposals,
                (torch.rand_like(neg_proposals.float()) < negatives).float(),
                torch.ones_like(neg_proposals.float()),
            )
        else:
            mask_neg = 1.0

        # Apply the mask: for negative examples, some losses will be zeroed out based on the sampling
        all_losses = all_losses * mask_neg

        return all_losses

    @abstractmethod
    def loss(self, x: Any) -> torch.Tensor:
        """Compute the total loss for the model.

        Args:
            x: Input data and labels.

        Returns:
            Scalar loss tensor.
        """
        pass


class BaseUniEncoderModel(BaseModel):
    """Base class for uni-encoder model architectures.

    Uni-encoder models use a single encoder for both text and entity labels,
    embedding them in the same semantic space.

    Attributes:
        token_rep_layer (Encoder): Token-level representation encoder.
        rnn (Optional[LstmSeq2SeqEncoder]): Optional LSTM layer for sequence modeling.
        cross_fuser (Optional[CrossFuser]): Optional cross-attention fusion layer.
    """

    def __init__(
        self, config: Any, from_pretrained: bool = False, cache_dir: Optional[Union[str, Path]] = None
    ) -> None:
        """Initialize the uni-encoder model.

        Args:
            config: Model configuration object.
            from_pretrained: Whether to load from pretrained weights.
            cache_dir: Directory for caching pretrained models.
        """
        super().__init__(config, from_pretrained, cache_dir)
        self.token_rep_layer = Encoder(config, from_pretrained, cache_dir=cache_dir)

        if self.config.num_rnn_layers>0:
            self.rnn = LstmSeq2SeqEncoder(config, num_layers=self.config.num_rnn_layers)

        if config.post_fusion_schema:
            self.cross_fuser = CrossFuser(
                self.config.hidden_size,
                self.config.hidden_size,
                num_heads=self.token_rep_layer.bert_layer.model.config.num_attention_heads,
                num_layers=self.config.num_post_fusion_layers,
                dropout=config.dropout,
                schema=config.post_fusion_schema,
            )

    def _extract_prompt_features_and_word_embeddings(
        self,
        token_embeds: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        text_lengths: torch.Tensor,
        words_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract entity label prompts and word embeddings from token embeddings.

        Args:
            token_embeds: Token-level embeddings of shape (B, L, D).
            input_ids: Input token IDs of shape (B, L).
            attention_mask: Attention mask of shape (B, L).
            text_lengths: Length of each text sequence in batch.
            words_mask: Mask indicating word boundaries.

        Returns:
            Tuple containing:
                - prompts_embedding: Entity label embeddings of shape (B, C, D).
                - prompts_embedding_mask: Mask for prompts of shape (B, C).
                - words_embedding: Word-level embeddings of shape (B, W, D).
                - mask: Mask for words of shape (B, W).
        """
        prompts_embedding, prompts_embedding_mask, words_embedding, mask = extract_prompt_features_and_word_embeddings(
            self.config.class_token_index,
            token_embeds,
            input_ids,
            attention_mask,
            text_lengths,
            words_mask,
            self.config.embed_ent_token,
        )
        return prompts_embedding, prompts_embedding_mask, words_embedding, mask

    def get_representations(
        self,
        input_ids: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        text_lengths: Optional[torch.Tensor] = None,
        words_mask: Optional[torch.LongTensor] = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get entity label and word representations from input.

        Args:
            input_ids: Input token IDs of shape (B, L).
            attention_mask: Attention mask of shape (B, L).
            text_lengths: Length of each text in batch.
            words_mask: Word boundary mask.
            **kwargs: Additional arguments for the encoder.

        Returns:
            Tuple containing:
                - prompts_embedding: Entity label embeddings of shape (B, C, D).
                - prompts_embedding_mask: Mask for prompts of shape (B, C).
                - words_embedding: Word embeddings of shape (B, W, D).
                - mask: Mask for words of shape (B, W).
        """
        token_embeds = self.token_rep_layer(input_ids, attention_mask, **kwargs)

        prompts_embedding, prompts_embedding_mask, words_embedding, mask = (
            self._extract_prompt_features_and_word_embeddings(
                token_embeds, input_ids, attention_mask, text_lengths, words_mask
            )
        )

        if hasattr(self, "rnn"):
            words_embedding = self.rnn(words_embedding, mask)

        return prompts_embedding, prompts_embedding_mask, words_embedding, mask


class UniEncoderSpanModel(BaseUniEncoderModel):
    """Span-based NER model using uni-encoder architecture.

    This model identifies entity spans by scoring all possible spans against
    entity type embeddings.

    Attributes:
        span_rep_layer (SpanRepLayer): Layer for computing span representations.
        prompt_rep_layer (nn.Module): Projection layer for entity label embeddings.
    """

    def __init__(
        self, config: Any, from_pretrained: bool = False, cache_dir: Optional[Union[str, Path]] = None
    ) -> None:
        """Initialize the span-based uni-encoder model.

        Args:
            config: Model configuration object.
            from_pretrained: Whether to load from pretrained weights.
            cache_dir: Directory for caching pretrained models.
        """
        super().__init__(config, from_pretrained, cache_dir)
        self.span_rep_layer = SpanRepLayer(
            span_mode=config.span_mode,
            hidden_size=config.hidden_size,
            max_width=config.max_width,
            dropout=config.dropout,
        )

        self.prompt_rep_layer = create_projection_layer(config.hidden_size, config.dropout)

    def forward(
        self,
        input_ids: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        words_embedding: Optional[torch.FloatTensor] = None,
        mask: Optional[torch.LongTensor] = None,
        prompts_embedding: Optional[torch.FloatTensor] = None,
        prompts_embedding_mask: Optional[torch.LongTensor] = None,
        words_mask: Optional[torch.LongTensor] = None,
        text_lengths: Optional[torch.Tensor] = None,
        span_idx: Optional[torch.LongTensor] = None,
        span_mask: Optional[torch.LongTensor] = None,
        labels: Optional[torch.FloatTensor] = None,
        **kwargs: Any,
    ) -> GLiNERBaseOutput:
        """Forward pass through the span-based model.

        Args:
            input_ids: Input token IDs of shape (B, L).
            attention_mask: Attention mask of shape (B, L).
            words_embedding: Pre-computed word embeddings of shape (B, W, D).
            mask: Mask for words of shape (B, W).
            prompts_embedding: Pre-computed entity label embeddings of shape (B, C, D).
            prompts_embedding_mask: Mask for prompts of shape (B, C).
            words_mask: Word boundary mask.
            text_lengths: Length of each text sequence.
            span_idx: Span indices of shape (B, L*K, 2).
            span_mask: Mask for valid spans of shape (B, L, K).
            labels: Ground truth labels of shape (B, L, K, C).
            **kwargs: Additional arguments.

        Returns:
            GLiNERBaseOutput containing logits, loss, and intermediate representations.
        """
        encoder_kwargs = {key: kwargs[key] for key in ("packing_config", "pair_attention_mask") if key in kwargs}

        prompts_embedding, prompts_embedding_mask, words_embedding, mask = self.get_representations(
            input_ids, attention_mask, text_lengths, words_mask, **encoder_kwargs
        )

        target_W = span_idx.size(1) // self.config.max_width
        words_embedding, mask = self._fit_length(words_embedding, mask, target_W)

        span_idx = span_idx * span_mask.unsqueeze(-1)

        span_rep = self.span_rep_layer(words_embedding, span_idx)

        target_C = prompts_embedding.size(1)
        if labels is not None:
            target_C = max(target_C, labels.size(-1))

        prompts_embedding, prompts_embedding_mask = self._fit_length(
            prompts_embedding, prompts_embedding_mask, target_C
        )

        prompts_embedding = self.prompt_rep_layer(prompts_embedding)
        scores = torch.einsum("BLKD,BCD->BLKC", span_rep, prompts_embedding)

        loss = None
        if labels is not None:
            loss = self.loss(scores, labels, prompts_embedding_mask, span_mask, **kwargs)

        output = GLiNERBaseOutput(
            logits=scores,
            loss=loss,
            prompts_embedding=prompts_embedding,
            prompts_embedding_mask=prompts_embedding_mask,
            words_embedding=words_embedding,
            mask=mask,
        )
        return output

    def loss(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor,
        prompts_embedding_mask: torch.Tensor,
        mask_label: torch.Tensor,
        alpha: float = -1.0,
        gamma: float = 0.0,
        prob_margin: float = 0.0,
        label_smoothing: float = 0.0,
        reduction: str = "sum",
        negatives: float = 1.0,
        masking: str = "none",
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute span classification loss.

        Args:
            scores: Predicted scores of shape (B, L, K, C).
            labels: Ground truth labels of shape (B, L, K, C).
            prompts_embedding_mask: Mask for valid entity types of shape (B, C).
            mask_label: Mask for valid spans of shape (B, L, K).
            alpha: Focal loss alpha parameter.
            gamma: Focal loss gamma parameter.
            prob_margin: Margin for probability adjustment.
            label_smoothing: Label smoothing factor.
            reduction: Loss reduction method ('sum' or 'mean').
            negatives: Negative sampling probability.
            masking: Masking strategy for negative sampling.
            **kwargs: Additional arguments.

        Returns:
            Scalar loss tensor.
        """
        batch_size = scores.shape[0]
        num_classes = prompts_embedding_mask.shape[-1]

        # Reshape scores and labels to match the expected shape
        BS, _, _, CL = scores.shape

        scores = scores.view(BS, -1, CL)
        labels = labels.view(BS, -1, CL)

        all_losses = self._loss(
            scores, labels, alpha, gamma, prob_margin, label_smoothing, negatives=negatives, masking=masking
        )

        masked_loss = all_losses.view(batch_size, -1, num_classes) * prompts_embedding_mask.unsqueeze(1)
        all_losses = masked_loss.view(-1, num_classes)

        mask_label = mask_label.view(-1, 1)

        all_losses = all_losses * mask_label.float()

        if reduction == "mean":
            loss = all_losses.mean()
        elif reduction == "sum":
            loss = all_losses.sum()
        else:
            warnings.warn(
                f"Invalid Value for config 'loss_reduction': '{reduction}' \n Supported reduction modes:"
                f" 'none', 'mean', 'sum'. It will be used 'sum' instead.",
                stacklevel=2,
            )
            loss = all_losses.sum()

        return loss


class UniEncoderTokenModel(BaseUniEncoderModel):
    """Token-based NER model using uni-encoder architecture.

    This model classifies each word independently as entity type or non-entity.

    Attributes:
        scorer (Scorer): Scoring layer for computing token-label compatibility.
    """

    def __init__(
        self, config: Any, from_pretrained: bool = False, cache_dir: Optional[Union[str, Path]] = None
    ) -> None:
        """Initialize the token-based uni-encoder model.

        Args:
            config: Model configuration object.
            from_pretrained: Whether to load from pretrained weights.
            cache_dir: Directory for caching pretrained models.
        """
        super().__init__(config, from_pretrained, cache_dir)
        self.scorer = Scorer(config.hidden_size, config.dropout)

    def forward(
        self,
        input_ids: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        words_embedding: Optional[torch.FloatTensor] = None,
        mask: Optional[torch.LongTensor] = None,
        prompts_embedding: Optional[torch.FloatTensor] = None,
        prompts_embedding_mask: Optional[torch.LongTensor] = None,
        words_mask: Optional[torch.LongTensor] = None,
        text_lengths: Optional[torch.Tensor] = None,
        labels: Optional[torch.FloatTensor] = None,
        **kwargs: Any,
    ) -> GLiNERBaseOutput:
        """Forward pass through the token-based model.

        Args:
            input_ids: Input token IDs of shape (B, L).
            attention_mask: Attention mask of shape (B, L).
            words_embedding: Pre-computed word embeddings of shape (B, W, D).
            mask: Mask for words of shape (B, W).
            prompts_embedding: Pre-computed entity label embeddings of shape (B, C, D).
            prompts_embedding_mask: Mask for prompts of shape (B, C).
            words_mask: Word boundary mask.
            text_lengths: Length of each text sequence.
            labels: Ground truth labels of shape (B, W, C).
            **kwargs: Additional arguments.

        Returns:
            GLiNERBaseOutput containing logits, loss, and intermediate representations.
        """
        encoder_kwargs = {key: kwargs[key] for key in ("packing_config", "pair_attention_mask") if key in kwargs}

        prompts_embedding, prompts_embedding_mask, words_embedding, mask = self.get_representations(
            input_ids, attention_mask, text_lengths, words_mask, **encoder_kwargs
        )

        if labels is not None:
            target_W = labels.shape[1]
            words_embedding, mask = self._fit_length(words_embedding, mask, target_W)

            target_C = prompts_embedding.size(1)
            if labels is not None:
                target_C = max(target_C, labels.size(-2))

            prompts_embedding, prompts_embedding_mask = self._fit_length(
                prompts_embedding, prompts_embedding_mask, target_C
            )

        scores = self.scorer(words_embedding, prompts_embedding)

        loss = None
        if labels is not None:
            loss = self.loss(scores, labels, prompts_embedding_mask, mask, **kwargs)

        output = GLiNERBaseOutput(
            logits=scores,
            loss=loss,
            prompts_embedding=prompts_embedding,
            prompts_embedding_mask=prompts_embedding_mask,
            words_embedding=words_embedding,
            mask=mask,
        )
        return output

    def loss(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor,
        prompts_embedding_mask: torch.Tensor,
        mask: torch.Tensor,
        alpha: float = -1.0,
        gamma: float = 0.0,
        prob_margin: float = 0.0,
        label_smoothing: float = 0.0,
        reduction: str = "sum",
        negatives: float = 1.0,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute token classification loss.

        Args:
            scores: Predicted scores of shape (B, W, C).
            labels: Ground truth labels of shape (B, W, C).
            prompts_embedding_mask: Mask for valid entity types of shape (B, C).
            mask: Mask for valid tokens of shape (B, W).
            alpha: Focal loss alpha parameter.
            gamma: Focal loss gamma parameter.
            prob_margin: Margin for probability adjustment.
            label_smoothing: Label smoothing factor.
            reduction: Loss reduction method ('sum' or 'mean').
            negatives: Negative sampling probability.
            **kwargs: Additional arguments.

        Returns:
            Scalar loss tensor.
        """
        all_losses = self._loss(scores, labels, alpha, gamma, prob_margin, label_smoothing, negatives)

        all_losses = all_losses * (mask.unsqueeze(-1) * prompts_embedding_mask.unsqueeze(1)).unsqueeze(-1)

        if reduction == "mean":
            loss = all_losses.mean()
        elif reduction == "sum":
            loss = all_losses.sum()
        else:
            warnings.warn(
                f"Invalid Value for config 'loss_reduction': '{reduction}' \n Supported reduction modes:"
                f" 'none', 'mean', 'sum'. It will be used 'sum' instead.",
                stacklevel=2,
            )
            loss = all_losses.sum()
        return loss


class BaseBiEncoderModel(BaseModel):
    """Base class for bi-encoder model architectures.

    Bi-encoder models use separate encoders for text and entity labels,
    allowing independent encoding before fusion.

    Attributes:
        token_rep_layer (BiEncoder): Bi-encoder for text and labels.
        rnn (Optional[LstmSeq2SeqEncoder]): Optional LSTM layer.
        cross_fuser (Optional[CrossFuser]): Optional cross-attention fusion layer.
    """

    def __init__(
        self, config: Any, from_pretrained: bool = False, cache_dir: Optional[Union[str, Path]] = None
    ) -> None:
        """Initialize the bi-encoder model.

        Args:
            config: Model configuration object.
            from_pretrained: Whether to load from pretrained weights.
            cache_dir: Directory for caching pretrained models.
        """
        super().__init__(config, from_pretrained, cache_dir)
        self.token_rep_layer = BiEncoder(config, from_pretrained, cache_dir=cache_dir)

        if self.config.num_rnn_layers:
            self.rnn = LstmSeq2SeqEncoder(config, num_layers=self.config.num_rnn_layers)

        if config.post_fusion_schema:
            self.cross_fuser = CrossFuser(
                self.config.hidden_size,
                self.config.hidden_size,
                num_heads=self.token_rep_layer.bert_layer.model.config.num_attention_heads,
                num_layers=self.config.num_post_fusion_layers,
                dropout=config.dropout,
                schema=config.post_fusion_schema,
            )

    def features_enhancement(
        self,
        text_embeds: torch.Tensor,
        labels_embeds: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
        labels_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Enhance features using cross-attention fusion.

        Args:
            text_embeds: Text embeddings of shape (B, W, D).
            labels_embeds: Label embeddings of shape (B, C, D).
            text_mask: Mask for text of shape (B, W).
            labels_mask: Mask for labels of shape (B, C).

        Returns:
            Tuple containing:
                - Enhanced text embeddings of shape (B, W, D).
                - Enhanced label embeddings of shape (B, C, D).
        """
        labels_embeds, text_embeds = self.cross_fuser(labels_embeds, text_embeds, labels_mask, text_mask)
        return text_embeds, labels_embeds

    def get_representations(
        self,
        input_ids: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        labels_embeds: Optional[torch.FloatTensor] = None,
        labels_input_ids: Optional[torch.FloatTensor] = None,
        labels_attention_mask: Optional[torch.LongTensor] = None,
        text_lengths: Optional[torch.Tensor] = None,
        words_mask: Optional[torch.LongTensor] = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get entity label and word representations using bi-encoder.

        Args:
            input_ids: Input token IDs of shape (B, L).
            attention_mask: Attention mask of shape (B, L).
            labels_embeds: Pre-computed label embeddings of shape (C, D).
            labels_input_ids: Label token IDs for encoding.
            labels_attention_mask: Attention mask for labels.
            text_lengths: Length of each text in batch.
            words_mask: Word boundary mask.
            **kwargs: Additional arguments for the encoder.

        Returns:
            Tuple containing:
                - labels_embeds: Label embeddings of shape (B, C, D).
                - labels_mask: Mask for labels of shape (B, C).
                - words_embedding: Word embeddings of shape (B, W, D).
                - mask: Mask for words of shape (B, W).
        """
        if labels_embeds is not None:
            token_embeds = self.token_rep_layer.encode_text(input_ids, attention_mask, **kwargs)
        else:
            token_embeds, labels_embeds = self.token_rep_layer(
                input_ids, attention_mask, labels_input_ids, labels_attention_mask, **kwargs
            )

        batch_size, _, embed_dim = token_embeds.shape  # batch size, seq length, embed dim
        max_text_length = text_lengths.max()

        words_embedding, mask = extract_word_embeddings(
            token_embeds, words_mask, attention_mask, batch_size, max_text_length, embed_dim, text_lengths
        )

        labels_embeds = labels_embeds.unsqueeze(0)
        labels_embeds = labels_embeds.expand(batch_size, -1, -1)
        labels_mask = torch.ones(labels_embeds.shape[:-1], dtype=attention_mask.dtype, device=attention_mask.device)

        labels_embeds = labels_embeds.to(words_embedding.dtype)

        if hasattr(self, "cross_fuser"):
            words_embedding, labels_embeds = self.features_enhancement(
                words_embedding, labels_embeds, text_mask=mask, labels_mask=labels_mask
            )

        return labels_embeds, labels_mask, words_embedding, mask


class BiEncoderSpanModel(BaseBiEncoderModel):
    """Span-based NER model using bi-encoder architecture.

    Attributes:
        span_rep_layer (SpanRepLayer): Layer for computing span representations.
        prompt_rep_layer (nn.Module): Projection layer for entity label embeddings.
    """

    def __init__(
        self, config: Any, from_pretrained: bool = False, cache_dir: Optional[Union[str, Path]] = None
    ) -> None:
        """Initialize the span-based bi-encoder model.

        Args:
            config: Model configuration object.
            from_pretrained: Whether to load from pretrained weights.
            cache_dir: Directory for caching pretrained models.
        """
        super().__init__(config, from_pretrained, cache_dir)
        self.span_rep_layer = SpanRepLayer(
            span_mode=config.span_mode,
            hidden_size=config.hidden_size,
            max_width=config.max_width,
            dropout=config.dropout,
        )

        self.prompt_rep_layer = create_projection_layer(config.hidden_size, config.dropout)

    def forward(
        self,
        input_ids: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        labels_embeds: Optional[torch.FloatTensor] = None,
        labels_input_ids: Optional[torch.FloatTensor] = None,
        labels_attention_mask: Optional[torch.LongTensor] = None,
        words_embedding: Optional[torch.FloatTensor] = None,
        mask: Optional[torch.LongTensor] = None,
        prompts_embedding: Optional[torch.FloatTensor] = None,
        prompts_embedding_mask: Optional[torch.LongTensor] = None,
        words_mask: Optional[torch.LongTensor] = None,
        text_lengths: Optional[torch.Tensor] = None,
        span_idx: Optional[torch.LongTensor] = None,
        span_mask: Optional[torch.LongTensor] = None,
        labels: Optional[torch.FloatTensor] = None,
        **kwargs: Any,
    ) -> GLiNERBaseOutput:
        """Forward pass through the bi-encoder span model.

        Args:
            input_ids: Input token IDs of shape (B, L).
            attention_mask: Attention mask of shape (B, L).
            labels_embeds: Pre-computed label embeddings of shape (C, D).
            labels_input_ids: Label token IDs for encoding.
            labels_attention_mask: Attention mask for labels.
            words_embedding: Pre-computed word embeddings.
            mask: Mask for words.
            prompts_embedding: Pre-computed entity label embeddings.
            prompts_embedding_mask: Mask for prompts.
            words_mask: Word boundary mask.
            text_lengths: Length of each text sequence.
            span_idx: Span indices of shape (B, L*K, 2).
            span_mask: Mask for valid spans of shape (B, L, K).
            labels: Ground truth labels of shape (B, L, K, C).
            **kwargs: Additional arguments.

        Returns:
            GLiNERBaseOutput containing logits, loss, and intermediate representations.
        """
        encoder_kwargs = {key: kwargs[key] for key in ("packing_config", "pair_attention_mask") if key in kwargs}

        prompts_embedding, prompts_embedding_mask, words_embedding, mask = self.get_representations(
            input_ids,
            attention_mask,
            labels_embeds,
            labels_input_ids,
            labels_attention_mask,
            text_lengths,
            words_mask,
            **encoder_kwargs,
        )

        target_W = span_idx.size(1) // self.config.max_width
        words_embedding, mask = self._fit_length(words_embedding, mask, target_W)

        span_idx = span_idx * span_mask.unsqueeze(-1)

        span_rep = self.span_rep_layer(words_embedding, span_idx)

        target_C = prompts_embedding.size(1)
        if labels is not None:
            target_C = max(target_C, labels.size(-1))

        prompts_embedding, prompts_embedding_mask = self._fit_length(
            prompts_embedding, prompts_embedding_mask, target_C
        )

        prompts_embedding = self.prompt_rep_layer(prompts_embedding)

        scores = torch.einsum("BLKD,BCD->BLKC", span_rep, prompts_embedding)

        loss = None
        if labels is not None:
            loss = self.loss(scores, labels, prompts_embedding_mask, span_mask, **kwargs)

        output = GLiNERBaseOutput(
            logits=scores,
            loss=loss,
            prompts_embedding=prompts_embedding,
            prompts_embedding_mask=prompts_embedding_mask,
            words_embedding=words_embedding,
            mask=mask,
        )
        return output

    def loss(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor,
        prompts_embedding_mask: torch.Tensor,
        mask_label: torch.Tensor,
        alpha: float = -1.0,
        gamma: float = 0.0,
        label_smoothing: float = 0.0,
        reduction: str = "sum",
        negatives: float = 1.0,
        masking: str = "none",
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute span classification loss for bi-encoder.

        Args:
            scores: Predicted scores of shape (B, L, K, C).
            labels: Ground truth labels of shape (B, L, K, C).
            prompts_embedding_mask: Mask for valid entity types of shape (B, C).
            mask_label: Mask for valid spans of shape (B, L, K).
            alpha: Focal loss alpha parameter.
            gamma: Focal loss gamma parameter.
            label_smoothing: Label smoothing factor.
            reduction: Loss reduction method ('sum' or 'mean').
            negatives: Negative sampling probability.
            masking: Masking strategy for negative sampling.
            **kwargs: Additional arguments.

        Returns:
            Scalar loss tensor.
        """
        batch_size = scores.shape[0]
        num_classes = prompts_embedding_mask.shape[-1]

        # Reshape scores and labels to match the expected shape
        BS, _, _, CL = scores.shape

        scores = scores.view(BS, -1, CL)
        labels = labels.view(BS, -1, CL)

        all_losses = self._loss(scores, labels, alpha, gamma, label_smoothing, negatives=negatives, masking=masking)

        masked_loss = all_losses.view(batch_size, -1, num_classes) * prompts_embedding_mask.unsqueeze(1)
        all_losses = masked_loss.view(-1, num_classes)

        mask_label = mask_label.view(-1, 1)

        all_losses = all_losses * mask_label.float()

        if reduction == "mean":
            loss = all_losses.mean()
        elif reduction == "sum":
            loss = all_losses.sum()
        else:
            warnings.warn(
                f"Invalid Value for config 'loss_reduction': '{reduction}' \n Supported reduction modes:"
                f" 'none', 'mean', 'sum'. It will be used 'sum' instead.",
                stacklevel=2,
            )
            loss = all_losses.sum()

        return loss


class BiEncoderTokenModel(BaseBiEncoderModel):
    """Token-based NER model using bi-encoder architecture.

    Attributes:
        scorer (Scorer): Scoring layer for computing token-label compatibility.
    """

    def __init__(
        self, config: Any, from_pretrained: bool = False, cache_dir: Optional[Union[str, Path]] = None
    ) -> None:
        """Initialize the token-based bi-encoder model.

        Args:
            config: Model configuration object.
            from_pretrained: Whether to load from pretrained weights.
            cache_dir: Directory for caching pretrained models.
        """
        super().__init__(config, from_pretrained, cache_dir)
        self.scorer = Scorer(config.hidden_size, config.dropout)

    def forward(
        self,
        input_ids: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        labels_embeds: Optional[torch.FloatTensor] = None,
        labels_input_ids: Optional[torch.FloatTensor] = None,
        labels_attention_mask: Optional[torch.LongTensor] = None,
        words_embedding: Optional[torch.FloatTensor] = None,
        mask: Optional[torch.LongTensor] = None,
        prompts_embedding: Optional[torch.FloatTensor] = None,
        prompts_embedding_mask: Optional[torch.LongTensor] = None,
        words_mask: Optional[torch.LongTensor] = None,
        text_lengths: Optional[torch.Tensor] = None,
        labels: Optional[torch.FloatTensor] = None,
        **kwargs: Any,
    ) -> GLiNERBaseOutput:
        """Forward pass through the bi-encoder token model.

        Args:
            input_ids: Input token IDs of shape (B, L).
            attention_mask: Attention mask of shape (B, L).
            labels_embeds: Pre-computed label embeddings of shape (C, D).
            labels_input_ids: Label token IDs for encoding.
            labels_attention_mask: Attention mask for labels.
            words_embedding: Pre-computed word embeddings.
            mask: Mask for words.
            prompts_embedding: Pre-computed entity label embeddings.
            prompts_embedding_mask: Mask for prompts.
            words_mask: Word boundary mask.
            text_lengths: Length of each text sequence.
            labels: Ground truth labels of shape (B, W, C).
            **kwargs: Additional arguments.

        Returns:
            GLiNERBaseOutput containing logits, loss, and intermediate representations.
        """
        encoder_kwargs = {key: kwargs[key] for key in ("packing_config", "pair_attention_mask") if key in kwargs}

        prompts_embedding, prompts_embedding_mask, words_embedding, mask = self.get_representations(
            input_ids,
            attention_mask,
            labels_embeds,
            labels_input_ids,
            labels_attention_mask,
            text_lengths,
            words_mask,
            **encoder_kwargs,
        )

        if labels is not None:
            target_W = labels.shape[1]
            words_embedding, mask = self._fit_length(words_embedding, mask, target_W)

            target_C = prompts_embedding.size(1)
            if labels is not None:
                target_C = max(target_C, labels.size(-2))

            prompts_embedding, prompts_embedding_mask = self._fit_length(
                prompts_embedding, prompts_embedding_mask, target_C
            )

        scores = self.scorer(words_embedding, prompts_embedding)

        loss = None
        if labels is not None:
            loss = self.loss(scores, labels, prompts_embedding_mask, mask, **kwargs)

        output = GLiNERBaseOutput(
            logits=scores,
            loss=loss,
            prompts_embedding=prompts_embedding,
            prompts_embedding_mask=prompts_embedding_mask,
            words_embedding=words_embedding,
            mask=mask,
        )
        return output

    def loss(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor,
        prompts_embedding_mask: torch.Tensor,
        mask: torch.Tensor,
        alpha: float = -1.0,
        gamma: float = 0.0,
        prob_margin: float = 0.0,
        label_smoothing: float = 0.0,
        reduction: str = "sum",
        negatives: float = 1.0,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute token classification loss for bi-encoder.

        Args:
            scores: Predicted scores of shape (B, W, C).
            labels: Ground truth labels of shape (B, W, C).
            prompts_embedding_mask: Mask for valid entity types of shape (B, C).
            mask: Mask for valid tokens of shape (B, W).
            alpha: Focal loss alpha parameter.
            gamma: Focal loss gamma parameter.
            prob_margin: Margin for probability adjustment.
            label_smoothing: Label smoothing factor.
            reduction: Loss reduction method ('sum' or 'mean').
            negatives: Negative sampling probability.
            **kwargs: Additional arguments.

        Returns:
            Scalar loss tensor.
        """
        all_losses = self._loss(scores, labels, alpha, gamma, prob_margin, label_smoothing, negatives)

        all_losses = all_losses * (mask.unsqueeze(-1) * prompts_embedding_mask.unsqueeze(1)).unsqueeze(-1)

        if reduction == "mean":
            loss = all_losses.mean()
        elif reduction == "sum":
            loss = all_losses.sum()
        else:
            warnings.warn(
                f"Invalid Value for config 'loss_reduction': '{reduction}' \n Supported reduction modes:"
                f" 'none', 'mean', 'sum'. It will be used 'sum' instead.",
                stacklevel=2,
            )
            loss = all_losses.sum()
        return loss


class UniEncoderSpanDecoderModel(UniEncoderSpanModel):
    """Span-based model with decoder for generating entity type labels.

    This model extends the span-based approach by adding a decoder that can
    generate entity type labels as sequences, enabling more flexible entity typing.

    Attributes:
        decoder (Decoder): Decoder module for label generation.
        _enc2dec_proj (Optional[nn.Module]): Projection layer if encoder and decoder
            dimensions differ.
    """

    def __init__(
        self, config: Any, from_pretrained: bool = False, cache_dir: Optional[Union[str, Path]] = None
    ) -> None:
        """Initialize the span-decoder model.

        Args:
            config: Model configuration object.
            from_pretrained: Whether to load from pretrained weights.
            cache_dir: Directory for caching pretrained models.
        """
        super().__init__(config, from_pretrained, cache_dir)
        self.decoder = Decoder(config, from_pretrained, cache_dir=cache_dir)
        if self.config.hidden_size != self.decoder.decoder_hidden_size:
            self._enc2dec_proj = create_projection_layer(
                self.config.hidden_size,
                self.config.dropout,
                self.decoder.decoder_hidden_size,
            )

    def select_decoder_embedding(
        self,
        representations: torch.FloatTensor,
        rep_mask: torch.LongTensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Select and pack valid representations for decoder input.

        Keeps only representations where mask == 1 and optionally projects them
        to the decoder hidden size.

        Args:
            representations: Tensor of shape (B, N, D) containing embeddings.
            rep_mask: Tensor of shape (B, N) where 1 indicates valid position.

        Returns:
            Tuple containing:
                - target_rep: FloatTensor of shape (B, M, D) with kept representations.
                - target_mask: LongTensor of shape (B, M) with 1 for valid, 0 for pad.
                - sel_idx: LongTensor of shape (B, M) with original column indices
                  (-1 for padding positions).
        """
        B, _, D = representations.shape
        lengths = rep_mask.sum(dim=-1)
        max_len = lengths.max().item()

        target_rep = representations.new_zeros(B, max_len, D)
        target_mask = rep_mask.new_zeros(B, max_len)
        sel_idx = rep_mask.new_full((B, max_len), -1)

        keep = rep_mask.bool()
        if keep.any():
            new_col_idx = (rep_mask.cumsum(dim=1) - 1)[keep]
            batch_idx, old_col_idx = torch.where(keep)

            target_rep[batch_idx, new_col_idx] = representations[batch_idx, old_col_idx]
            target_mask[batch_idx, new_col_idx] = 1
            sel_idx[batch_idx, new_col_idx] = old_col_idx

        return target_rep, target_mask, sel_idx

    def get_raw_decoder_inputs(
        self, representations: torch.Tensor, rep_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract valid span tokens for decoder input.

        Args:
            representations: Span representations of shape (B, S, T, D).
            rep_mask: Mask of shape (B, S, T).

        Returns:
            Tuple containing:
                - span_tokens: Valid span tokens of shape (M, T, D).
                - span_tokens_mask: Mask for span tokens of shape (M, T).
        """
        B, S, T, D = representations.shape
        BN = B * S
        valid_spans = rep_mask.any(-1)
        keep_mask = valid_spans.view(-1)

        if not keep_mask.any():
            empty = representations.new_empty(0, 0, D)
            return empty, representations.new_empty(0, 0, dtype=rep_mask.dtype)

        keep_idx = keep_mask.nonzero(as_tuple=False).squeeze(1)

        span_tokens = representations.view(BN, T, D)[keep_idx]
        span_tokens_mask = rep_mask.view(BN, T)[keep_idx]
        return span_tokens, span_tokens_mask

    def decode_labels(
        self,
        decoder_embedding: Optional[torch.FloatTensor] = None,
        decoder_embedding_mask: Optional[torch.LongTensor] = None,
        decoder_labels_ids: Optional[torch.FloatTensor] = None,
        decoder_labels_mask: Optional[torch.LongTensor] = None,
        decoder_labels: Optional[torch.FloatTensor] = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decode labels using the decoder in teacher forcing mode.

        Args:
            decoder_embedding: Span token embeddings of shape (B, N, T, D).
            decoder_embedding_mask: Mask for span tokens of shape (B, N, T).
            decoder_labels_ids: Label token IDs of shape (M, L).
            decoder_labels_mask: Mask for labels of shape (M, L).
            decoder_labels: Ground truth labels for loss computation of shape (M, L).
            **kwargs: Additional arguments.

        Returns:
            Tuple containing:
                - loss: Cross-entropy loss scalar.
                - decoder_outputs: Decoder logits of shape (M, S+L-1, V).
        """
        span_tokens, span_tokens_mask = self.get_raw_decoder_inputs(decoder_embedding, decoder_embedding_mask)

        label_embeds = self.decoder.ids_to_embeds(decoder_labels_ids)

        decoder_inputs = torch.cat([span_tokens, label_embeds[:, :-1, :]], dim=1)

        attn_inputs = torch.cat([span_tokens_mask.to(decoder_labels_mask.dtype), decoder_labels_mask[:, :-1]], dim=1)

        decoder_outputs = self.decoder(inputs_embeds=decoder_inputs, attention_mask=attn_inputs)

        blank_for_spans = torch.full(
            (decoder_labels.size(0), span_tokens.size(1)),
            -100,
            dtype=decoder_labels.dtype,
            device=decoder_labels.device,
        )

        targets = torch.cat([blank_for_spans, decoder_labels], dim=1)

        loss = cross_entropy_loss(decoder_outputs, targets[:, 1:])

        return (loss, decoder_outputs)

    def generate_labels(
        self,
        decoder_embedding: Optional[torch.FloatTensor] = None,
        decoder_embedding_mask: Optional[torch.LongTensor] = None,
        max_new_tokens: int = 32,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        temperature: float = 1.0,
        do_sample: bool = False,
        num_return_sequences: int = 1,
        labels_trie: Optional[Any] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Generate entity type labels from decoder embeddings.

        Args:
            decoder_embedding: Span token embeddings of shape (B, N, D).
            decoder_embedding_mask: Mask for span tokens of shape (B, N).
            max_new_tokens: Maximum number of tokens to generate.
            eos_token_id: End-of-sequence token ID.
            pad_token_id: Padding token ID.
            temperature: Sampling temperature.
            do_sample: Whether to use sampling instead of greedy decoding.
            num_return_sequences: Number of sequences to generate per input.
            labels_trie: Optional trie for constrained decoding.
            **kwargs: Additional generation arguments.

        Returns:
            Generated token IDs of shape (M, L).
        """
        span_tokens, _ = self.get_raw_decoder_inputs(decoder_embedding, decoder_embedding_mask)
        results = self.decoder.generate_from_embeds(
            span_tokens,
            attention_mask=None,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            temperature=temperature,
            do_sample=do_sample,
            num_return_sequences=num_return_sequences,
            labels_trie=labels_trie,
            **kwargs,
        )
        return results

    def select_span_decoder_embedding(
        self,
        prompts_embedding: torch.Tensor,
        prompts_embedding_mask: torch.Tensor,
        span_rep: torch.Tensor,
        span_scores: torch.Tensor,
        span_mask: torch.Tensor,
        decoder_text_embeds: Optional[torch.Tensor] = None,
        decoder_words_mask: Optional[torch.Tensor] = None,
        span_labels: Optional[torch.Tensor] = None,
        threshold: float = 0.5,
        top_k: Optional[int] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        decoder_labels_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Select span embeddings for decoder input based on predictions or labels.

        This method selects which spans to provide to the decoder, either based on
        ground truth labels (during training) or model predictions (during inference).
        It can operate in two modes:
        1. "prompt" mode: Uses entity type embeddings as decoder input.
        2. "span" mode: Uses contextualized tokens within each span as decoder input.

        Args:
            prompts_embedding: Entity type embeddings of shape (B, C, D).
            prompts_embedding_mask: Mask for prompts of shape (B, C).
            span_rep: Span representations of shape (B, L, K, D).
            span_scores: Span classification scores of shape (B, L, K, C).
            span_mask: Mask for valid spans of shape (B, L, K).
            decoder_text_embeds: Text embeddings for span mode of shape (B, T, D).
            decoder_words_mask: Word position mask of shape (B, T).
            span_labels: Ground truth labels of shape (B, L, K, C).
            threshold: Confidence threshold for selecting spans.
            top_k: Optional limit on number of spans to select.
            decoder_input_ids: Debugging parameter for input IDs.
            decoder_labels_ids: Debugging parameter for label IDs.

        Returns:
            Tuple containing:
                - span_rep_kept: Selected span embeddings for decoder of shape (B, S, T, D)
                  or None if no valid spans.
                - span_msk: Mask for selected spans of shape (B, S, T) or None.
                - span_sel_idx: Original indices of selected spans of shape (B, S) or None.
        """
        if self.config.decoder_mode == "prompt":
            return self.select_decoder_embedding(prompts_embedding, prompts_embedding_mask)[:3]

        B, L, K, D = span_rep.shape
        flat_rep = span_rep.view(B, L * K, D)
        flat_mask = span_mask.view(B, L * K)

        if span_labels is not None:
            flat_prob = span_labels.max(-1).values.view(B, L * K)
            keep = (flat_prob == 1) & flat_mask.bool()
        else:
            flat_prob = torch.sigmoid(span_scores).max(-1).values.view(B, L * K)
            keep = (flat_prob > threshold) & flat_mask.bool()

        if top_k:
            sel_scores = flat_prob.masked_fill(~keep, -1.0)
            top_idx = sel_scores.topk(k=min(top_k, sel_scores.size(1)), dim=1).indices
            keep.zero_()
            keep.scatter_(1, top_idx, True)

        span_rep_kept, span_msk, span_sel_idx = self.select_decoder_embedding(flat_rep, keep.long())

        if hasattr(self, "_enc2dec_proj"):
            span_rep_kept = self._enc2dec_proj(span_rep_kept)
        span_rep_kept = span_rep_kept.unsqueeze(2)
        span_msk = span_msk.unsqueeze(-1)

        if decoder_text_embeds is None or decoder_words_mask is None:
            return span_rep_kept, span_msk.unsqueeze(-1), span_sel_idx

        if span_rep_kept.numel() == 0:
            return None, None, None

        decoder_text_embeds = decoder_text_embeds.to(dtype=span_rep_kept.dtype)

        S = span_rep_kept.shape[1]
        dec_D = span_rep_kept.shape[-1]
        span_start = span_sel_idx // self.config.max_width + 1
        span_end = span_sel_idx % self.config.max_width + span_start

        token_in_span = (decoder_words_mask.unsqueeze(1) >= span_start.unsqueeze(-1)) & (
            decoder_words_mask.unsqueeze(1) <= span_end.unsqueeze(-1)
        )

        tokens_per_span = token_in_span.sum(-1)
        max_tokens = int(tokens_per_span.max())

        span_rep_new = span_rep_kept.new_zeros(B, S, max_tokens + 1, dec_D)
        span_rep_mask = torch.zeros(B, S, max_tokens + 1, dtype=torch.bool, device=decoder_text_embeds.device)

        left_offset = (max_tokens + 1 - tokens_per_span).clamp(min=0)
        pos_in_span = (token_in_span.cumsum(-1) - 1).masked_fill(~token_in_span, 0)
        pos_in_span = pos_in_span + left_offset.unsqueeze(-1)

        b_idx, s_idx, tok_idx = torch.where(token_in_span)
        span_rep_new[b_idx, s_idx, pos_in_span[b_idx, s_idx, tok_idx]] = decoder_text_embeds[b_idx, tok_idx]
        span_rep_mask[b_idx, s_idx, pos_in_span[b_idx, s_idx, tok_idx]] = True
        kept_pos = (left_offset - 1).clamp(min=0)

        b_flat = torch.arange(B, device=decoder_text_embeds.device).view(-1, 1).expand(B, S).reshape(-1)
        s_flat = torch.arange(S, device=decoder_text_embeds.device).view(1, -1).expand(B, S).reshape(-1)
        t_flat = kept_pos.reshape(-1)

        span_rep_new[b_flat, s_flat, t_flat] = span_rep_kept.reshape(-1, dec_D)
        span_rep_mask[b_flat, s_flat, t_flat] = True
        span_rep_mask = span_rep_mask & span_msk.bool()
        return span_rep_new, span_rep_mask, span_sel_idx

    def forward(
        self,
        input_ids: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.FloatTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        decoder_labels_ids: Optional[torch.FloatTensor] = None,
        decoder_labels_mask: Optional[torch.LongTensor] = None,
        decoder_words_mask: Optional[torch.LongTensor] = None,
        words_embedding: Optional[torch.FloatTensor] = None,
        mask: Optional[torch.LongTensor] = None,
        prompts_embedding: Optional[torch.FloatTensor] = None,
        prompts_embedding_mask: Optional[torch.LongTensor] = None,
        words_mask: Optional[torch.LongTensor] = None,
        text_lengths: Optional[torch.Tensor] = None,
        span_idx: Optional[torch.LongTensor] = None,
        span_mask: Optional[torch.LongTensor] = None,
        labels: Optional[torch.FloatTensor] = None,
        decoder_labels: Optional[torch.FloatTensor] = None,
        threshold: Optional[float] = 0.5,
        **kwargs: Any,
    ) -> GLiNERDecoderOutput:
        """Forward pass through the span-decoder model.

        Args:
            input_ids: Input token IDs of shape (B, L).
            attention_mask: Attention mask of shape (B, L).
            decoder_input_ids: Decoder input IDs for span mode.
            decoder_attention_mask: Decoder attention mask.
            decoder_labels_ids: Label token IDs for decoding of shape (M, L).
            decoder_labels_mask: Mask for decoder labels of shape (M, L).
            decoder_words_mask: Word position mask for span mode.
            words_embedding: Pre-computed word embeddings.
            mask: Mask for words.
            prompts_embedding: Pre-computed entity type embeddings.
            prompts_embedding_mask: Mask for prompts.
            words_mask: Word boundary mask.
            text_lengths: Length of each text sequence.
            span_idx: Span indices of shape (B, L*K, 2).
            span_mask: Mask for valid spans of shape (B, L, K).
            labels: Ground truth span labels of shape (B, L, K, C).
            decoder_labels: Ground truth decoder labels of shape (M, L).
            threshold: Confidence threshold for span selection.
            **kwargs: Additional arguments.

        Returns:
            GLiNERDecoderOutput containing logits, losses, and decoder information.
        """
        encoder_kwargs = {key: kwargs[key] for key in ("packing_config", "pair_attention_mask") if key in kwargs}

        prompts_embedding, prompts_embedding_mask, words_embedding, mask = self.get_representations(
            input_ids, attention_mask, text_lengths, words_mask, **encoder_kwargs
        )

        target_W = span_idx.size(1) // self.config.max_width
        words_embedding, mask = self._fit_length(words_embedding, mask, target_W)

        span_idx = span_idx * span_mask.unsqueeze(-1)

        span_rep = self.span_rep_layer(words_embedding, span_idx)

        target_C = prompts_embedding.size(1)
        if labels is not None:
            target_C = max(target_C, labels.size(-1))

        prompts_embedding, prompts_embedding_mask = self._fit_length(
            prompts_embedding, prompts_embedding_mask, target_C
        )

        prompts_embedding = self.prompt_rep_layer(prompts_embedding)

        scores = torch.einsum("BLKD,BCD->BLKC", span_rep, prompts_embedding)

        decoder_embedding = decoder_mask = decoder_loss = decoder_span_idx = None
        if hasattr(self, "decoder"):
            if self.config.decoder_mode == "span":
                decoder_text_embeds = self.decoder.ids_to_embeds(decoder_input_ids)
            else:
                decoder_text_embeds = None

            decoder_embedding, decoder_mask, decoder_span_idx = self.select_span_decoder_embedding(
                prompts_embedding,
                prompts_embedding_mask,
                span_rep,
                scores,
                span_mask,
                decoder_text_embeds=decoder_text_embeds,
                decoder_words_mask=decoder_words_mask,
                span_labels=labels,
                threshold=threshold,
                decoder_input_ids=decoder_input_ids,
                decoder_labels_ids=decoder_labels_ids,
            )

            if decoder_labels is not None:
                decoder_loss, _ = self.decode_labels(
                    decoder_embedding, decoder_mask, decoder_labels_ids, decoder_labels_mask, decoder_labels
                )

        loss = None
        if labels is not None:
            loss = self.loss(scores, labels, prompts_embedding_mask, span_mask, decoder_loss=decoder_loss, **kwargs)

        output = GLiNERDecoderOutput(
            logits=scores,
            loss=loss,
            decoder_loss=decoder_loss,
            prompts_embedding=prompts_embedding,
            prompts_embedding_mask=prompts_embedding_mask,
            decoder_embedding=decoder_embedding,
            decoder_embedding_mask=decoder_mask,
            decoder_span_idx=decoder_span_idx,
            words_embedding=words_embedding,
            mask=mask,
        )
        return output

    def loss(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor,
        prompts_embedding_mask: torch.Tensor,
        mask_label: torch.Tensor,
        alpha: float = -1.0,
        gamma: float = 0.0,
        label_smoothing: float = 0.0,
        reduction: str = "sum",
        negatives: float = 1.0,
        masking: str = "none",
        decoder_loss: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute combined loss for span classification and decoder.

        Args:
            scores: Predicted span scores of shape (B, L, K, C).
            labels: Ground truth span labels of shape (B, L, K, C).
            prompts_embedding_mask: Mask for valid entity types of shape (B, C).
            mask_label: Mask for valid spans of shape (B, L, K).
            alpha: Focal loss alpha parameter.
            gamma: Focal loss gamma parameter.
            label_smoothing: Label smoothing factor.
            reduction: Loss reduction method ('sum' or 'mean').
            negatives: Negative sampling probability.
            masking: Masking strategy for negative sampling.
            decoder_loss: Optional decoder loss to combine with span loss.
            **kwargs: Additional arguments.

        Returns:
            Scalar combined loss tensor.
        """
        batch_size = scores.shape[0]
        num_classes = prompts_embedding_mask.shape[-1]

        BS, _, _, CL = scores.shape

        scores = scores.view(BS, -1, CL)
        labels = labels.view(BS, -1, CL)

        all_losses = self._loss(scores, labels, alpha, gamma, label_smoothing, negatives=negatives, masking=masking)

        masked_loss = all_losses.view(batch_size, -1, num_classes) * prompts_embedding_mask.unsqueeze(1)
        all_losses = masked_loss.view(-1, num_classes)

        mask_label = mask_label.view(-1, 1)

        all_losses = all_losses * mask_label.float()

        if reduction == "mean":
            loss = all_losses.mean()
        elif reduction == "sum":
            loss = all_losses.sum()
        else:
            warnings.warn(
                f"Invalid Value for config 'loss_reduction': '{reduction}' \n Supported reduction modes:"
                f" 'none', 'mean', 'sum'. It will be used 'sum' instead.",
                stacklevel=2,
            )
            loss = all_losses.sum()

        if decoder_loss is not None:
            loss = decoder_loss * self.config.decoder_loss_coef + loss * self.config.span_loss_coef

        return loss


class UniEncoderSpanRelexModel(UniEncoderSpanModel):
    """Span-based NER model with relation extraction capabilities.

    This model extends span-based NER to also extract relations between
    identified entities, predicting both entity types and relation types
    in a joint model.

    Attributes:
        relations_rep_layer (Optional[RelationsRepLayer]): Layer for computing
            pairwise entity relations (adjacency matrix).
        triples_score_layer (Optional[TriplesScoreLayer]): Layer for scoring
            (head, relation, tail) triples.
        pair_rep_layer (Optional[nn.Module]): Alternative layer for relation
            scoring via concatenation.
    """

    def __init__(
        self, config: Any, from_pretrained: bool = False, cache_dir: Optional[Union[str, Path]] = None
    ) -> None:
        """Initialize the span-based relation extraction model.

        Args:
            config: Model configuration object.
            from_pretrained: Whether to load from pretrained weights.
            cache_dir: Directory for caching pretrained models.
        """
        super().__init__(config, from_pretrained, cache_dir)

        if config.relations_layer is not None:
            self.relations_rep_layer = RelationsRepLayer(
                in_dim=config.hidden_size, relation_mode=config.relations_layer
            )

            if config.triples_layer is not None:
                self.triples_score_layer = TriplesScoreLayer(config.triples_layer)
            else:
                self.pair_rep_layer = create_projection_layer(
                    config.hidden_size * 2, config.dropout, config.hidden_size
                )

    def select_span_target_embedding(
        self,
        span_rep: torch.FloatTensor,
        span_scores: torch.FloatTensor,
        span_mask: torch.LongTensor,
        span_labels: Optional[torch.FloatTensor] = None,
        threshold: float = 0.5,
        top_k: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Select entity spans for relation extraction.

        Filters spans based on entity classification scores or ground truth labels,
        keeping only high-confidence or positive entity spans for relation modeling.

        Args:
            span_rep: Span representations of shape (B, L, K, D).
            span_scores: Span classification scores of shape (B, L, K, C).
            span_mask: Mask for valid spans of shape (B, L, K).
            span_labels: Optional ground truth labels of shape (B, L, K, C).
            threshold: Confidence threshold for selecting spans.
            top_k: Optional limit on number of spans to select.

        Returns:
            Tuple containing:
                - target_rep: Selected span representations of shape (B, E, D).
                - target_mask: Mask for selected spans of shape (B, E).
        """
        B, L, K, D = span_rep.shape

        span_rep_flat = span_rep.view(B, L * K, D)
        span_mask_flat = span_mask.view(B, L * K)

        if span_labels is not None:
            span_prob_flat = span_labels.max(dim=-1).values.view(B, L * K)
            keep = (span_prob_flat == 1).bool()
        else:
            span_prob_flat = torch.sigmoid(span_scores).max(dim=-1).values.view(B, L * K)
            keep = (span_prob_flat > threshold) & span_mask_flat.bool()

        if top_k is not None and top_k > 0:
            sel_scores = span_prob_flat.masked_fill(~keep, -1.0)
            top_idx = sel_scores.topk(k=min(top_k, sel_scores.size(1)), dim=1).indices
            keep = torch.zeros_like(keep)
            keep.scatter_(1, top_idx, True)

        rep_mask = keep.long()

        target_rep, target_mask = self.select_target_embedding(representations=span_rep_flat, rep_mask=rep_mask)

        return target_rep, target_mask

    def select_target_embedding(
        self, representations: Optional[torch.FloatTensor] = None, rep_mask: Optional[torch.LongTensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pack valid representations by removing masked positions.

        Args:
            representations: Tensor of shape (B, N, D).
            rep_mask: Mask of shape (B, N) where 1 indicates valid position.

        Returns:
            Tuple containing:
                - target_rep: Packed representations of shape (B, M, D).
                - target_mask: Packed mask of shape (B, M).
        """
        B, N, D = representations.shape
        lengths = rep_mask.sum(dim=-1)
        max_len = lengths.max().item()

        if max_len != N:
            target_rep = representations.new_zeros(B, max_len, D)
            target_mask = rep_mask.new_zeros(B, max_len)

            new_col_idx = rep_mask.cumsum(dim=1) - 1
            keep = rep_mask.bool()

            batch_idx, old_col_idx = torch.where(keep)
            new_col_idx = new_col_idx[keep]

            target_rep[batch_idx, new_col_idx] = representations[batch_idx, old_col_idx]
            target_mask[batch_idx, new_col_idx] = 1
        else:
            target_rep = representations
            target_mask = rep_mask

        return target_rep, target_mask

    def forward(
        self,
        input_ids: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        words_embedding: Optional[torch.FloatTensor] = None,
        mask: Optional[torch.LongTensor] = None,
        prompts_embedding: Optional[torch.FloatTensor] = None,
        prompts_embedding_mask: Optional[torch.LongTensor] = None,
        words_mask: Optional[torch.LongTensor] = None,
        text_lengths: Optional[torch.Tensor] = None,
        span_idx: Optional[torch.LongTensor] = None,
        span_mask: Optional[torch.LongTensor] = None,
        labels: Optional[torch.FloatTensor] = None,
        adj_matrix: Optional[torch.FloatTensor] = None,
        rel_matrix: Optional[torch.FloatTensor] = None,
        threshold: Optional[float] = 0.5,
        adjacency_threshold: Optional[float] = 0.5,
        **kwargs: Any,
    ) -> GLiNERRelexOutput:
        """Forward pass through the relation extraction model.

        Args:
            input_ids: Input token IDs of shape (B, L).
            attention_mask: Attention mask of shape (B, L).
            words_embedding: Pre-computed word embeddings.
            mask: Mask for words.
            prompts_embedding: Pre-computed entity type embeddings.
            prompts_embedding_mask: Mask for entity types.
            words_mask: Word boundary mask.
            text_lengths: Length of each text sequence.
            span_idx: Span indices of shape (B, L*K, 2).
            span_mask: Mask for valid spans of shape (B, L, K).
            labels: Ground truth entity labels of shape (B, L, K, C).
            adj_matrix: Ground truth adjacency matrix of shape (B, E, E).
            rel_matrix: Ground truth relation labels of shape (B, N, C_rel).
            threshold: Confidence threshold for entity selection.
            adjacency_threshold: Threshold for relation adjacency.
            **kwargs: Additional arguments.

        Returns:
            GLiNERRelexOutput containing entity and relation predictions.
        """
        encoder_kwargs = {key: kwargs[key] for key in ("packing_config", "pair_attention_mask") if key in kwargs}

        token_embeds = self.token_rep_layer(input_ids, attention_mask, **encoder_kwargs)

        prompts_embedding, prompts_embedding_mask, words_embedding, mask = (
            self._extract_prompt_features_and_word_embeddings(
                token_embeds, input_ids, attention_mask, text_lengths, words_mask
            )
        )

        if hasattr(self, "rnn"):
            words_embedding = self.rnn(words_embedding, mask)

        target_W = span_idx.size(1) // self.config.max_width
        words_embedding, mask = self._fit_length(words_embedding, mask, target_W)

        span_idx = span_idx * span_mask.unsqueeze(-1)

        span_rep = self.span_rep_layer(words_embedding, span_idx)

        target_C = prompts_embedding.size(1)
        if labels is not None:
            target_C = max(target_C, labels.size(-1))

        prompts_embedding, prompts_embedding_mask = self._fit_length(
            prompts_embedding, prompts_embedding_mask, target_C
        )

        prompts_embedding = self.prompt_rep_layer(prompts_embedding)
        batch_size, _, embed_dim = prompts_embedding.shape
        scores = torch.einsum("BLKD,BCD->BLKC", span_rep, prompts_embedding)

        pair_idx, pair_mask, pair_scores = None, None, None
        rel_prompts_embedding_mask = None
        pred_adj_matrix = None

        if hasattr(self, "relations_rep_layer"):
            target_span_rep, target_span_mask = self.select_span_target_embedding(
                span_rep, scores, span_mask, labels, threshold
            )
            pred_adj_matrix = self.relations_rep_layer(target_span_rep, target_span_mask)

            rel_prompts_embedding, rel_prompts_embedding_mask = extract_prompt_features(
                self.config.rel_token_index,
                token_embeds,
                input_ids,
                attention_mask,
                batch_size,
                embed_dim,
                self.config.embed_rel_token,
            )

            B, _, D = target_span_rep.shape
            C_rel = rel_prompts_embedding.size(1)

            adj_for_selection = adj_matrix if (labels is not None and adj_matrix is not None) else pred_adj_matrix

            pair_idx, pair_mask, head_rep_selected, tail_rep_selected = build_entity_pairs(
                adj_for_selection, target_span_rep, threshold=adjacency_threshold
            )

            N = head_rep_selected.size(1)

            if hasattr(self, "pair_rep_layer"):
                pair_rep = torch.cat((head_rep_selected, tail_rep_selected), dim=-1)
                pair_rep = self.pair_rep_layer(pair_rep)
                pair_scores = torch.einsum("BND,BCD->BNC", pair_rep, rel_prompts_embedding)

            elif hasattr(self, "triples_score_layer"):
                h = head_rep_selected.unsqueeze(2).expand(B, N, C_rel, D)
                t = tail_rep_selected.unsqueeze(2).expand(B, N, C_rel, D)
                r = rel_prompts_embedding.unsqueeze(1).expand(B, N, C_rel, D)

                h_flat = h.reshape(B * N * C_rel, D)
                t_flat = t.reshape(B * N * C_rel, D)
                r_flat = r.reshape(B * N * C_rel, D)

                triple_scores_flat = self.triples_score_layer(h_flat, r_flat, t_flat)
                pair_scores = triple_scores_flat.view(B, N, C_rel)

        loss = None
        if labels is not None:
            loss = self.loss(scores, labels, prompts_embedding_mask, span_mask, **kwargs)

            if adj_matrix is not None and rel_matrix is not None and hasattr(self, "relations_rep_layer"):
                adj_mask = target_span_mask.float().unsqueeze(1) * target_span_mask.float().unsqueeze(2)
                adj_loss = self.adj_loss(pred_adj_matrix, adj_matrix, adj_mask, **kwargs)

                rel_labels_selected = rel_matrix
                rel_mask_selected = pair_mask.unsqueeze(-1).expand(B, N, C_rel)
                class_mask = rel_prompts_embedding_mask.unsqueeze(1).expand(B, N, C_rel)

                rel_loss = self.rel_loss(pair_scores, rel_labels_selected, rel_mask_selected, class_mask, **kwargs)

                loss = (
                    loss * self.config.span_loss_coef
                    + adj_loss * self.config.adjacency_loss_coef
                    + rel_loss * self.config.relation_loss_coef
                )

        output = GLiNERRelexOutput(
            logits=scores,
            loss=loss,
            prompts_embedding=prompts_embedding,
            prompts_embedding_mask=prompts_embedding_mask,
            words_embedding=words_embedding,
            mask=mask,
            rel_idx=pair_idx,
            rel_logits=pair_scores,
            rel_mask=pair_mask,
        )
        return output

    def adj_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        adj_mask: torch.Tensor,
        alpha: float = -1.0,
        gamma: float = 0.0,
        prob_margin: float = 0.0,
        label_smoothing: float = 0.0,
        reduction: str = "sum",
        masking: str = "span",
        negatives: float = 1.0,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute adjacency matrix loss for entity pair relations.

        Args:
            logits: Predicted adjacency scores of shape (B, E, E).
            labels: Ground truth adjacency of shape (B, E, E).
            adj_mask: Mask for valid entity pairs of shape (B, E, E).
            alpha: Focal loss alpha parameter.
            gamma: Focal loss gamma parameter.
            prob_margin: Margin for probability adjustment.
            label_smoothing: Label smoothing factor.
            reduction: Loss reduction method ('sum' or 'mean').
            masking: Masking strategy for negative sampling.
            negatives: Negative sampling probability.
            **kwargs: Additional arguments.

        Returns:
            Scalar adjacency loss tensor.
        """
        B = logits.size(0)

        logits = logits.unsqueeze(-1).view(B, -1, 1)
        labels = labels.unsqueeze(-1).view(B, -1, 1)

        all_losses = self._loss(
            logits,
            labels,
            alpha,
            gamma,
            prob_margin,
            label_smoothing,
            negatives=negatives,
            masking=masking,
            normalize_prob=False,
        )

        masked_loss = all_losses * adj_mask.unsqueeze(-1).view(B, -1, 1)

        if reduction == "mean":
            num_valid = adj_mask.sum()
            loss = masked_loss.sum() / num_valid if num_valid > 0 else torch.tensor(0.0, device=logits.device)
        elif reduction == "sum":
            loss = masked_loss.sum()
        else:
            warnings.warn(
                f"Invalid Value for config 'loss_reduction': '{reduction}' \n Supported reduction modes:"
                f" 'none', 'mean', 'sum'. It will be used 'sum' instead.",
                stacklevel=2,
            )
            loss = masked_loss.sum()
        return loss

    def rel_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        pair_mask: torch.Tensor,
        class_mask: torch.Tensor,
        alpha: float = -1.0,
        gamma: float = 0.0,
        prob_margin: float = 0.0,
        label_smoothing: float = 0.0,
        reduction: str = "sum",
        negatives: float = 1.0,
        masking: str = "span",
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute relation classification loss for selected entity pairs.

        Args:
            logits: Predicted relation scores of shape (B, N, C).
            labels: Ground truth relation labels of shape (B, N, C).
            pair_mask: Mask for valid pairs of shape (B, N, C).
            class_mask: Mask for valid relation classes of shape (B, N, C).
            alpha: Focal loss alpha parameter.
            gamma: Focal loss gamma parameter.
            prob_margin: Margin for probability adjustment.
            label_smoothing: Label smoothing factor.
            reduction: Loss reduction method ('sum' or 'mean').
            negatives: Negative sampling probability.
            masking: Masking strategy for negative sampling.
            **kwargs: Additional arguments.

        Returns:
            Scalar relation classification loss tensor.
        """
        B, _, C = logits.shape

        all_losses = self._loss(
            logits, labels, alpha, gamma, prob_margin, label_smoothing, negatives=negatives, masking=masking
        )

        combined_mask = pair_mask * class_mask
        masked_loss = all_losses * combined_mask.view(B, -1, C)

        if reduction == "mean":
            num_valid = combined_mask.sum()
            loss = masked_loss.sum() / num_valid if num_valid > 0 else torch.tensor(0.0, device=logits.device)
        elif reduction == "sum":
            loss = masked_loss.sum()
        else:
            warnings.warn(
                f"Invalid Value for config 'loss_reduction': '{reduction}' \n Supported reduction modes:"
                f" 'none', 'mean', 'sum'. It will be used 'sum' instead.",
                stacklevel=2,
            )
            loss = masked_loss.sum()

        return loss
