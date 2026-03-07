from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Union, Optional
from functools import partial

import torch

from .utils import has_overlapping, has_overlapping_nested


class BaseDecoder(ABC):
    """
    Abstract base class for all decoders.

    Args:
        config: Configuration object containing decoder parameters.
    """

    def __init__(self, config):
        self.config = config

    @abstractmethod
    def decode(self, *args, **kwargs):
        """
        Decode model output into structured predictions.

        Args:
            *args: Variable positional arguments.
            **kwargs: Variable keyword arguments.

        Returns:
            Decoded predictions in the appropriate format.
        """
        pass

    def _get_id_to_class_for_sample(
        self, id_to_classes: Union[Dict[int, str], List[Dict[int, str]]], sample_idx: int
    ) -> Dict[int, str]:
        """
        Get id_to_classes mapping for a specific sample.

        Args:
            id_to_classes (Union[Dict[int, str], List[Dict[int, str]]]): Either a single
                mapping shared across all samples or per-sample mappings.
            sample_idx (int): Index of the sample in the batch.

        Returns:
            Dict[int, str]: Mapping from class IDs to class names for this sample.
        """
        if isinstance(id_to_classes, list):
            return id_to_classes[sample_idx]
        return id_to_classes

    def greedy_search(self, spans: List[tuple], flat_ner: bool = True, multi_label: bool = False) -> List[tuple]:
        """
        Perform greedy search to remove overlapping spans.

        Sorts spans by confidence score (descending) and keeps only non-overlapping
        spans according to the specified NER mode.

        Args:
            spans (List[tuple]): List of span tuples containing at minimum
                (start, end, ..., score).
            flat_ner (bool): Whether to use flat NER (no nesting allowed) or
                nested NER (allows nesting).
            multi_label (bool): Whether to allow multiple labels for the same
                span position.

        Returns:
            List[tuple]: Filtered list of non-overlapping spans, sorted by start position.
        """
        if flat_ner:
            has_ov = partial(has_overlapping, multi_label=multi_label)
        else:
            has_ov = partial(has_overlapping_nested, multi_label=multi_label)

        new_list = []
        # Sort by probability (descending)
        span_prob = sorted(spans, key=lambda x: -x[-1])

        for i in range(len(spans)):
            b = span_prob[i]
            flag = False
            for new in new_list:
                if has_ov(b[:-1], new):
                    flag = True
                    break
            if not flag:
                new_list.append(b)

        # Sort by start position
        new_list = sorted(new_list, key=lambda x: x[0])
        return new_list


class BaseSpanDecoder(BaseDecoder):
    """
    Base class for span-based decoders with common decoding logic.

    Provides shared functionality for finding candidate spans, validating them,
    and decoding batch items.
    """

    def _find_candidate_spans(
        self, probs: torch.Tensor, threshold: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Find all span candidates above threshold.

        Args:
            probs (torch.Tensor): Probability tensor of shape (L, K, C) for one sample,
                where L is sequence length, K is max span width, C is number of classes.
            threshold (float): Confidence threshold for predictions.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Tuple of
                (start_indices, width_indices, class_indices) for spans above threshold.
        """
        return torch.where(probs > threshold)

    def _is_valid_span(self, start: int, width: int, tokens: List[str]) -> bool:
        """
        Check if a span is valid (doesn't exceed sentence length).

        Args:
            start (int): Start position of the span.
            width (int): Span width (0-indexed, so actual span length is width + 1).
            tokens (List[str]): List of tokens for this sample.

        Returns:
            bool: True if span is valid, False otherwise.
        """
        end = start + width + 1
        return end <= len(tokens)

    @abstractmethod
    def _build_span_tuple(
        self,
        start: int,
        width: int,
        class_idx: int,
        flat_idx: int,
        score: float,
        id_to_class: Dict[int, str],
        span_label_map: Dict[int, List[str]],
    ) -> tuple:
        """
        Build a span tuple with decoder-specific format.

        Args:
            start (int): Start position of the span.
            width (int): Span width (0-indexed).
            class_idx (int): Class index.
            flat_idx (int): Flattened span index (start * K + width).
            score (float): Confidence score for this span.
            id_to_class (Dict[int, str]): Mapping from class IDs to class names.
            span_label_map (Dict[int, List[str]]): Mapping from flat span indices
                to generated labels (empty for non-generative decoders).

        Returns:
            tuple: Span tuple in decoder-specific format.
        """
        raise NotImplementedError("Subclasses must implement _build_span_tuple")

    def _decode_batch_item(
        self,
        probs_i: torch.Tensor,
        tokens_i: List[str],
        id_to_class_i: Dict[int, str],
        K: int,
        threshold: float,
        flat_ner: bool,
        multi_label: bool,
        span_label_map: Dict[int, List[str]],
    ) -> List[tuple]:
        """
        Decode spans for a single batch item.

        Finds all candidate spans above threshold, validates them, builds span tuples,
        and applies greedy search to remove overlaps.

        Args:
            probs_i (torch.Tensor): Probability tensor of shape (L, K, C) for this sample.
            tokens_i (List[str]): List of tokens for this sample.
            id_to_class_i (Dict[int, str]): Class ID to class name mapping for this sample.
            K (int): Maximum span width.
            threshold (float): Confidence threshold for predictions.
            flat_ner (bool): Whether to enforce non-overlapping spans.
            multi_label (bool): Whether to allow multiple labels per span.
            span_label_map (Dict[int, List[str]]): Mapping from flat span indices to
                generated labels (empty dict for non-generative decoders).

        Returns:
            List[tuple]: List of decoded span tuples for this sample.
        """
        span_i = []

        # Find all spans above threshold
        s_idx, k_idx, c_idx = self._find_candidate_spans(probs_i, threshold)

        for s, k, c in zip(s_idx.tolist(), k_idx.tolist(), c_idx.tolist()):
            # Skip if span exceeds sentence length
            if not self._is_valid_span(s, k, tokens_i):
                continue

            # Calculate flat index (matches encoder's indexing)
            flat_idx = s * K + k
            score = probs_i[s, k, c].item()

            # Build span tuple (implementation varies by subclass)
            span_tuple = self._build_span_tuple(s, k, c, flat_idx, score, id_to_class_i, span_label_map)
            span_i.append(span_tuple)

        # Remove overlapping spans using greedy search
        span_i = self.greedy_search(span_i, flat_ner, multi_label=multi_label)
        return span_i

    def decode(
        self,
        tokens: List[List[str]],
        id_to_classes: Union[Dict[int, str], List[Dict[int, str]]],
        model_output: torch.Tensor,
        flat_ner: bool = False,
        threshold: float = 0.5,
        multi_label: bool = False,
        **kwargs,
    ) -> List[List[tuple]]:
        """
        Decode model output to extract named entity spans.

        Args:
            tokens (List[List[str]]): Tokenized input text for each sample in the batch.
            id_to_classes (Union[Dict[int, str], List[Dict[int, str]]]): Mapping from
                class IDs to class names. Can be a single dict (shared) or list (per-sample).
            model_output (torch.Tensor): Raw logits from the model with shape (B, L, K, C),
                where B is batch size, L is sequence length, K is max span width,
                C is number of classes.
            flat_ner (bool): Whether to enforce non-overlapping spans.
            threshold (float): Confidence threshold for span predictions.
            multi_label (bool): Whether to allow multiple labels per span.
            **kwargs: Additional keyword arguments (unused in base class).

        Returns:
            List[List[tuple]]: For each sample in batch, list of span tuples.
        """
        B, _, K, _ = model_output.shape  # B, L, K, C
        probs = torch.sigmoid(model_output)

        # Decode spans for each sample in the batch
        spans = []
        for i in range(B):
            probs_i = probs[i]
            id_to_class_i = self._get_id_to_class_for_sample(id_to_classes, i)

            # For base decoder, span_label_map is empty
            span_label_map = {}

            span_i = self._decode_batch_item(
                probs_i=probs_i,
                tokens_i=tokens[i],
                id_to_class_i=id_to_class_i,
                K=K,
                threshold=threshold,
                flat_ner=flat_ner,
                multi_label=multi_label,
                span_label_map=span_label_map,
            )
            spans.append(span_i)

        return spans


class SpanDecoder(BaseSpanDecoder):
    """
    Simple span decoder without generative labels.

    Returns spans in format: (start, end, entity_type, score)
    """

    def _build_span_tuple(
        self,
        start: int,
        width: int,
        class_idx: int,
        flat_idx: int,
        score: float,
        id_to_class: Dict[int, str],
        span_label_map: Dict[int, List[str]],
    ) -> tuple:
        """
        Build span tuple without generative labels.

        Args:
            start (int): Start position of the span.
            width (int): Span width (0-indexed).
            class_idx (int): Class index.
            flat_idx (int): Flattened span index (unused in this decoder).
            score (float): Confidence score for this span.
            id_to_class (Dict[int, str]): Mapping from class IDs to class names.
            span_label_map (Dict[int, List[str]]): Unused in this decoder.

        Returns:
            tuple: Span tuple in format (start, end, entity_type, score).
        """
        ent_type = id_to_class[class_idx + 1]  # +1 because 0 is <pad>
        return (start, start + width, ent_type, score)


class SpanGenerativeDecoder(BaseSpanDecoder):
    """
    Span decoder with generative label support.

    Supports two decoder modes:
    - 'prompt': Generated labels replace the original class names
    - 'span': Generated labels are added as additional fields to each span

    Returns spans in format: (start, end, entity_type, generated_entity_type, score)
    """

    def _update_id_to_classes_with_generated(
        self, id_to_classes: Union[Dict, List[Dict]], gen_labels: List[str], batch_size: int
    ) -> Union[Dict, List[Dict]]:
        """
        Update id_to_classes mapping with generated labels for prompt mode.

        In prompt mode, the generated labels replace the original class names in the
        id_to_class mapping. This method maps generated labels back to class IDs.

        Args:
            id_to_classes (Union[Dict, List[Dict]]): Original mapping from class IDs
                to class names.
            gen_labels (List[str]): Generated labels from the decoder, flattened across batch.
            batch_size (int): Number of samples in the batch.

        Returns:
            Union[Dict, List[Dict]]: Updated id_to_classes mapping with generated labels.
        """
        new_id_to_classes = []
        cursor = 0

        for i in range(batch_size):
            original = id_to_classes[i] if isinstance(id_to_classes, list) else id_to_classes
            k = len(original)  # Number of labels for this example

            # Map each class ID to its generated label
            mapping = {idx + 1: gen_labels[cursor + idx] for idx in range(k)}
            new_id_to_classes.append(mapping)
            cursor += k

        return new_id_to_classes

    def _build_span_label_map_for_batch(
        self, sel_idx: torch.LongTensor, gen_labels: List[str], num_gen_sequences: int
    ) -> List[Dict[int, List[str]]]:
        """
        Build mapping from flat span indices to generated labels for span mode.

        In span mode, each valid span gets one or more generated labels. This method
        creates a mapping from flat span index to its corresponding generated label(s).

        Args:
            sel_idx (torch.LongTensor): Tensor of shape (B, M) containing indices of
                selected spans. -1 indicates padding.
            gen_labels (List[str]): All generated labels, flattened across batch.
            num_gen_sequences (int): Number of label sequences generated per span.

        Returns:
            List[Dict[int, List[str]]]: One dict per batch element, mapping
                flat_span_idx -> list of generated labels.
        """
        batch_size = sel_idx.shape[0]
        span_label_maps = [{} for _ in range(batch_size)]
        cursor = 0  # Tracks position in gen_labels

        for b in range(batch_size):
            valid_pos = sel_idx[b] != -1
            n = valid_pos.sum().item()  # Number of valid spans in this batch element

            if n > 0:
                # Get the flat indices of spans that were kept
                flat_indices = sel_idx[b, valid_pos].tolist()

                # Calculate the range of labels for this batch element
                start_index = cursor * num_gen_sequences
                end_index = start_index + n * num_gen_sequences
                span_labels = gen_labels[start_index:end_index]

                # Group labels: each span gets num_gen_sequences consecutive labels
                labels_b = [span_labels[i * num_gen_sequences : (i + 1) * num_gen_sequences] for i in range(n)]

                # Create mapping from flat_index to labels
                span_label_maps[b] = dict(zip(flat_indices, labels_b))
                cursor += n

        return span_label_maps

    def _build_span_tuple(
        self,
        start: int,
        width: int,
        class_idx: int,
        flat_idx: int,
        score: float,
        id_to_class: Dict[int, str],
        span_label_map: Dict[int, List[str]],
    ) -> tuple:
        """
        Build span tuple with generative labels.

        Args:
            start (int): Start position of the span.
            width (int): Span width (0-indexed).
            class_idx (int): Class index.
            flat_idx (int): Flattened span index (start * K + width).
            score (float): Confidence score for this span.
            id_to_class (Dict[int, str]): Mapping from class IDs to class names.
            span_label_map (Dict[int, List[str]]): Mapping from flat span indices
                to generated labels.

        Returns:
            tuple: Span tuple in format (start, end, entity_type, generated_entity_type, score).
                generated_entity_type is None if not found in span_label_map.
        """
        ent_type = id_to_class[class_idx + 1]  # +1 because 0 is <pad>
        gen_ent_type = span_label_map.get(flat_idx)
        return (start, start + width, ent_type, gen_ent_type, score)

    def decode_generative(
        self,
        tokens: List[List[str]],
        id_to_classes: Union[Dict[int, str], List[Dict[int, str]]],
        model_output: torch.Tensor,
        gen_labels: List[str],
        sel_idx: Optional[torch.LongTensor] = None,
        num_gen_sequences: int = 1,
        flat_ner: bool = False,
        threshold: float = 0.5,
        multi_label: bool = False,
    ) -> List[List[tuple]]:
        """
        Decode model output with generated labels.

        Handles both 'prompt' and 'span' decoder modes:
        - prompt mode: Generated labels replace class names in id_to_classes
        - span mode: Generated labels are added to span tuples via span_label_map

        Args:
            tokens (List[List[str]]): Tokenized input text for each sample in the batch.
            id_to_classes (Union[Dict[int, str], List[Dict[int, str]]]): Mapping from
                class IDs to class names.
            model_output (torch.Tensor): Raw logits from the model with shape (B, L, K, C).
            gen_labels (List[str]): Generated labels from the decoder, flattened across batch.
            sel_idx (Optional[torch.LongTensor]): Tensor of shape (B, M) with selected
                span indices. Required for span mode, unused for prompt mode.
            num_gen_sequences (int): Number of label sequences generated per span.
            flat_ner (bool): Whether to enforce non-overlapping spans.
            threshold (float): Confidence threshold for span predictions.
            multi_label (bool): Whether to allow multiple labels per span.

        Returns:
            List[List[tuple]]: For each sample, list of span tuples with generated labels.
        """
        B, _, K, _ = model_output.shape  # B, L, K, C
        probs = torch.sigmoid(model_output)

        # Handle prompt mode: update id_to_classes with generated labels
        if self.config.decoder_mode == "prompt":
            id_to_classes = self._update_id_to_classes_with_generated(id_to_classes, gen_labels, B)
            # In prompt mode, span_label_map is empty (labels already in id_to_classes)
            span_label_maps = [{} for _ in range(B)]

        # Handle span mode: build span_label_map from sel_idx and gen_labels
        elif self.config.decoder_mode == "span":
            if sel_idx is not None:
                span_label_maps = self._build_span_label_map_for_batch(sel_idx, gen_labels, num_gen_sequences)
            else:
                span_label_maps = [{} for _ in range(B)]

        else:
            # No decoder mode or unknown mode
            span_label_maps = [{} for _ in range(B)]

        # Decode spans for each sample in the batch
        spans = []
        for i in range(B):
            probs_i = probs[i]
            id_to_class_i = self._get_id_to_class_for_sample(id_to_classes, i)
            span_label_map_i = span_label_maps[i]

            span_i = self._decode_batch_item(
                probs_i=probs_i,
                tokens_i=tokens[i],
                id_to_class_i=id_to_class_i,
                K=K,
                threshold=threshold,
                flat_ner=flat_ner,
                multi_label=multi_label,
                span_label_map=span_label_map_i,
            )
            spans.append(span_i)

        return spans

    def decode(
        self,
        tokens: List[List[str]],
        id_to_classes: Union[Dict[int, str], List[Dict[int, str]]],
        model_output: torch.Tensor,
        flat_ner: bool = False,
        threshold: float = 0.5,
        multi_label: bool = False,
        gen_labels: Optional[List[str]] = None,
        sel_idx: Optional[torch.LongTensor] = None,
        num_gen_sequences: int = 1,
        **kwargs,
    ) -> List[List[tuple]]:
        """
        Decode model output, with optional generative label support.

        If gen_labels are provided and decoder has a labels_decoder, uses generative
        decoding. Otherwise falls back to standard span decoding.

        Args:
            tokens (List[List[str]]): Tokenized input text for each sample in the batch.
            id_to_classes (Union[Dict[int, str], List[Dict[int, str]]]): Mapping from
                class IDs to class names.
            model_output (torch.Tensor): Raw logits from the model with shape (B, L, K, C).
            flat_ner (bool): Whether to enforce non-overlapping spans.
            threshold (float): Confidence threshold for span predictions.
            multi_label (bool): Whether to allow multiple labels per span.
            gen_labels (Optional[List[str]]): Generated labels from decoder. If provided,
                triggers generative decoding.
            sel_idx (Optional[torch.LongTensor]): Selected span indices for span mode.
            num_gen_sequences (int): Number of label sequences generated per span.
            **kwargs: Additional keyword arguments (unused).

        Returns:
            List[List[tuple]]: For each sample, list of span tuples.
        """
        # Use generative decoding if labels_decoder is configured and gen_labels provided
        if self.config.labels_decoder is not None and gen_labels is not None:
            return self.decode_generative(
                tokens=tokens,
                id_to_classes=id_to_classes,
                model_output=model_output,
                gen_labels=gen_labels,
                sel_idx=sel_idx,
                num_gen_sequences=num_gen_sequences,
                flat_ner=flat_ner,
                threshold=threshold,
                multi_label=multi_label,
            )

        # Fall back to standard decoding without generative labels
        return super().decode(
            tokens=tokens,
            id_to_classes=id_to_classes,
            model_output=model_output,
            flat_ner=flat_ner,
            threshold=threshold,
            multi_label=multi_label,
        )


class SpanRelexDecoder(BaseSpanDecoder):
    """Span decoder with relation extraction support.

    Extends the base span decoder to decode both entity spans and the relations
    between them. Entity spans are extracted first using the parent class logic,
    then relations are decoded by identifying pairs of entities and their
    relationship types based on model predictions.

    The decoder supports:
    - Entity span extraction with confidence thresholding
    - Relation extraction between detected entities
    - Flexible entity and relation label mappings (per-sample or global)
    - Optional flat NER (non-overlapping entities)
    - Multi-label entity classification
    """

    def _build_span_tuple(
        self,
        start: int,
        width: int,
        class_idx: int,
        flat_idx: int,
        score: float,
        id_to_class: Dict[int, str],
        span_label_map: Dict[int, List[str]],
    ) -> tuple:
        """Build an entity span tuple for relation extraction.

        Constructs a tuple representing a detected entity span with its boundaries,
        type, and confidence score. This format is used for both entity representation
        and as input to relation extraction.

        Args:
            start: Starting token position of the span (inclusive).
            width: Width of the span in tokens (0-indexed, so actual span length is width + 1).
            class_idx: Index of the entity class (0-indexed in the model output).
            flat_idx: Flattened span index in the original span representation.
                Unused in this decoder but required by the parent class interface.
            score: Confidence score for this entity span prediction (typically 0-1).
            id_to_class: Dictionary mapping class indices to entity type names.
                Keys are 1-indexed (0 reserved for padding).
            span_label_map: Mapping from span indices to allowed labels.
                Unused in this decoder but required by the parent class interface.

        Returns:
            Tuple in format (start, end, entity_type, score) where:
            - start: Starting token position (inclusive)
            - end: Ending token position (exclusive)
            - entity_type: String name of the entity type
            - score: Confidence score (float)
        """
        ent_type = id_to_class[class_idx + 1]  # +1 because 0 is <pad>
        return (start, start + width, ent_type, score)

    def _build_entity_span_to_decoded_idx(
        self,
        spans: List[List[tuple]],
        entity_spans: Optional[torch.Tensor],
        batch_size: int,
    ) -> List[Optional[dict]]:
        """Build mapping from model entity indices to decoded span indices.

        Maps entity positions in the model's internal target_span_rep
        (referenced by pair_idx) to positions in the decoded spans list.
        Uses span boundaries (start, end) for matching.

        Args:
            spans: Decoded entity spans per sample, each as (start, end, type, score).
            entity_spans: Tensor of shape (B, E, 2) with span boundaries for each
                entity in target_span_rep. None if not available.
            batch_size: Number of samples.

        Returns:
            List of dicts mapping model entity index -> decoded span index,
            or None entries if entity_spans is not available.
        """
        if entity_spans is None:
            return [None] * batch_size

        mappings = []
        for i in range(batch_size):
            # Build reverse lookup: (start, end) -> index in decoded spans
            span_boundary_to_idx = {}
            for idx, span_tuple in enumerate(spans[i]):
                key = (span_tuple[0], span_tuple[1])
                if key not in span_boundary_to_idx:
                    span_boundary_to_idx[key] = idx

            # Map each entity in target_span_rep to decoded span index
            model_to_decoded = {}
            for e in range(entity_spans.size(1)):
                start = entity_spans[i, e, 0].item()
                end = entity_spans[i, e, 1].item()
                key = (start, end)
                if key in span_boundary_to_idx:
                    model_to_decoded[e] = span_boundary_to_idx[key]

            mappings.append(model_to_decoded)
        return mappings

    def _decode_relations(
        self,
        model_output,
        spans: List[List[tuple]],
        rel_idx: Optional[torch.Tensor],
        rel_logits: Optional[torch.Tensor],
        rel_mask: Optional[torch.Tensor],
        rel_id_to_classes: Union[Dict[int, str], List[Dict[int, str]]],
        threshold: float,
        batch_size: int,
        entity_spans: Optional[torch.Tensor] = None,
    ) -> List[List[tuple]]:
        """Decode relations between detected entity spans.

        Uses entity_spans (model's internal entity boundaries) to correctly map
        pair_idx values (which reference target_span_rep positions) to positions
        in the decoded spans list. This is necessary because greedy search may
        remove or reorder entities during decoding.

        Args:
            model_output: Model output object containing relation predictions.
            spans: List of entity spans for each sample in the batch.
                Each sample contains a list of tuples: (start, end, entity_type, score).
            rel_idx: Tensor of shape (batch_size, num_relations, 2) containing
                indices of head and tail entities for each potential relation.
            rel_logits: Tensor of shape (batch_size, num_relations, num_relation_classes)
                containing logits for relation classifications.
            rel_mask: Optional boolean tensor of shape (batch_size, num_relations)
                indicating which relations are valid (True) vs. padding (False).
            rel_id_to_classes: Mapping from relation class IDs to relation names.
                Class IDs are 1-indexed (0 reserved for "no relation" or padding).
            threshold: Minimum confidence score (after sigmoid) for a relation.
            batch_size: Number of samples in the batch.
            entity_spans: Optional tensor of shape (B, E, 2) with span boundaries
                for entities in target_span_rep. Used to map pair_idx to decoded spans.

        Returns:
            List of relation lists, one per sample. Each relation is a tuple:
            (head_idx, relation_label, tail_idx, score) where head_idx and tail_idx
            are indices into the corresponding sample's decoded spans list.
        """
        relations = [[] for _ in range(batch_size)]

        # Check if relation outputs are available
        if rel_idx is None or rel_logits is None:
            return relations

        # Get or create relation mask
        if rel_mask is None:
            # Create default mask (all valid)
            rel_mask = torch.ones(rel_idx[..., 0].shape, dtype=torch.bool, device=rel_idx.device)

        rel_probs = torch.sigmoid(rel_logits)

        # Build mapping from model entity indices to decoded span indices
        idx_mappings = self._build_entity_span_to_decoded_idx(spans, entity_spans, batch_size)

        # Decode relations for each sample
        for i in range(batch_size):
            rel_id_to_class_i = rel_id_to_classes[i] if isinstance(rel_id_to_classes, list) else rel_id_to_classes
            idx_map = idx_mappings[i]

            # Process each potential relation
            for j in range(rel_idx.size(1)):
                # Skip if masked out
                if not rel_mask[i, j]:
                    continue

                model_head_idx = rel_idx[i, j, 0].item()
                model_tail_idx = rel_idx[i, j, 1].item()

                # Skip invalid indices
                if model_head_idx < 0 or model_tail_idx < 0:
                    continue

                # Map model entity indices to decoded span indices
                if idx_map is not None:
                    head_idx = idx_map.get(model_head_idx)
                    tail_idx = idx_map.get(model_tail_idx)
                    if head_idx is None or tail_idx is None:
                        continue
                else:
                    # Fallback: use model indices directly (legacy behavior)
                    head_idx = model_head_idx
                    tail_idx = model_tail_idx
                    if head_idx >= len(spans[i]) or tail_idx >= len(spans[i]):
                        continue

                # Check each relation class
                for c, p in enumerate(rel_probs[i, j]):
                    prob = p.item()

                    # Skip low confidence predictions
                    if prob <= threshold:
                        continue

                    # Skip if class ID not in mapping
                    # (c + 1 because 0 may be "no-relation" or padding)
                    if (c + 1) not in rel_id_to_class_i:
                        continue

                    rel_label = rel_id_to_class_i[c + 1]

                    # Append relation: (head_idx, relation_label, tail_idx, score)
                    relations[i].append((head_idx, rel_label, tail_idx, prob))

        return relations

    def decode(
        self,
        tokens: List[List[str]],
        id_to_classes: Union[Dict[int, str], List[Dict[int, str]]],
        model_output,
        rel_idx: Optional[torch.Tensor] = None,
        rel_logits: Optional[torch.Tensor] = None,
        rel_mask: Optional[torch.Tensor] = None,
        flat_ner: bool = False,
        threshold: float = 0.5,
        relation_threshold: float = 0.5,
        multi_label: bool = False,
        rel_id_to_classes: Optional[Union[Dict[int, str], List[Dict[int, str]]]] = None,
        entity_spans: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[List[List[tuple]], List[List[tuple]]]:
        """Decode model output to extract entities and relations.

        Main decoding method that extracts both entity spans and relations from
        model outputs. First decodes entity spans using the parent class logic,
        then decodes relations between the detected entities.

        Args:
            tokens: Tokenized input text for each sample in the batch.
            id_to_classes: Mapping from entity class IDs to entity type names.
            model_output: Model output object containing both entity logits and
                optionally relation predictions.
            rel_idx: Optional tensor of shape (batch_size, num_relations, 2).
            rel_logits: Optional tensor of shape (batch_size, num_relations, num_relation_classes).
            rel_mask: Optional boolean tensor of shape (batch_size, num_relations).
            flat_ner: If True, applies greedy filtering for non-overlapping entities.
            threshold: Minimum confidence score for entity predictions.
            relation_threshold: Minimum confidence score for relation predictions.
            multi_label: If True, allows multiple entity types per span.
            rel_id_to_classes: Optional mapping from relation class IDs to relation names.
            entity_spans: Optional tensor of shape (B, E, 2) with span boundaries
                for entities in target_span_rep. Used for correct index mapping.
            **kwargs: Additional keyword arguments passed to the parent class decode method.

        Returns:
            Tuple of (spans, relations).
        """
        # Decode entity spans using base class logic
        spans = super().decode(
            tokens=tokens,
            id_to_classes=id_to_classes,
            model_output=model_output,
            flat_ner=flat_ner,
            threshold=threshold,
            multi_label=multi_label,
            **kwargs,
        )

        # Decode relations if requested
        relations = [[] for _ in range(len(tokens))]  # Default: empty lists

        if rel_id_to_classes is not None:
            relations = self._decode_relations(
                model_output=model_output,
                rel_idx=rel_idx,
                rel_logits=rel_logits,
                rel_mask=rel_mask,
                spans=spans,
                rel_id_to_classes=rel_id_to_classes,
                threshold=relation_threshold,
                batch_size=len(tokens),
                entity_spans=entity_spans,
            )

        return spans, relations


class TokenDecoder(BaseDecoder):
    """
    Token-based decoder for sequence labeling tasks.

    Uses BIO-style tagging with separate start, end, and inside predictions
    to identify entity spans. Can also decode from span-level predictions.
    """

    def _get_indices_above_threshold(self, scores: torch.Tensor, threshold: float) -> List[torch.Tensor]:
        """
        Get indices where scores exceed threshold.

        Args:
            scores (torch.Tensor): Score tensor for one sample.
            threshold (float): Confidence threshold.

        Returns:
            List[torch.Tensor]: List of tensors containing indices above threshold.
        """
        scores = torch.sigmoid(scores)
        return [k.tolist() for k in torch.where(scores > threshold)]

    def _calculate_span_score(
        self,
        start_idx: tuple,
        end_idx: tuple,
        scores_inside_i: torch.Tensor,
        start_i: torch.Tensor,
        end_i: torch.Tensor,
        id_to_classes: Dict[int, str],
        threshold: float,
    ) -> List[tuple]:
        """
        Calculate spans and their scores from start/end/inside predictions.

        Matches start and end positions of the same class, validates inside scores,
        and computes final span scores.

        Args:
            start_idx (tuple): Tuple of (positions, classes) for start predictions.
            end_idx (tuple): Tuple of (positions, classes) for end predictions.
            scores_inside_i (torch.Tensor): Inside scores for this sample.
            start_i (torch.Tensor): Start scores for this sample.
            end_i (torch.Tensor): End scores for this sample.
            id_to_classes (Dict[int, str]): Mapping from class IDs to class names.
            threshold (float): Confidence threshold.

        Returns:
            List[tuple]: List of span tuples (start, end, entity_type, score).
        """
        span_i = []
        for st, cls_st in zip(*start_idx):
            for ed, cls_ed in zip(*end_idx):
                if ed >= st and cls_st == cls_ed:
                    ins = scores_inside_i[st : ed + 1, cls_st]
                    if (ins < threshold).any():
                        continue
                    # Get the start and end scores for this span
                    start_score = start_i[st, cls_st]
                    end_score = end_i[ed, cls_ed]
                    # Concatenate the inside scores with start and end scores
                    combined = torch.cat([ins, start_score.unsqueeze(0), end_score.unsqueeze(0)])
                    # The span score is the minimum value among these scores
                    spn_score = combined.min().item()
                    span_i.append((st, ed, id_to_classes[cls_st + 1], spn_score))
        return span_i

    def _decode_from_spans(
        self,
        tokens: List[List[str]],
        id_to_classes: Union[Dict[int, str], List[Dict[int, str]]],
        span_logits: torch.Tensor,
        span_idx: torch.Tensor,
        span_mask: torch.Tensor,
        flat_ner: bool = False,
        threshold: float = 0.5,
        multi_label: bool = False,
    ) -> List[List[tuple]]:
        """
        Decode from span-level predictions.

        Args:
            tokens (List[List[str]]): Tokenized input text for each sample in the batch.
            id_to_classes (Union[Dict[int, str], List[Dict[int, str]]]): Mapping from
                class IDs to class names.
            span_logits (torch.Tensor): Span classification logits with shape (B, S, C),
                where B is batch size, S is max spans, C is number of classes.
            span_idx (torch.Tensor): Span indices with shape (B, S, 2), containing
                [start, end] positions for each span.
            span_mask (torch.Tensor): Boolean mask with shape (B, S) indicating
                valid spans.
            flat_ner (bool): Whether to enforce non-overlapping spans.
            threshold (float): Confidence threshold for predictions.
            multi_label (bool): Whether to allow multiple labels per span.

        Returns:
            List[List[tuple]]: For each sample, list of span tuples in format
                (start, end, entity_type, None, score).
        """
        batch_size = span_logits.size(0)
        spans = []

        # Apply sigmoid to get probabilities
        span_probs = torch.sigmoid(span_logits)

        for i in range(batch_size):
            id_to_class_i = self._get_id_to_class_for_sample(id_to_classes, i)
            span_scores = []

            # Get valid spans for this sample
            valid_mask = span_mask[i]
            valid_indices = torch.where(valid_mask)[0]

            for span_pos in valid_indices:
                span_start = span_idx[i, span_pos, 0].item()
                span_end = span_idx[i, span_pos, 1].item()

                # Get probabilities for all classes for this span
                probs = span_probs[i, span_pos]

                # Find classes above threshold
                class_indices = torch.where(probs > threshold)[0]

                for class_idx in class_indices:
                    class_id = class_idx.item() + 1  # Convert to 1-indexed
                    if class_id in id_to_class_i:
                        entity_type = id_to_class_i[class_id]
                        score = probs[class_idx].item()
                        span_scores.append((span_start, span_end, entity_type, score))

            # Apply greedy search to handle overlapping spans if needed
            span_i = self.greedy_search(span_scores, flat_ner, multi_label)
            spans.append(span_i)
        return spans

    def decode(
        self,
        tokens: List[List[str]],
        id_to_classes: Union[Dict[int, str], List[Dict[int, str]]],
        model_output: Optional[torch.Tensor] = None,
        flat_ner: bool = False,
        threshold: float = 0.5,
        multi_label: bool = False,
        span_logits: Optional[torch.Tensor] = None,
        span_idx: Optional[torch.Tensor] = None,
        span_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> List[List[tuple]]:
        """
        Decode predictions to extract spans.

        Supports two decoding modes:
        1. Token-level BIO decoding (default): Uses model_output with start/end/inside predictions
        2. Span-level decoding: Uses span_logits, span_idx, and span_mask

        Args:
            tokens (List[List[str]]): Tokenized input text for each sample in the batch.
            id_to_classes (Union[Dict[int, str], List[Dict[int, str]]]): Mapping from
                class IDs to class names.
            model_output (torch.Tensor, optional): Raw logits from the model with shape
                (B, L, C, 3), where the last dimension represents [start, end, inside]
                predictions. Used for token-level decoding.
            flat_ner (bool): Whether to enforce non-overlapping spans.
            threshold (float): Confidence threshold for predictions.
            multi_label (bool): Whether to allow multiple labels per span.
            span_logits (torch.Tensor, optional): Span classification logits with shape
                (B, S, C). Used for span-level decoding.
            span_idx (torch.Tensor, optional): Span indices with shape (B, S, 2).
                Used for span-level decoding.
            span_mask (torch.Tensor, optional): Boolean mask with shape (B, S).
                Used for span-level decoding.
            **kwargs: Additional keyword arguments (unused).

        Returns:
            List[List[tuple]]: For each sample, list of span tuples in format
                (start, end, entity_type, None, score).

        Raises:
            ValueError: If neither model_output nor span-level inputs are provided,
                or if span-level inputs are incomplete.
        """
        # Check if span-level decoding is requested
        if span_logits is not None and span_idx is not None and span_mask is not None:
            return self._decode_from_spans(
                tokens=tokens,
                id_to_classes=id_to_classes,
                span_logits=span_logits,
                span_idx=span_idx,
                span_mask=span_mask,
                flat_ner=flat_ner,
                threshold=threshold,
                multi_label=multi_label,
            )

        # Check if token-level decoding is requested
        if model_output is not None:
            model_output = model_output.permute(3, 0, 1, 2)
            scores_start, scores_end, scores_inside = model_output
            spans = []

            for i, _ in enumerate(tokens):
                id_to_class_i = self._get_id_to_class_for_sample(id_to_classes, i)
                span_scores = self._calculate_span_score(
                    self._get_indices_above_threshold(scores_start[i], threshold),
                    self._get_indices_above_threshold(scores_end[i], threshold),
                    torch.sigmoid(scores_inside[i]),
                    torch.sigmoid(scores_start[i]),
                    torch.sigmoid(scores_end[i]),
                    id_to_class_i,
                    threshold,
                )
                span_i = self.greedy_search(span_scores, flat_ner, multi_label)
                spans.append(span_i)
            return spans

        # Neither decoding mode has sufficient inputs
        if span_logits is not None or span_idx is not None or span_mask is not None:
            raise ValueError(
                "For span-level decoding, all three parameters must be provided: span_logits, span_idx, and span_mask"
            )

        raise ValueError(
            "Either model_output (for token-level decoding) or "
            "(span_logits, span_idx, span_mask) (for span-level decoding) must be provided"
        )


class TokenRelexDecoder(TokenDecoder):
    """Token-based decoder with relation extraction support.

    Extends the token-based BIO decoder to decode both entity spans and the relations
    between them. Entity spans are extracted first using BIO-style tagging logic,
    then relations are decoded by identifying pairs of entities and their
    relationship types based on model predictions.

    The decoder supports:
    - Entity span extraction via BIO tagging with confidence thresholding
    - Relation extraction between detected entities
    - Flexible entity and relation label mappings (per-sample or global)
    - Optional flat NER (non-overlapping entities)
    - Multi-label entity classification
    """

    def _build_entity_span_to_decoded_idx(
        self,
        spans: List[List[tuple]],
        entity_spans: Optional[torch.Tensor],
        batch_size: int,
    ) -> List[Optional[dict]]:
        """Build mapping from model entity indices to decoded span indices.

        Uses span boundaries (start, end) to match model entities to decoded spans.

        Args:
            spans: Decoded entity spans per sample, each as (start, end, type, score).
            entity_spans: Tensor of shape (B, E, 2) with span boundaries for each
                entity in target_span_rep. None if not available.
            batch_size: Number of samples.

        Returns:
            List of dicts mapping model entity index -> decoded span index.
        """
        if entity_spans is None:
            return [None] * batch_size

        mappings = []
        for i in range(batch_size):
            span_boundary_to_idx = {}
            for idx, span_tuple in enumerate(spans[i]):
                key = (span_tuple[0], span_tuple[1])
                if key not in span_boundary_to_idx:
                    span_boundary_to_idx[key] = idx

            model_to_decoded = {}
            for e in range(entity_spans.size(1)):
                start = entity_spans[i, e, 0].item()
                end = entity_spans[i, e, 1].item()
                key = (start, end)
                if key in span_boundary_to_idx:
                    model_to_decoded[e] = span_boundary_to_idx[key]

            mappings.append(model_to_decoded)
        return mappings

    def _decode_relations(
        self,
        spans: List[List[tuple]],
        rel_idx: Optional[torch.Tensor],
        rel_logits: Optional[torch.Tensor],
        rel_mask: Optional[torch.Tensor],
        rel_id_to_classes: Union[Dict[int, str], List[Dict[int, str]]],
        threshold: float,
        batch_size: int,
        entity_spans: Optional[torch.Tensor] = None,
    ) -> List[List[tuple]]:
        """Decode relations between detected entity spans.

        Uses entity_spans (model's internal entity boundaries) to correctly map
        pair_idx values to positions in the decoded spans list.

        Args:
            spans: List of entity spans for each sample in the batch.
            rel_idx: Tensor of shape (batch_size, num_relations, 2).
            rel_logits: Tensor of shape (batch_size, num_relations, num_relation_classes).
            rel_mask: Optional boolean tensor of shape (batch_size, num_relations).
            rel_id_to_classes: Mapping from relation class IDs to relation names.
            threshold: Minimum confidence score (after sigmoid).
            batch_size: Number of samples in the batch.
            entity_spans: Optional tensor of shape (B, E, 2) with span boundaries
                for entities in target_span_rep.

        Returns:
            List of relation lists, one per sample.
        """
        relations = [[] for _ in range(batch_size)]

        # Check if relation outputs are available
        if rel_idx is None or rel_logits is None:
            return relations

        # Get or create relation mask
        if rel_mask is None:
            rel_mask = torch.ones(rel_idx[..., 0].shape, dtype=torch.bool, device=rel_idx.device)

        rel_probs = torch.sigmoid(rel_logits)

        # Build mapping from model entity indices to decoded span indices
        idx_mappings = self._build_entity_span_to_decoded_idx(spans, entity_spans, batch_size)

        # Decode relations for each sample
        for i in range(batch_size):
            rel_id_to_class_i = rel_id_to_classes[i] if isinstance(rel_id_to_classes, list) else rel_id_to_classes
            idx_map = idx_mappings[i]

            # Process each potential relation
            for j in range(rel_idx.size(1)):
                # Skip if masked out
                if not rel_mask[i, j]:
                    continue

                model_head_idx = rel_idx[i, j, 0].item()
                model_tail_idx = rel_idx[i, j, 1].item()

                # Skip invalid indices
                if model_head_idx < 0 or model_tail_idx < 0:
                    continue

                # Map model entity indices to decoded span indices
                if idx_map is not None:
                    head_idx = idx_map.get(model_head_idx)
                    tail_idx = idx_map.get(model_tail_idx)
                    if head_idx is None or tail_idx is None:
                        continue
                else:
                    head_idx = model_head_idx
                    tail_idx = model_tail_idx
                    if head_idx >= len(spans[i]) or tail_idx >= len(spans[i]):
                        continue

                # Check each relation class
                for c, p in enumerate(rel_probs[i, j]):
                    prob = p.item()

                    # Skip low confidence predictions
                    if prob <= threshold:
                        continue

                    # Skip if class ID not in mapping (c + 1 because 0 is padding)
                    if (c + 1) not in rel_id_to_class_i:
                        continue

                    rel_label = rel_id_to_class_i[c + 1]
                    relations[i].append((head_idx, rel_label, tail_idx, prob))

        return relations

    def decode(
        self,
        tokens: List[List[str]],
        id_to_classes: Union[Dict[int, str], List[Dict[int, str]]],
        model_output: torch.Tensor,
        rel_idx: Optional[torch.Tensor] = None,
        rel_logits: Optional[torch.Tensor] = None,
        rel_mask: Optional[torch.Tensor] = None,
        flat_ner: bool = False,
        threshold: float = 0.5,
        relation_threshold: float = 0.5,
        multi_label: bool = False,
        rel_id_to_classes: Optional[Union[Dict[int, str], List[Dict[int, str]]]] = None,
        entity_spans: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[List[List[tuple]], List[List[tuple]]]:
        """Decode model output to extract entities and relations.

        Main decoding method that extracts both entity spans and relations from
        model outputs. First decodes entity spans using BIO-style token tagging,
        then decodes relations between the detected entities.

        Args:
            tokens: Tokenized input text for each sample in the batch.
                Each sample is a list of token strings.
            id_to_classes: Mapping from entity class IDs to entity type names.
                Can be either:
                - Dict: Single mapping used for all samples (global entity schema)
                - List[Dict]: Per-sample mappings for different entity schemas
                Class IDs are 1-indexed (0 is reserved for padding).
            model_output: Model output tensor with shape (B, L, C, 3) where the last
                dimension contains [start, end, inside] predictions for BIO tagging.
            rel_idx: Optional tensor of shape (batch_size, num_relations, 2) containing
                head and tail entity indices for each potential relation.
            rel_logits: Optional tensor of shape (batch_size, num_relations, num_relation_classes)
                containing relation classification logits.
            rel_mask: Optional boolean tensor of shape (batch_size, num_relations)
                indicating valid relations. If None, all relations are considered valid.
            flat_ner: If True, applies greedy filtering to ensure non-overlapping
                entity spans. If False, allows overlapping entities. Defaults to False.
            threshold: Minimum confidence score (0-1) for entity predictions
                to be included in the output. Defaults to 0.5.
            relation_threshold: Minimum confidence score (0-1) for relation
                predictions to be included in the output. Defaults to 0.5.
            multi_label: If True, allows multiple entity types per span. If False,
                only the highest-scoring entity type per span is kept. Defaults to False.
            rel_id_to_classes: Optional mapping from relation class IDs to relation names.
                If None, relation decoding is skipped and empty relation lists are returned.
                Can be either a single Dict or List[Dict] for per-sample mappings.
                Class IDs are 1-indexed.
            **kwargs: Additional keyword arguments passed to the parent class decode method.

        Returns:
            Tuple of (spans, relations) where:
            - spans: List of entity span lists, one per sample. Each entity span is
              a tuple: (start, end, entity_type, score)
            - relations: List of relation lists, one per sample. Each relation is
              a tuple: (head_idx, relation_label, tail_idx, score) where head_idx
              and tail_idx are indices into the corresponding sample's spans list.

        Examples:
            >>> decoder = TokenRelexDecoder(config)
            >>> tokens = [["John", "works", "at", "Microsoft"]]
            >>> id_to_classes = {1: "PERSON", 2: "ORG"}
            >>> rel_id_to_classes = {1: "works_at"}
            >>> spans, relations = decoder.decode(
            ...     tokens=tokens,
            ...     id_to_classes=id_to_classes,
            ...     model_output=output,
            ...     rel_id_to_classes=rel_id_to_classes,
            ...     threshold=0.5,
            ... )
            >>> # spans[0] might be: [(0, 0, "PERSON", 0.9), (3, 3, "ORG", 0.85)]
            >>> # relations[0] might be: [(0, "works_at", 1, 0.8)]
        """
        # Decode entity spans using parent class BIO-style logic
        spans = super().decode(
            tokens=tokens,
            id_to_classes=id_to_classes,
            model_output=model_output,
            flat_ner=flat_ner,
            threshold=threshold,
            multi_label=multi_label,
            **kwargs,
        )

        # Decode relations if requested
        relations = [[] for _ in range(len(tokens))]

        if rel_id_to_classes is not None:
            relations = self._decode_relations(
                spans=spans,
                rel_idx=rel_idx,
                rel_logits=rel_logits,
                rel_mask=rel_mask,
                rel_id_to_classes=rel_id_to_classes,
                threshold=relation_threshold,
                batch_size=len(tokens),
                entity_spans=entity_spans,
            )

        return spans, relations


class TokenGenerativeDecoder(TokenDecoder, SpanGenerativeDecoder):
    """Token-based decoder with generative label support.

    Extends the token decoder to support generated labels from an encoder-decoder
    architecture. Supports two decoder modes:
    - 'prompt': Generated labels replace the original class names
    - 'span': Generated labels are added as additional fields to each span

    Returns spans in format: (start, end, entity_type, generated_entity_type, score)
    """

    def decode_generative(
        self,
        tokens: List[List[str]],
        id_to_classes: Union[Dict[int, str], List[Dict[int, str]]],
        model_output: torch.Tensor,
        gen_labels: List[str],
        sel_idx: Optional[torch.LongTensor] = None,
        num_gen_sequences: int = 1,
        flat_ner: bool = False,
        threshold: float = 0.5,
        multi_label: bool = False,
        span_logits: Optional[torch.Tensor] = None,
        span_idx: Optional[torch.Tensor] = None,
        span_mask: Optional[torch.Tensor] = None,
    ) -> List[List[tuple]]:
        """Decode model output with generated labels.

        Handles both 'prompt' and 'span' decoder modes:
        - prompt mode: Generated labels replace class names in id_to_classes
        - span mode: Generated labels are added to span tuples via span_label_map

        Args:
            tokens (List[List[str]]): Tokenized input text for each sample in the batch.
            id_to_classes (Union[Dict[int, str], List[Dict[int, str]]]): Mapping from
                class IDs to class names.
            model_output (torch.Tensor): Raw logits from the model with shape (B, W, C, 3).
            gen_labels (List[str]): Generated labels from the decoder, flattened across batch.
            sel_idx (Optional[torch.LongTensor]): Tensor of shape (B, M) with selected
                span indices. Required for span mode, unused for prompt mode.
            num_gen_sequences (int): Number of label sequences generated per span.
            flat_ner (bool): Whether to enforce non-overlapping spans.
            threshold (float): Confidence threshold for span predictions.
            multi_label (bool): Whether to allow multiple labels per span.
            span_logits (torch.Tensor, optional): Span classification logits.
            span_idx (torch.Tensor, optional): Span indices.
            span_mask (torch.Tensor, optional): Span mask.

        Returns:
            List[List[tuple]]: For each sample, list of span tuples with generated labels.
        """
        B = model_output.size(0)

        # Handle prompt mode: update id_to_classes with generated labels
        if self.config.decoder_mode == "prompt":
            id_to_classes = self._update_id_to_classes_with_generated(id_to_classes, gen_labels, B)
            span_label_maps = [{} for _ in range(B)]

        # Handle span mode: build span_label_map from sel_idx and gen_labels
        elif self.config.decoder_mode == "span":
            if sel_idx is not None:
                span_label_maps = self._build_span_label_map_for_batch(sel_idx, gen_labels, num_gen_sequences)
            else:
                span_label_maps = [{} for _ in range(B)]
        else:
            span_label_maps = [{} for _ in range(B)]

        batch_size = span_logits.size(0)
        spans = []

        span_probs = torch.sigmoid(span_logits)

        for i in range(batch_size):
            id_to_class_i = self._get_id_to_class_for_sample(id_to_classes, i)
            span_label_map_i = span_label_maps[i]
            span_scores = []

            valid_mask = span_mask[i]
            valid_indices = torch.where(valid_mask)[0]

            for span_pos in valid_indices:
                span_start = span_idx[i, span_pos, 0].item()
                span_end = span_idx[i, span_pos, 1].item()

                probs = span_probs[i, span_pos]
                class_indices = torch.where(probs > threshold)[0]

                for class_idx in class_indices:
                    class_id = class_idx.item() + 1
                    if class_id in id_to_class_i:
                        entity_type = id_to_class_i[class_id]
                        score = probs[class_idx].item()
                        gen_label = span_label_map_i.get(span_pos.item())
                        span_scores.append((span_start, span_end, entity_type, gen_label, score))

            span_i = self.greedy_search(span_scores, flat_ner, multi_label)
            spans.append(span_i)

        return spans

    def decode(
        self,
        tokens: List[List[str]],
        id_to_classes: Union[Dict[int, str], List[Dict[int, str]]],
        model_output: Optional[torch.Tensor] = None,
        flat_ner: bool = False,
        threshold: float = 0.5,
        multi_label: bool = False,
        gen_labels: Optional[List[str]] = None,
        sel_idx: Optional[torch.LongTensor] = None,
        num_gen_sequences: int = 1,
        span_logits: Optional[torch.Tensor] = None,
        span_idx: Optional[torch.Tensor] = None,
        span_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> List[List[tuple]]:
        """Decode model output, with optional generative label support.

        If gen_labels are provided and decoder has a labels_decoder, uses generative
        decoding. Otherwise falls back to standard token decoding.

        Args:
            tokens: Tokenized input text.
            id_to_classes: Class ID to name mapping.
            model_output: Token-level logits (B, W, C, 3).
            flat_ner: Whether to enforce non-overlapping spans.
            threshold: Confidence threshold.
            multi_label: Allow multiple labels per span.
            gen_labels: Generated labels from decoder.
            sel_idx: Selected span indices for span mode.
            num_gen_sequences: Number of sequences per span.
            span_logits: Span classification logits.
            span_idx: Span indices.
            span_mask: Span mask.
            **kwargs: Additional arguments.

        Returns:
            List of span tuples, with generated labels if available.
        """
        # Use generative decoding if labels_decoder is configured and gen_labels provided
        if self.config.labels_decoder is not None and gen_labels is not None:
            return self.decode_generative(
                tokens=tokens,
                id_to_classes=id_to_classes,
                model_output=model_output,
                gen_labels=gen_labels,
                sel_idx=sel_idx,
                num_gen_sequences=num_gen_sequences,
                flat_ner=flat_ner,
                threshold=threshold,
                multi_label=multi_label,
                span_logits=span_logits,
                span_idx=span_idx,
                span_mask=span_mask,
            )

        # Fall back to standard decoding without generative labels
        return super().decode(
            tokens=tokens,
            id_to_classes=id_to_classes,
            model_output=model_output,
            flat_ner=flat_ner,
            threshold=threshold,
            multi_label=multi_label,
            span_logits=span_logits,
            span_idx=span_idx,
            span_mask=span_mask,
        )
