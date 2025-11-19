from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union, Optional

from .processor import (
    BaseProcessor,
    BiEncoderSpanProcessor,
    BiEncoderTokenProcessor,
    UniEncoderSpanProcessor,
    UniEncoderTokenProcessor,
    UniEncoderSpanDecoderProcessor,
    RelationExtractionSpanProcessor,
)


class BaseDataCollator(ABC):
    """
    Abstract base class for all data collators.

    Provides common functionality for collating batches and preparing model inputs.
    Subclasses should implement processor-specific logic and field handling.
    """

    def __init__(
        self,
        config,
        data_processor: Optional[BaseProcessor] = None,
        return_tokens: bool = False,
        return_id_to_classes: bool = False,
        return_entities: bool = False,
        prepare_labels: bool = True,
    ):
        """
        Initialize the base data collator.

        Args:
            config: Configuration object containing model/training parameters.
            data_processor: Processor instance for handling data transformations.
                If None, subclass should provide a default processor.
            return_tokens: Whether to include tokenized text in output.
            return_id_to_classes: Whether to include class ID to name mappings.
            return_entities: Whether to include entity annotations.
            prepare_labels: Whether to prepare labels for training.
        """
        self.config = config
        self.data_processor = data_processor
        self.return_tokens = return_tokens
        self.return_id_to_classes = return_id_to_classes
        self.return_entities = return_entities
        self.prepare_labels = prepare_labels

    def collate_batch(self, input_x: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        """
        Collate raw input examples into a batch.

        Args:
            input_x: List of raw input examples.
            **kwargs: Additional arguments passed to the processor's collate_raw_batch.

        Returns:
            Dict containing collated raw batch data.
        """
        raw_batch = self.data_processor.collate_raw_batch(input_x, **kwargs)
        return raw_batch

    def collate_function(self, raw_batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Transform raw batch into model input format.

        Args:
            raw_batch: Raw collated batch from collate_batch.
            **kwargs: Additional arguments passed to the processor's collate_fn.

        Returns:
            Dict containing model-ready inputs.
        """
        model_input = self.data_processor.collate_fn(raw_batch, **kwargs)
        return model_input

    def _add_conditional_returns(self, model_input: Dict[str, Any], raw_batch: Dict[str, Any]) -> None:
        """
        Add optional fields to model input based on collator configuration.

        Args:
            model_input: Model input dictionary to update in-place.
            raw_batch: Raw batch containing optional fields.
        """
        if self.return_tokens:
            model_input["tokens"] = raw_batch.get("tokens")
        if self.return_id_to_classes:
            model_input["id_to_classes"] = raw_batch.get("id_to_classes")
        if self.return_entities:
            model_input["entities"] = raw_batch.get("entities")

    @staticmethod
    def _filter_none_values(data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Remove entries with None values from dictionary.

        Args:
            data: Dictionary potentially containing None values.

        Returns:
            Dictionary with all None values filtered out.
        """
        return {k: v for k, v in data.items() if v is not None}

    @staticmethod
    def _get_id_to_classes_for_sample(
        id_to_classes: Union[Dict[int, str], List[Dict[int, str]]], sample_idx: int
    ) -> Dict[int, str]:
        """
        Get id_to_classes mapping for a specific sample.

        Args:
            id_to_classes: Either a single mapping shared across all samples
                or per-sample mappings.
            sample_idx: Index of the sample in the batch.

        Returns:
            Mapping from class IDs to class names for this sample.
        """
        if isinstance(id_to_classes, list):
            return id_to_classes[sample_idx]
        return id_to_classes

    @abstractmethod
    def __call__(self, input_x: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        """
        Main collation method to be implemented by subclasses.

        Args:
            input_x: List of raw input examples.
            **kwargs: Additional collation arguments.

        Returns:
            Collated and processed batch ready for model input.
        """
        raise NotImplementedError("Subclasses must implement __call__ method")


class BaseSpanCollator(BaseDataCollator):
    """
    Base collator for span-based processors.

    Provides common logic for handling span indices, span masks, and span labels.
    Used by all span-level NER/RE models.
    """

    def _add_span_fields(self, model_input: Dict[str, Any], raw_batch: Dict[str, Any]) -> None:
        """
        Add span-specific fields to model input.

        Args:
            model_input: Model input dictionary to update in-place.
            raw_batch: Raw batch containing span-related fields.
        """
        model_input.update(
            {
                "span_idx": raw_batch.get("span_idx"),
                "span_mask": raw_batch.get("span_mask"),
                "text_lengths": raw_batch.get("seq_length"),
            }
        )


class BaseTokenCollator(BaseDataCollator):
    """
    Base collator for token-based processors.

    Provides common logic for handling token-level annotations and entity IDs.
    Used by all token-level NER models.
    """

    def _add_token_fields(self, model_input: Dict[str, Any], raw_batch: Dict[str, Any]) -> None:
        """
        Add token-specific fields to model input.

        Args:
            model_input: Model input dictionary to update in-place.
            raw_batch: Raw batch containing token-related fields.
        """
        model_input.update({"text_lengths": raw_batch.get("seq_length"), "entities_id": raw_batch.get("entities_id")})


class SpanDataCollator(BaseSpanCollator):
    """
    Unified data collator for all span-based processors.

    Handles span-based NER with various architectures:
    - UniEncoder: Single encoder with span classification
    - BiEncoder: Separate encoders for text and entity types
    - EncoderDecoder: Generative entity typing with decoder

    Automatically adapts behavior based on processor type.

    Required Processors: UniEncoderSpanProcessor, BiEncoderSpanProcessor,
                         or UniEncoderSpanDecoderProcessor
    """

    def __init__(
        self,
        config,
        data_processor: Optional[
            Union[UniEncoderSpanProcessor, BiEncoderSpanProcessor, UniEncoderSpanDecoderProcessor]
        ] = None,
        return_tokens: bool = False,
        return_id_to_classes: bool = False,
        return_entities: bool = False,
        prepare_labels: bool = True,
        prepare_entities: bool = True,
    ):
        """
        Initialize unified span collator.

        Args:
            config: Configuration object.
            data_processor: Span processor instance (Uni/Bi/EncoderDecoder).
            return_tokens: Whether to return tokenized text.
            return_id_to_classes: Whether to return class mappings.
            return_entities: Whether to return entity annotations.
            prepare_labels: Whether to prepare training labels.
            prepare_entities: Whether to encode entity types (BiEncoder only).
        """
        super().__init__(config, data_processor, return_tokens, return_id_to_classes, return_entities, prepare_labels)
        self.prepare_entities = prepare_entities

    def __call__(
        self, input_x: List[Dict[str, Any]], entity_types: Optional[Union[List[str], List[List[str]]]] = None, **kwargs
    ) -> Dict[str, Any]:
        """
        Collate batch for span-based model.

        Args:
            input_x: List of input examples with 'tokenized_text' and 'ner' fields.
            entity_types: Optional entity type specifications.
            **kwargs: Additional arguments for collation.

        Returns:
            Model-ready batch dictionary adapted to processor type.
        """
        raw_batch = self.collate_batch(input_x, entity_types=entity_types, **kwargs)

        # Build collate_fn kwargs based on processor type
        collate_kwargs = {"prepare_labels": self.prepare_labels, **kwargs}
        if isinstance(self.data_processor, BiEncoderSpanProcessor):
            collate_kwargs["prepare_entities"] = self.prepare_entities

        model_input = self.collate_function(raw_batch, **collate_kwargs)

        self._add_span_fields(model_input, raw_batch)

        # Add decoder-specific fields for EncoderDecoder models
        if isinstance(self.data_processor, UniEncoderSpanDecoderProcessor):
            decoder_fields = ["decoder_labels_ids", "decoder_labels_mask", "decoder_labels"]
            for field in decoder_fields:
                if field in model_input:
                    model_input[field] = model_input.get(field)

        self._add_conditional_returns(model_input, raw_batch)

        return self._filter_none_values(model_input)


class TokenDataCollator(BaseTokenCollator):
    """
    Unified data collator for all token-based processors.

    Handles token-level NER with various architectures:
    - UniEncoder: Single encoder with BIO/BIOES tagging
    - BiEncoder: Separate encoders for text and entity types

    Automatically adapts behavior based on processor type.

    Required Processors: UniEncoderTokenProcessor or BiEncoderTokenProcessor
    """

    def __init__(
        self,
        config,
        data_processor: Optional[Union[UniEncoderTokenProcessor, BiEncoderTokenProcessor]] = None,
        return_tokens: bool = False,
        return_id_to_classes: bool = False,
        return_entities: bool = False,
        prepare_labels: bool = True,
        prepare_entities: bool = True,
    ):
        """
        Initialize unified token collator.

        Args:
            config: Configuration object.
            data_processor: Token processor instance (Uni/Bi).
            return_tokens: Whether to return tokenized text.
            return_id_to_classes: Whether to return class mappings.
            return_entities: Whether to return entity annotations.
            prepare_labels: Whether to prepare training labels.
            prepare_entities: Whether to encode entity types (BiEncoder only).
        """
        super().__init__(config, data_processor, return_tokens, return_id_to_classes, return_entities, prepare_labels)
        self.prepare_entities = prepare_entities

    def __call__(
        self, input_x: List[Dict[str, Any]], entity_types: Optional[Union[List[str], List[List[str]]]] = None, **kwargs
    ) -> Dict[str, Any]:
        """
        Collate batch for token-based model.

        Args:
            input_x: List of input examples with 'tokenized_text' and 'ner' fields.
            entity_types: Optional entity type specifications.
            **kwargs: Additional arguments for collation.

        Returns:
            Model-ready batch with token-level labels adapted to processor type.
        """
        raw_batch = self.collate_batch(input_x, entity_types=entity_types, **kwargs)

        # Build collate_fn kwargs based on processor type
        collate_kwargs = {"prepare_labels": self.prepare_labels, **kwargs}
        if isinstance(self.data_processor, BiEncoderTokenProcessor):
            collate_kwargs["prepare_entities"] = self.prepare_entities

        model_input = self.collate_function(raw_batch, **collate_kwargs)

        self._add_token_fields(model_input, raw_batch)
        self._add_conditional_returns(model_input, raw_batch)

        return self._filter_none_values(model_input)


class RelationExtractionSpanDataCollator(BaseSpanCollator):
    """Data collator for RelationExtractionSpanProcessor.

    Handles joint entity and relation extraction at span level.
    Produces both entity labels and relation adjacency matrices.

    This collator is kept separate due to its unique handling of:
    - Relation adjacency matrices
    - Dual classification (entities + relations)
    - Relation-specific configuration

    Required Processor: RelationExtractionSpanProcessor
    """

    def __init__(
        self,
        config,
        data_processor: Optional[RelationExtractionSpanProcessor] = None,
        return_tokens: bool = False,
        return_id_to_classes: bool = False,
        return_entities: bool = False,
        return_rel_id_to_classes: bool = False,
        return_relations: bool = False,
        prepare_labels: bool = True,
    ):
        """
        Initialize RelationExtraction span collator.

        Args:
            config: Configuration object.
            data_processor: RelationExtractionSpanProcessor instance.
            return_tokens: Whether to return tokenized text.
            return_id_to_classes: Whether to return entity class mappings.
            return_entities: Whether to return entity annotations.
            return_rel_id_to_classes: Whether to return relation class mappings.
            return_relations: Whether to return relation annotations.
            prepare_labels: Whether to prepare training labels.
        """
        super().__init__(config, data_processor, return_tokens, return_id_to_classes, return_entities, prepare_labels)
        self.return_rel_id_to_classes = return_rel_id_to_classes
        self.return_relations = return_relations

    def collate_batch(
        self,
        input_x: List[Dict[str, Any]],
        entity_types: Optional[Union[List[str], List[List[str]]]] = None,
        relation_types: Optional[Union[List[str], List[List[str]]]] = None,
        ner_negatives: Optional[List[str]] = None,
        rel_negatives: Optional[List[str]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Collate raw batch data for relation extraction.

        Args:
            input_x: List of input examples.
            entity_types: Optional entity type specifications.
            relation_types: Optional relation type specifications.
            ner_negatives: Optional negative entity types for sampling.
            rel_negatives: Optional negative relation types for sampling.
            **kwargs: Additional arguments.

        Returns:
            Collated raw batch with entity and relation information.
        """
        if not self.data_processor:
            raise ValueError("data_processor must be provided for collate_batch")

        # Call processor's collate_raw_batch with both entity and relation types
        raw_batch = self.data_processor.collate_raw_batch(
            input_x,
            entity_types=entity_types,
            relation_types=relation_types,
            ner_negatives=ner_negatives,
            rel_negatives=rel_negatives,
            key="ner",
            **kwargs,
        )

        return raw_batch

    def __call__(
        self,
        input_x: List[Dict[str, Any]],
        entity_types: Optional[Union[List[str], List[List[str]]]] = None,
        relation_types: Optional[Union[List[str], List[List[str]]]] = None,
        ner_negatives: Optional[List[str]] = None,
        rel_negatives: Optional[List[str]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Collate batch for RelationExtraction span model.

        Args:
            input_x: List of input examples with 'tokenized_text', 'ner',
                and 'relations' fields.
            entity_types: Optional entity type specifications.
            relation_types: Optional relation type specifications.
            ner_negatives: Optional negative entity types for sampling.
            rel_negatives: Optional negative relation types for sampling.
            **kwargs: Additional arguments for collation.

        Returns:
            Model-ready batch with entity spans, relation adjacency matrix,
            and relation classification targets.
        """
        # Collate raw batch with both entity and relation types
        raw_batch = self.collate_batch(
            input_x,
            entity_types=entity_types,
            relation_types=relation_types,
            ner_negatives=ner_negatives,
            rel_negatives=rel_negatives,
            **kwargs,
        )

        model_input = self.collate_function(
            raw_batch, prepare_labels=self.prepare_labels, prepare_entities=True, **kwargs
        )

        self._add_span_fields(model_input, raw_batch)

        self._add_conditional_returns(model_input, raw_batch)

        if self.return_rel_id_to_classes:
            model_input["rel_id_to_classes"] = raw_batch.get("rel_id_to_classes")

        if "rel_class_to_ids" in raw_batch:
            model_input["rel_class_to_ids"] = raw_batch["rel_class_to_ids"]

        if self.return_relations:
            model_input["relations"] = raw_batch.get("relations")
        return self._filter_none_values(model_input)

    def _filter_none_values(self, batch_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Remove None values from batch dictionary.

        Args:
            batch_dict: Dictionary potentially containing None values.

        Returns:
            Dictionary with None values removed.
        """
        return {k: v for k, v in batch_dict.items() if v is not None}


class UniEncoderSpanDataCollator(SpanDataCollator):
    """
    Backward compatibility alias for SpanDataCollator with UniEncoderSpanProcessor.

    Use SpanDataCollator directly for new code.
    """

    pass


class BiEncoderSpanDataCollator(SpanDataCollator):
    """
    Backward compatibility alias for SpanDataCollator with BiEncoderSpanProcessor.

    Use SpanDataCollator directly for new code.
    """

    pass


class UniEncoderSpanDecoderDataCollator(SpanDataCollator):
    """
    Backward compatibility alias for SpanDataCollator with EncoderDecoderSpanProcessor.

    Use SpanDataCollator directly for new code.
    """

    pass


class UniEncoderTokenDataCollator(TokenDataCollator):
    """
    Backward compatibility alias for TokenDataCollator with UniEncoderTokenProcessor.

    Use TokenDataCollator directly for new code.
    """

    pass


class BiEncoderTokenDataCollator(TokenDataCollator):
    """
    Backward compatibility alias for TokenDataCollator with BiEncoderTokenProcessor.

    Use TokenDataCollator directly for new code.
    """

    pass
