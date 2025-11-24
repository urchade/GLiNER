import random
import warnings
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Union, Optional, Sequence
from collections import defaultdict

import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from .utils import make_mapping, get_negatives, pad_2d_tensor, prepare_span_idx, prepare_word_mask
from .tokenizer import WordsSplitter


class BaseProcessor(ABC):
    """Abstract base class for data processors.

    This class provides the common interface and utilities for all processor
    implementations, handling tokenization, label preparation, and batch collation
    for NER and RE tasks.
    """

    def __init__(self, config, tokenizer, words_splitter):
        """Initialize the base processor.

        Args:
            config: Configuration object containing model and processing parameters.
            tokenizer: Transformer tokenizer for subword tokenization.
            words_splitter: Word-level tokenizer/splitter. If None, creates one
                based on config.words_splitter_type.
        """
        self.config = config
        self.transformer_tokenizer = tokenizer
        if words_splitter is None:
            self.words_splitter = WordsSplitter(splitter_type=config.words_splitter_type)
        else:
            self.words_splitter = words_splitter
        self.ent_token = config.ent_token
        self.sep_token = config.sep_token

        # Check if the tokenizer has unk_token and pad_token
        self._check_and_set_special_tokens(self.transformer_tokenizer)

    def _check_and_set_special_tokens(self, tokenizer):
        """Check and set special tokens for the tokenizer.

        Ensures the tokenizer has necessary special tokens (unk_token, pad_token).
        If pad_token is missing, attempts to use eos_token as a fallback.

        Args:
            tokenizer: The tokenizer to check and modify.

        Warnings:
            UserWarning: If unk_token or pad_token is missing.
        """
        if tokenizer.unk_token is None:
            if hasattr(tokenizer, "unk_token_id") and tokenizer.unk_token_id is not None:
                # Tokenizer has unk_token_id but not unk_token
                pass
            else:
                warnings.warn("Tokenizer missing 'unk_token'. This may cause issues.", UserWarning, stacklevel=2)

        if tokenizer.pad_token is None:
            # Try to use eos_token as pad_token (common practice)
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                warnings.warn(
                    "Tokenizer missing 'pad_token'. Consider setting it explicitly.", UserWarning, stacklevel=2
                )

    @staticmethod
    def get_dict(spans: List[Tuple[int, int, str]], classes_to_id: Dict[str, int]) -> Dict[Tuple[int, int], int]:
        """Create a dictionary mapping spans to their class IDs.

        Args:
            spans: List of tuples (start, end, label) representing entity spans.
            classes_to_id: Mapping from class labels to integer IDs.

        Returns:
            Dictionary mapping (start, end) tuples to class IDs.
        """
        dict_tag = defaultdict(int)
        for span in spans:
            if span[2] in classes_to_id:
                dict_tag[(span[0], span[1])] = classes_to_id[span[2]]
        return dict_tag

    @abstractmethod
    def preprocess_example(
        self, tokens: List[str], ner: List[Tuple[int, int, str]], classes_to_id: Dict[str, int]
    ) -> Dict:
        """Preprocess a single example for model input.

        Args:
            tokens: List of token strings.
            ner: List of NER annotations as (start, end, label) tuples.
            classes_to_id: Mapping from class labels to integer IDs.

        Returns:
            Dictionary containing preprocessed example data.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses should implement this method")

    @abstractmethod
    def create_labels(self) -> torch.Tensor:
        """Create label tensors from batch data.

        Returns:
            Tensor containing labels for the batch.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses should implement this method")

    @abstractmethod
    def tokenize_and_prepare_labels(self):
        """Tokenize inputs and prepare labels for a batch.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        pass

    def prepare_inputs(
        self,
        texts: Sequence[Sequence[str]],
        entities: Union[Sequence[Sequence[str]], Dict[int, Sequence[str]], Sequence[str]],
        blank: Optional[str] = None,
        add_entities: Optional[bool] = True,
        **kwargs,
    ) -> Tuple[List[List[str]], List[int]]:
        """Prepare input texts with entity type prompts.

        Prepends entity type special tokens that aggregates entity label information.

        Args:
            texts: Sequences of token strings, one per example.
            entities: Entity types to extract. Can be:
                - List of lists (per-example entity types)
                - Dictionary (shared entity types)
                - List of strings (same types for all examples)
            blank: Optional blank entity token for zero-shot scenarios.
            add_entities: Whether to add entity text string to the prompt.
            **kwargs: Additional keyword arguments.

        Returns:
            Tuple containing:
                - List of input text sequences with prepended prompts
                - List of prompt lengths for each example
        """
        input_texts: List[List[str]] = []
        prompt_lengths: List[int] = []

        for i, text in enumerate(texts):
            ents = self._select_entities(i, entities, blank)

            ents = self._maybe_remap_entities(ents)
            prompt: List[str] = []
            for ent in ents:
                prompt.append(self.ent_token)
                if add_entities:
                    prompt.append(str(ent))

            prompt += self._extra_prompt_tokens(i, text, ents)

            prompt.append(self.sep_token)
            prompt_lengths.append(len(prompt))
            input_texts.append(prompt + list(text))
        return input_texts, prompt_lengths

    def _select_entities(
        self,
        i: int,
        entities: Union[Sequence[Sequence[str]], Dict[int, Sequence[str]], Sequence[str]],
        blank: Optional[str] = None,
    ) -> List[str]:
        """Select entities for a specific example.

        Args:
            i: Index of the example.
            entities: Entity specifications (see prepare_inputs).
            blank: Optional blank entity token.

        Returns:
            List of entity type strings for this example.
        """
        if blank is not None:
            return [blank]
        if isinstance(entities, dict):
            return list(entities)
        if entities and isinstance(entities[0], (list, tuple, dict)):  # per-item lists
            return list(entities[i])  # type: ignore[index]
        if entities and isinstance(entities[0], str):  # same for all
            return list(entities)  # type: ignore[list-item]
        return []

    def _maybe_remap_entities(self, ents: Sequence[str]) -> List[str]:
        """Optionally remap entity types.

        Default implementation returns entities as-is. Subclasses can override
        to provide custom entity type remapping.

        Args:
            ents: Sequence of entity type strings.

        Returns:
            List of (potentially remapped) entity type strings.
        """
        return list(ents)

    def _extra_prompt_tokens(self, i: int, text: Sequence[str], ents: Sequence[str]) -> List[str]:
        """Add extra tokens to the prompt.

        Default implementation returns no extra tokens. Subclasses can override
        to add custom prompt tokens.

        Args:
            i: Index of the example.
            text: The text sequence.
            ents: The entity types for this example.

        Returns:
            List of extra prompt tokens (default: empty list).
        """
        return []

    def prepare_word_mask(self, texts, tokenized_inputs, skip_first_words=None, token_level=False):
        """Prepare word-level masks for tokenized inputs.

        Creates masks that map subword tokens back to their original words.

        Args:
            texts: Original text sequences.
            tokenized_inputs: Tokenized inputs from transformer tokenizer.
            skip_first_words: Optional list of word counts to skip per example
                (e.g., prompt words).
            token_level: If True, create token-level masks instead of word-level.

        Returns:
            Word mask array.
        """
        return prepare_word_mask(
            texts,
            tokenized_inputs,
            skip_first_words=skip_first_words,
            token_level=token_level,
        )

    def tokenize_inputs(self, texts, entities, blank=None, **kwargs):
        """Tokenize input texts with entity prompts.

        Args:
            texts: Sequences of token strings.
            entities: Entity types for extraction.
            blank: Optional blank entity token.
            **kwargs: Additional keyword arguments.

        Returns:
            Dictionary containing tokenized inputs with keys:
                - input_ids: Token IDs
                - attention_mask: Attention mask
                - words_mask: Word-level mask
        """
        input_texts, prompt_lengths = self.prepare_inputs(texts, entities, blank=blank, **kwargs)

        tokenized_inputs = self.transformer_tokenizer(
            input_texts,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            padding="longest",
        )
        words_masks = self.prepare_word_mask(texts, tokenized_inputs, prompt_lengths)
        tokenized_inputs["words_mask"] = torch.tensor(words_masks)

        return tokenized_inputs

    def batch_generate_class_mappings(
        self, batch_list: List[Dict], negatives: Optional[List[str]] = None, key: str = "ner", sampled_neg: int = 100
    ) -> Tuple[List[Dict[str, int]], List[Dict[int, str]]]:
        """Generate class mappings for a batch with negative sampling.

        Creates bidirectional mappings between class labels and integer IDs,
        with support for negative type sampling to improve model robustness.

        Args:
            batch_list: List of example dictionaries.
            negatives: Optional pre-sampled negative types. If None, samples
                from batch.
            key: Key to access labels in batch dictionaries (default: 'ner').
            sampled_neg: Number of negative types to sample if negatives is None.

        Returns:
            Tuple containing:
                - List of class-to-ID mappings (one per example)
                - List of ID-to-class mappings (one per example)
        """
        if negatives is None:
            negatives = get_negatives(batch_list, sampled_neg=sampled_neg, key=key)
        class_to_ids = []
        id_to_classes = []
        for b in batch_list:
            max_neg_type_ratio = int(self.config.max_neg_type_ratio)
            neg_type_ratio = random.randint(0, max_neg_type_ratio) if max_neg_type_ratio else 0

            if f"{key}_negatives" in b:  # manually setting negative types
                negs_i = b[f"{key}_negatives"]
            else:  # in-batch negative types
                negs_i = negatives[: len(b[key]) * neg_type_ratio] if neg_type_ratio else []

            if f"{key}_labels" in b:  # labels are predefined
                types = b[f"{key}_labels"]
            else:
                types = list(set([el[-1] for el in b[key]] + negs_i))
                random.shuffle(types)
                types = types[: int(self.config.max_types)]

            class_to_id = {k: v for v, k in enumerate(types, start=1)}
            id_to_class = {k: v for v, k in class_to_id.items()}
            class_to_ids.append(class_to_id)
            id_to_classes.append(id_to_class)

        return class_to_ids, id_to_classes

    def collate_raw_batch(
        self,
        batch_list: List[Dict],
        entity_types: Optional[List[Union[str, List[str]]]] = None,
        negatives: Optional[List[str]] = None,
        class_to_ids: Optional[Union[Dict[str, int], List[Dict[str, int]]]] = None,
        id_to_classes: Optional[Union[Dict[int, str], List[Dict[int, str]]]] = None,
        key="ner",
    ) -> Dict:
        """Collate a raw batch with optional dynamic or provided label mappings.

        Args:
            batch_list: List of raw example dictionaries.
            entity_types: Optional predefined entity types. Can be a single list
                for all examples or list of lists for per-example types.
            negatives: Optional list of negative entity types.
            class_to_ids: Optional predefined class-to-ID mapping(s).
            id_to_classes: Optional predefined ID-to-class mapping(s).
            key: Key for accessing labels in batch (default: 'ner').

        Returns:
            Dictionary containing collated batch data ready for model input.
        """
        if class_to_ids is None and entity_types is None:
            # Dynamically infer per-example mappings
            class_to_ids, id_to_classes = self.batch_generate_class_mappings(batch_list, negatives)
        elif class_to_ids is None:
            # Build mappings from entity_types
            if entity_types and isinstance(entity_types[0], list):
                # Per-example mappings
                built = [make_mapping(t) for t in entity_types]  # list of (fwd, rev)
                class_to_ids, id_to_classes = list(zip(*built))
                class_to_ids, id_to_classes = list(class_to_ids), list(id_to_classes)
            else:
                # Single mapping for all examples
                class_to_ids, id_to_classes = make_mapping(entity_types or [])

        if isinstance(class_to_ids, list):
            batch = [
                self.preprocess_example(b["tokenized_text"], b[key], class_to_ids[i]) for i, b in enumerate(batch_list)
            ]
        else:
            batch = [self.preprocess_example(b["tokenized_text"], b[key], class_to_ids) for b in batch_list]

        return self.create_batch_dict(batch, class_to_ids, id_to_classes)

    def collate_fn(self, batch, prepare_labels=True, *args, **kwargs):
        """Collate function for DataLoader.

        Args:
            batch: Batch of examples from dataset.
            prepare_labels: Whether to prepare labels (default: True).
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Dictionary containing model inputs and labels.
        """
        model_input_batch = self.tokenize_and_prepare_labels(batch, prepare_labels, *args, **kwargs)
        return model_input_batch

    @abstractmethod
    def create_batch_dict(
        self, batch: List[Dict], class_to_ids: List[Dict[str, int]], id_to_classes: List[Dict[int, str]]
    ) -> Dict:
        """Create a batch dictionary from preprocessed examples.

        Args:
            batch: List of preprocessed example dictionaries.
            class_to_ids: List of class-to-ID mappings.
            id_to_classes: List of ID-to-class mappings.

        Returns:
            Dictionary containing collated batch tensors.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses should implement this method")

    def create_dataloader(self, data, entity_types=None, *args, **kwargs) -> DataLoader:
        """Create a PyTorch DataLoader with the processor's collate function.

        Args:
            data: Dataset to load.
            entity_types: Optional entity types for extraction.
            *args: Additional positional arguments for DataLoader.
            **kwargs: Additional keyword arguments for DataLoader.

        Returns:
            DataLoader instance configured with this processor's collate_fn.
        """
        return DataLoader(data, *args, collate_fn=lambda x: self.collate_fn(x, entity_types), **kwargs)


class UniEncoderSpanProcessor(BaseProcessor):
    """Processor for span-based NER with uni-encoder architecture.

    This processor handles span enumeration and labeling for models that
    predict entity types for all possible spans up to a maximum width.
    """

    def preprocess_example(self, tokens, ner, classes_to_id):
        """Preprocess a single example for span-based prediction.

        Enumerates all possible spans up to max_width and creates labels
        for each span based on NER annotations.

        Args:
            tokens: List of token strings.
            ner: List of NER annotations as (start, end, label) tuples.
            classes_to_id: Mapping from class labels to integer IDs.

        Returns:
            Dictionary containing:
                - tokens: Token strings
                - span_idx: Tensor of span indices (start, end)
                - span_label: Tensor of span labels
                - seq_length: Sequence length
                - entities: Original NER annotations

        Warnings:
            UserWarning: If sequence length exceeds max_len (gets truncated).
        """
        max_width = self.config.max_width
        num_tokens = len(tokens)
        if num_tokens == 0:
            tokens = ["[PAD]"]
        max_len = self.config.max_len
        if num_tokens > max_len:
            warnings.warn(f"Sentence of length {num_tokens} has been truncated to {max_len}", stacklevel=2)
            tokens = tokens[:max_len]
        num_tokens = len(tokens)
        spans_idx = prepare_span_idx(num_tokens, max_width)
        dict_lab = self.get_dict(ner, classes_to_id) if ner else defaultdict(int)
        span_label = torch.LongTensor([dict_lab[i] for i in spans_idx])
        spans_idx = torch.LongTensor(spans_idx)
        valid_span_mask = spans_idx[:, 1] > num_tokens - 1
        span_label = span_label.masked_fill(valid_span_mask, -1)

        return {
            "tokens": tokens,
            "span_idx": spans_idx,
            "span_label": span_label,
            "seq_length": num_tokens,
            "entities": ner,
        }

    def create_batch_dict(self, batch, class_to_ids, id_to_classes):
        """Create a batch dictionary from preprocessed span examples.

        Args:
            batch: List of preprocessed example dictionaries.
            class_to_ids: List of class-to-ID mappings.
            id_to_classes: List of ID-to-class mappings.

        Returns:
            Dictionary containing:
                - seq_length: Sequence lengths
                - span_idx: Padded span indices
                - tokens: Token strings
                - span_mask: Mask for valid spans
                - span_label: Padded span labels
                - entities: Original NER annotations
                - classes_to_id: Class mappings
                - id_to_classes: Reverse class mappings
        """
        tokens = [el["tokens"] for el in batch]
        entities = [el["entities"] for el in batch]
        span_idx = pad_sequence([b["span_idx"] for b in batch], batch_first=True, padding_value=0)
        span_label = pad_sequence([el["span_label"] for el in batch], batch_first=True, padding_value=-1)
        seq_length = torch.LongTensor([el["seq_length"] for el in batch]).unsqueeze(-1)
        span_mask = span_label != -1

        return {
            "seq_length": seq_length,
            "span_idx": span_idx,
            "tokens": tokens,
            "span_mask": span_mask,
            "span_label": span_label,
            "entities": entities,
            "classes_to_id": class_to_ids,
            "id_to_classes": id_to_classes,
        }

    def create_labels(self, batch):
        """Create one-hot encoded labels for spans.

        Creates multi-label one-hot vectors for each span, allowing spans
        to have multiple entity types.

        Args:
            batch: Batch dictionary containing tokens, entities, and class mappings.

        Returns:
            Tensor of shape (batch_size, max_spans, num_classes) containing
            one-hot encoded labels.
        """
        labels_batch = []
        for i in range(len(batch["tokens"])):
            tokens = batch["tokens"][i]
            classes_to_id = batch["classes_to_id"][i]
            ner = batch["entities"][i]
            num_classes = len(classes_to_id)
            spans_idx = torch.LongTensor(prepare_span_idx(len(tokens), self.config.max_width))
            span_to_index = {(spans_idx[idx, 0].item(), spans_idx[idx, 1].item()): idx for idx in range(len(spans_idx))}
            labels_one_hot = torch.zeros(len(spans_idx), num_classes + 1, dtype=torch.float)
            end_token_idx = len(tokens) - 1
            span_labels_dict = {}
            for start, end, label in ner:
                span = (start, end)
                if label in classes_to_id and span in span_to_index:
                    idx = span_to_index[span]
                    class_id = classes_to_id[label]
                    labels_one_hot[idx, class_id] = 1.0
                    span_labels_dict[idx] = label
            valid_span_mask = spans_idx[:, 1] > end_token_idx
            labels_one_hot[valid_span_mask, :] = 0.0
            labels_one_hot = labels_one_hot[:, 1:]
            labels_batch.append(labels_one_hot)
        labels_batch = pad_2d_tensor(labels_batch) if len(labels_batch) > 1 else labels_batch[0].unsqueeze(0)
        return labels_batch

    def tokenize_and_prepare_labels(self, batch, prepare_labels, *args, **kwargs):
        """Tokenize inputs and prepare span labels for a batch.

        Args:
            batch: Batch dictionary with tokens and class mappings.
            prepare_labels: Whether to prepare labels.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Dictionary containing tokenized inputs and optionally labels.
        """
        tokenized_input = self.tokenize_inputs(batch["tokens"], batch["classes_to_id"])
        if prepare_labels:
            labels = self.create_labels(batch)
            tokenized_input["labels"] = labels

        return tokenized_input


class UniEncoderTokenProcessor(BaseProcessor):
    """Processor for token-based NER with uni-encoder architecture.

    This processor handles token-level classification where each token is
    labeled with BIO-style tags (Begin, Inside, Outside) for each entity type.
    """

    def preprocess_example(self, tokens, ner, classes_to_id):
        """Preprocess a single example for token-based prediction.

        Args:
            tokens: List of token strings.
            ner: List of NER annotations as (start, end, label) tuples.
            classes_to_id: Mapping from class labels to integer IDs.

        Returns:
            Dictionary containing:
                - tokens: Token strings
                - seq_length: Sequence length
                - entities: Original NER annotations
                - entities_id: Entity annotations with class IDs

        Warnings:
            UserWarning: If sequence length exceeds max_len (gets truncated).
        """
        # Ensure there is always a token list, even if it's empty
        if len(tokens) == 0:
            tokens = ["[PAD]"]

        # Limit the length of tokens based on configuration maximum length
        max_len = self.config.max_len
        if len(tokens) > max_len:
            warnings.warn(f"Sentence of length {len(tokens)} has been truncated to {max_len}", stacklevel=2)
            tokens = tokens[:max_len]

        # Generate entity IDs based on the NER spans provided and their classes
        try:  # 'NoneType' object is not iterable
            entities_id = [[i, j, classes_to_id[k]] for i, j, k in ner if k in classes_to_id]
        except TypeError:
            entities_id = []

        example = {"tokens": tokens, "seq_length": len(tokens), "entities": ner, "entities_id": entities_id}
        return example

    def create_batch_dict(self, batch, class_to_ids, id_to_classes):
        """Create a batch dictionary from preprocessed token examples.

        Args:
            batch: List of preprocessed example dictionaries.
            class_to_ids: List of class-to-ID mappings.
            id_to_classes: List of ID-to-class mappings.

        Returns:
            Dictionary containing:
                - tokens: Token strings
                - seq_length: Sequence lengths
                - entities: Original NER annotations
                - entities_id: Entity annotations with class IDs
                - classes_to_id: Class mappings
                - id_to_classes: Reverse class mappings
        """
        # Extract relevant data from batch for batch processing
        tokens = [el["tokens"] for el in batch]
        seq_length = torch.LongTensor([el["seq_length"] for el in batch]).unsqueeze(-1)
        entities = [el["entities"] for el in batch]
        entities_id = [el["entities_id"] for el in batch]

        # Assemble and return the batch dictionary
        batch_dict = {
            "tokens": tokens,
            "seq_length": seq_length,
            "entities": entities,
            "entities_id": entities_id,
            "classes_to_id": class_to_ids,
            "id_to_classes": id_to_classes,
        }

        return batch_dict

    def create_labels(self, entities_id, batch_size, seq_len, num_classes):
        """Create token-level labels with begin/inside/end markers.

        Creates labels indicating which tokens are at the start, end, or inside
        of entity spans for each entity type.

        Args:
            entities_id: List of entity annotations with class IDs for each example.
            batch_size: Size of the batch.
            seq_len: Maximum sequence length in batch.
            num_classes: Number of entity classes.

        Returns:
            Tensor of shape (batch_size, seq_len, num_classes, 3) where the last
            dimension contains [start_marker, end_marker, inside_marker].
        """
        word_labels = torch.zeros(batch_size, seq_len, num_classes, 3, dtype=torch.float)

        for i, sentence_entities in enumerate(entities_id):
            for st, ed, sp_label in sentence_entities:
                class_idx = sp_label - 1  # Convert to 0-indexed

                # skip entities that point beyond sequence length
                if st >= seq_len or ed >= seq_len:
                    continue

                word_labels[i, st, class_idx, 0] = 1  # start token
                word_labels[i, ed, class_idx, 1] = 1  # end token
                word_labels[i, st : ed + 1, class_idx, 2] = 1  # inside tokens (inclusive)

        return word_labels

    def tokenize_and_prepare_labels(self, batch, prepare_labels, *args, **kwargs):
        """Tokenize inputs and prepare token-level labels for a batch.

        Args:
            batch: Batch dictionary with tokens and class mappings.
            prepare_labels: Whether to prepare labels.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Dictionary containing tokenized inputs and optionally labels.
        """
        batch_size = len(batch["tokens"])
        seq_len = batch["seq_length"].max()
        num_classes = max([len(cid) for cid in batch["classes_to_id"]])

        tokenized_input = self.tokenize_inputs(batch["tokens"], batch["classes_to_id"])

        if prepare_labels:
            labels = self.create_labels(batch["entities_id"], batch_size, seq_len, num_classes)
            tokenized_input["labels"] = labels
        return tokenized_input


class BaseBiEncoderProcessor(BaseProcessor):
    """Base processor for bi-encoder architectures.

    Bi-encoder models use separate encoders for text and entity types.
    """

    def __init__(self, config, tokenizer, words_splitter, labels_tokenizer):
        """Initialize the bi-encoder processor.

        Args:
            config: Configuration object.
            tokenizer: Transformer tokenizer for text encoding.
            words_splitter: Word-level tokenizer/splitter.
            labels_tokenizer: Separate tokenizer for entity type encoding.
        """
        super().__init__(config, tokenizer, words_splitter)
        self.labels_tokenizer = labels_tokenizer

        # Check special tokens for additional tokenizers
        if self.labels_tokenizer:
            self._check_and_set_special_tokens(self.labels_tokenizer)

    def tokenize_inputs(self, texts, entities=None):
        """Tokenize inputs for bi-encoder architecture.

        Separately tokenizes text sequences and entity types using different
        tokenizers.

        Args:
            texts: Sequences of token strings.
            entities: Optional list of entity types to encode.

        Returns:
            Dictionary containing:
                - input_ids: Text token IDs
                - attention_mask: Text attention mask
                - words_mask: Word-level mask
                - labels_input_ids: Entity type token IDs (if entities provided)
                - labels_attention_mask: Entity type attention mask (if entities provided)
        """
        tokenized_inputs = self.transformer_tokenizer(
            texts, is_split_into_words=True, return_tensors="pt", truncation=True, padding="longest"
        )

        if entities is not None:
            tokenized_labels = self.labels_tokenizer(entities, return_tensors="pt", truncation=True, padding="longest")

            tokenized_inputs["labels_input_ids"] = tokenized_labels["input_ids"]
            tokenized_inputs["labels_attention_mask"] = tokenized_labels["attention_mask"]

        words_masks = self.prepare_word_mask(texts, tokenized_inputs, skip_first_words=None)
        tokenized_inputs["words_mask"] = torch.tensor(words_masks)
        return tokenized_inputs

    def batch_generate_class_mappings(
        self, batch_list: List[Dict], *args
    ) -> Tuple[List[Dict[str, int]], List[Dict[int, str]]]:
        """Generate class mappings for bi-encoder with batch-level type pooling.

        Unlike uni-encoder which generates per-example mappings, bi-encoder
        creates a single shared mapping across the batch for more efficient
        entity type encoding.

        Args:
            batch_list: List of example dictionaries.
            *args: Variable length argument list (unused).

        Returns:
            Tuple containing:
                - List of identical class-to-ID mappings (one per example)
                - List of identical ID-to-class mappings (one per example)
        """
        classes = []
        for b in batch_list:
            if "ner_negatives" in b:  # manually setting negative types
                negs_i = b["ner_negatives"]
            else:  # in-batch negative types
                negs_i = []

            types = list(set([el[-1] for el in b["ner"]] + negs_i))

            if "ner_label" in b:  # labels are predefined
                types = b["ner_label"]

            classes.extend(types)
        random.shuffle(classes)
        classes = list(set(classes))[: int(self.config.max_types * len(batch_list))]
        class_to_id = {k: v for v, k in enumerate(classes, start=1)}
        id_to_class = {k: v for v, k in class_to_id.items()}

        class_to_ids = [class_to_id for i in range(len(batch_list))]
        id_to_classes = [id_to_class for i in range(len(batch_list))]

        return class_to_ids, id_to_classes


class BiEncoderSpanProcessor(UniEncoderSpanProcessor, BaseBiEncoderProcessor):
    """Processor for span-based NER with bi-encoder architecture.

    Combines span enumeration from UniEncoderSpanProcessor with the bi-encoder
    approach from BaseBiEncoderProcessor.
    """

    def tokenize_and_prepare_labels(self, batch, prepare_labels, prepare_entities=True, *args, **kwargs):
        """Tokenize inputs and prepare span labels for bi-encoder.

        Args:
            batch: Batch dictionary with tokens and class mappings.
            prepare_labels: Whether to prepare labels.
            prepare_entities: Whether to encode entity types separately.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Dictionary containing tokenized inputs, entity encodings, and optionally labels.
        """
        if prepare_entities:
            if isinstance(batch["classes_to_id"], dict):
                entities = list(batch["classes_to_id"])
            else:
                entities = list(batch["classes_to_id"][0])
        else:
            entities = None
        tokenized_input = self.tokenize_inputs(batch["tokens"], entities)
        if prepare_labels:
            labels = self.create_labels(batch)
            tokenized_input["labels"] = labels
        return tokenized_input


class BiEncoderTokenProcessor(UniEncoderTokenProcessor, BaseBiEncoderProcessor):
    """Processor for token-based NER with bi-encoder architecture.

    Combines token-level classification from UniEncoderTokenProcessor with the
    dual-encoder approach from BaseBiEncoderProcessor.
    """

    def tokenize_and_prepare_labels(self, batch, prepare_labels, prepare_entities=True, **kwargs):
        """Tokenize inputs and prepare token-level labels for bi-encoder.

        Args:
            batch: Batch dictionary with tokens and class mappings.
            prepare_labels: Whether to prepare labels.
            prepare_entities: Whether to encode entity types separately.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Dictionary containing tokenized inputs, entity encodings, and optionally labels.
        """
        if prepare_entities:
            if isinstance(batch["classes_to_id"], dict):
                entities = list(batch["classes_to_id"])
            else:
                entities = list(batch["classes_to_id"][0])
        else:
            entities = None
        batch_size = len(batch["tokens"])
        seq_len = batch["seq_length"].max()
        num_classes = len(entities)

        tokenized_input = self.tokenize_inputs(batch["tokens"], entities)

        if prepare_labels:
            labels = self.create_labels(batch["entities_id"], batch_size, seq_len, num_classes)
            tokenized_input["labels"] = labels

        return tokenized_input


class UniEncoderSpanDecoderProcessor(UniEncoderSpanProcessor):
    """Processor for span-based NER with encoder-decoder architecture.

    Extends span-based processing with a decoder that generates entity type
    labels autoregressively, enabling more flexible prediction strategies.
    """

    def __init__(self, config, tokenizer, words_splitter, decoder_tokenizer):
        """Initialize the encoder-decoder processor.

        Args:
            config: Configuration object.
            tokenizer: Transformer tokenizer for encoding.
            words_splitter: Word-level tokenizer/splitter.
            decoder_tokenizer: Separate tokenizer for decoder (label generation).
        """
        super().__init__(config, tokenizer, words_splitter)
        self.decoder_tokenizer = decoder_tokenizer

        # Check special tokens for additional tokenizers
        if self.decoder_tokenizer:
            self._check_and_set_special_tokens(self.decoder_tokenizer)

    def tokenize_inputs(self, texts, entities, blank=None):
        """Tokenize inputs for encoder-decoder architecture.

        Prepares both encoder and decoder inputs, with optional decoder context
        based on configuration.

        Args:
            texts: Sequences of token strings.
            entities: Entity types for extraction.
            blank: Optional blank entity token for zero-shot scenarios.

        Returns:
            Dictionary containing encoder and decoder tokenized inputs.
        """
        add_entities = True
        if self.config.decoder_mode == 'prompt':
            add_entities = False

        input_texts, prompt_lengths = self.prepare_inputs(texts, entities, blank=blank, add_entities=add_entities)

        tokenized_inputs = self.transformer_tokenizer(
            input_texts,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            padding="longest",
        )
        words_masks = self.prepare_word_mask(texts, tokenized_inputs, skip_first_words=prompt_lengths)
        tokenized_inputs["words_mask"] = torch.tensor(words_masks)

        # Add decoder inputs if decoder tokenizer is available and mode is 'span'
        if self.config.decoder_mode == "span":
            decoder_input_texts = [[f" {t}" if i else t for i, t in enumerate(tokens)] for tokens in input_texts]
            decoder_tokenized_inputs = self.decoder_tokenizer(
                decoder_input_texts,
                is_split_into_words=True,
                return_tensors="pt",
                truncation=True,
                padding="longest",
            )
            tokenized_inputs["decoder_input_ids"] = decoder_tokenized_inputs["input_ids"]
            tokenized_inputs["decoder_attention_mask"] = decoder_tokenized_inputs["attention_mask"]

            if self.config.full_decoder_context:
                decoder_words_masks = self.prepare_word_mask(
                    texts, decoder_tokenized_inputs, skip_first_words=prompt_lengths, token_level=True
                )
                tokenized_inputs["decoder_words_mask"] = torch.tensor(decoder_words_masks)

        return tokenized_inputs

    def create_labels(self, batch, blank=None):
        """Create labels for both span classification and decoder generation.

        Args:
            batch: Batch dictionary containing tokens, entities, and class mappings.
            blank: Optional blank entity token for zero-shot scenarios.

        Returns:
            Tuple containing:
                - Span classification labels (one-hot encoded)
                - Decoder generation labels (tokenized entity types) or None
        """
        labels_batch = []
        decoder_label_strings = []

        for i in range(len(batch["tokens"])):
            tokens = batch["tokens"][i]
            classes_to_id = batch["classes_to_id"][i]
            ner = batch["entities"][i]
            num_classes = len(classes_to_id)

            spans_idx = torch.LongTensor(prepare_span_idx(len(tokens), self.config.max_width))
            span_to_index = {(spans_idx[idx, 0].item(), spans_idx[idx, 1].item()): idx for idx in range(len(spans_idx))}

            if blank is not None:
                num_classes = 1

            labels_one_hot = torch.zeros(len(spans_idx), num_classes + 1, dtype=torch.float)
            end_token_idx = len(tokens) - 1
            used_spans = set()
            span_labels_dict = {}

            for start, end, label in ner:
                span = (start, end)
                if label in classes_to_id and span in span_to_index:
                    idx = span_to_index[span]
                    if self.config.decoder_mode == "span":
                        class_id = classes_to_id[label] if blank is None else 1
                    else:
                        class_id = classes_to_id[label]

                    if labels_one_hot[idx, class_id] == 0 and idx not in used_spans:
                        used_spans.add(idx)
                        if end <= end_token_idx:
                            labels_one_hot[idx, class_id] = 1.0
                            span_labels_dict[idx] = label

            valid_span_mask = spans_idx[:, 1] > end_token_idx
            labels_one_hot[valid_span_mask, :] = 0.0
            labels_one_hot = labels_one_hot[:, 1:]
            labels_batch.append(labels_one_hot)
            
            if self.config.decoder_mode == 'span':
                # Collect decoder label strings in order
                sorted_idxs = sorted(span_labels_dict.keys())
                for idx in sorted_idxs:
                    decoder_label_strings.append(span_labels_dict[idx])
            elif self.config.decoder_mode == 'prompt':
                decoder_label_strings.extend(list(classes_to_id))
                
        labels_batch = pad_2d_tensor(labels_batch) if len(labels_batch) > 1 else labels_batch[0].unsqueeze(0)

        decoder_tokenized_input = None

        if not decoder_label_strings:
            decoder_label_strings = ["other"]

        decoder_tokenized_input = self.decoder_tokenizer(
            decoder_label_strings, return_tensors="pt", truncation=True, padding="longest", add_special_tokens=True
        )
        decoder_input_ids = decoder_tokenized_input["input_ids"]
        decoder_attention_mask = decoder_tokenized_input["attention_mask"]
        decoder_labels = decoder_input_ids.clone()
        decoder_labels.masked_fill(~decoder_attention_mask.bool(), -100)
        decoder_tokenized_input["labels"] = decoder_labels
        return labels_batch, decoder_tokenized_input

    def tokenize_and_prepare_labels(self, batch, prepare_labels, *args, **kwargs):
        """Tokenize inputs and prepare labels for encoder-decoder training.

        Args:
            batch: Batch dictionary with tokens and class mappings.
            prepare_labels: Whether to prepare labels.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Dictionary containing encoder inputs, decoder inputs, and labels.
        """
        blank = None
        if random.uniform(0, 1) < self.config.blank_entity_prob and prepare_labels:
            blank = "entity"

        tokenized_input = self.tokenize_inputs(batch["tokens"], batch["classes_to_id"], blank)

        if prepare_labels:
            labels, decoder_tokenized_input = self.create_labels(batch, blank=blank)
            tokenized_input["labels"] = labels

            if decoder_tokenized_input is not None:
                tokenized_input["decoder_labels_ids"] = decoder_tokenized_input["input_ids"]
                tokenized_input["decoder_labels_mask"] = decoder_tokenized_input["attention_mask"]
                tokenized_input["decoder_labels"] = decoder_tokenized_input["labels"]

        return tokenized_input


class RelationExtractionSpanProcessor(UniEncoderSpanProcessor):
    """Processor for joint entity and relation extraction.

    Extends span-based NER processing to additionally handle relation extraction
    between entity pairs, supporting end-to-end joint training.
    """

    def __init__(self, config, tokenizer, words_splitter):
        """Initialize the relation extraction processor.

        Args:
            config: Configuration object.
            tokenizer: Transformer tokenizer.
            words_splitter: Word-level tokenizer/splitter.
        """
        super().__init__(config, tokenizer, words_splitter)
        self.rel_token = config.rel_token

    def batch_generate_class_mappings(
        self,
        batch_list: List[Dict],
        ner_negatives: Optional[List[str]] = None,
        rel_negatives: Optional[List[str]] = None,
        sampled_neg: int = 100,
    ) -> Tuple[List[Dict[str, int]], List[Dict[int, str]], List[Dict[str, int]], List[Dict[int, str]]]:
        """Generate class mappings for both entities and relations.

        Creates separate mappings for entity types and relation types with
        support for negative sampling for both.

        Args:
            batch_list: List of example dictionaries.
            ner_negatives: Optional pre-sampled negative entity types.
            rel_negatives: Optional pre-sampled negative relation types.
            sampled_neg: Number of negative types to sample if negatives not provided.

        Returns:
            Tuple containing:
                - List of entity class-to-ID mappings
                - List of entity ID-to-class mappings
                - List of relation class-to-ID mappings
                - List of relation ID-to-class mappings
        """
        if ner_negatives is None:
            ner_negatives = get_negatives(batch_list, sampled_neg=sampled_neg, key="ner")
        if rel_negatives is None:
            rel_negatives = get_negatives(batch_list, sampled_neg=sampled_neg, key="relations")

        class_to_ids = []
        id_to_classes = []
        rel_class_to_ids = []
        rel_id_to_classes = []

        for b in batch_list:
            max_neg_type_ratio = int(self.config.max_neg_type_ratio)
            neg_type_ratio = random.randint(0, max_neg_type_ratio) if max_neg_type_ratio else 0

            # Process NER types
            if "ner_negatives" in b:
                negs_i = b["ner_negatives"]
            else:
                negs_i = ner_negatives[: len(b["ner"]) * neg_type_ratio] if neg_type_ratio else []

            if "ner_labels" in b:
                types = b["ner_labels"]
            else:
                types = list(set([el[-1] for el in b["ner"]] + negs_i))
                random.shuffle(types)
                types = types[: int(self.config.max_types)]

            class_to_id = {k: v for v, k in enumerate(types, start=1)}
            id_to_class = {k: v for v, k in class_to_id.items()}
            class_to_ids.append(class_to_id)
            id_to_classes.append(id_to_class)

            # Process relation types
            if "rel_negatives" in b:
                rel_negs_i = b["rel_negatives"]
            else:
                rel_negs_i = rel_negatives[: len(b.get("relations", [])) * neg_type_ratio] if neg_type_ratio else []

            if "rel_labels" in b:
                rel_types = b["rel_labels"]
            else:
                rel_types = list(set([el[-1] for el in b.get("relations", [])] + rel_negs_i))
                random.shuffle(rel_types)
                rel_types = rel_types[: int(self.config.max_types)]

            rel_class_to_id = {k: v for v, k in enumerate(rel_types, start=1)}
            rel_id_to_class = {k: v for v, k in rel_class_to_id.items()}
            rel_class_to_ids.append(rel_class_to_id)
            rel_id_to_classes.append(rel_id_to_class)

        return class_to_ids, id_to_classes, rel_class_to_ids, rel_id_to_classes

    def collate_raw_batch(
        self,
        batch_list: List[Dict],
        entity_types: Optional[List[Union[str, List[str]]]] = None,
        relation_types: Optional[List[Union[str, List[str]]]] = None,
        ner_negatives: Optional[List[str]] = None,
        rel_negatives: Optional[List[str]] = None,
        class_to_ids: Optional[Union[Dict[str, int], List[Dict[str, int]]]] = None,
        id_to_classes: Optional[Union[Dict[int, str], List[Dict[int, str]]]] = None,
        rel_class_to_ids: Optional[Union[Dict[str, int], List[Dict[str, int]]]] = None,
        rel_id_to_classes: Optional[Union[Dict[int, str], List[Dict[int, str]]]] = None,
        key="ner",
    ) -> Dict:
        """Collate a raw batch with entity and relation label mappings.

        Args:
            batch_list: List of raw example dictionaries.
            entity_types: Optional predefined entity types.
            relation_types: Optional predefined relation types.
            ner_negatives: Optional negative entity types.
            rel_negatives: Optional negative relation types.
            class_to_ids: Optional entity class-to-ID mapping(s).
            id_to_classes: Optional entity ID-to-class mapping(s).
            rel_class_to_ids: Optional relation class-to-ID mapping(s).
            rel_id_to_classes: Optional relation ID-to-class mapping(s).
            key: Key for accessing labels in batch (default: 'ner').

        Returns:
            Dictionary containing collated batch data for joint entity and
            relation extraction.
        """
        if class_to_ids is None and entity_types is None:
            # Dynamically infer per-example mappings
            class_to_ids, id_to_classes, rel_class_to_ids, rel_id_to_classes = self.batch_generate_class_mappings(
                batch_list, ner_negatives, rel_negatives
            )
        elif class_to_ids is None:
            # Build mappings from entity_types
            if entity_types and isinstance(entity_types[0], list):
                built = [make_mapping(t) for t in entity_types]
                class_to_ids, id_to_classes = list(zip(*built))
                class_to_ids, id_to_classes = list(class_to_ids), list(id_to_classes)
            else:
                class_to_ids, id_to_classes = make_mapping(entity_types or [])

            # Build relation mappings
            if relation_types and isinstance(relation_types[0], list):
                built = [make_mapping(t) for t in relation_types]
                rel_class_to_ids, rel_id_to_classes = list(zip(*built))
                rel_class_to_ids, rel_id_to_classes = list(rel_class_to_ids), list(rel_id_to_classes)
            else:
                rel_class_to_ids, rel_id_to_classes = make_mapping(relation_types or [])

        if isinstance(class_to_ids, list):
            batch = [
                self.preprocess_example(
                    b["tokenized_text"],
                    b[key],
                    class_to_ids[i],
                    b.get("relations", []),
                    rel_class_to_ids[i] if isinstance(rel_class_to_ids, list) else rel_class_to_ids,
                )
                for i, b in enumerate(batch_list)
            ]
        else:
            batch = [
                self.preprocess_example(
                    b["tokenized_text"], b[key], class_to_ids, b.get("relations", []), rel_class_to_ids
                )
                for b in batch_list
            ]

        return self.create_batch_dict(batch, class_to_ids, id_to_classes, rel_class_to_ids, rel_id_to_classes)

    def preprocess_example(self, tokens, ner, classes_to_id, relations, rel_classes_to_id):
        """Preprocess a single example for joint entity and relation extraction.

        Processes both entity spans and relation triplets, ensuring consistent
        indexing when entities are reordered.

        Args:
            tokens: List of token strings.
            ner: List of entity annotations as (start, end, label) tuples.
            classes_to_id: Mapping from entity class labels to integer IDs.
            relations: List of relation annotations as (head_idx, tail_idx, rel_type) tuples.
            rel_classes_to_id: Mapping from relation class labels to integer IDs.

        Returns:
            Dictionary containing:
                - tokens: Token strings
                - span_idx: Tensor of span indices
                - span_label: Tensor of entity labels for each span
                - seq_length: Sequence length
                - entities: Original entity annotations
                - relations: Original relation annotations
                - rel_idx: Tensor of relation head/tail indices
                - rel_label: Tensor of relation type labels

        Warnings:
            UserWarning: If sequence length exceeds max_len (gets truncated).
        """
        max_width = self.config.max_width

        if len(tokens) == 0:
            tokens = ["[PAD]"]
        max_len = self.config.max_len
        if len(tokens) > max_len:
            warnings.warn(f"Sentence of length {len(tokens)} has been truncated to {max_len}", stacklevel=2)
            tokens = tokens[:max_len]

        num_tokens = len(tokens)
        spans_idx = prepare_span_idx(num_tokens, max_width)

        if ner is not None and len(ner) > 0:
            indexed_ner = list(enumerate(ner))
            indexed_ner_sorted = sorted(indexed_ner, key=lambda x: (x[1][0], x[1][1]))

            ner_sorted = [entity for _, entity in indexed_ner_sorted]

            old_to_new_idx = {old_idx: new_idx for new_idx, (old_idx, _) in enumerate(indexed_ner_sorted)}

            if relations is not None and len(relations) > 0:
                updated_relations = []
                for head_idx, tail_idx, rel_type in relations:
                    if head_idx in old_to_new_idx and tail_idx in old_to_new_idx:
                        new_head_idx = old_to_new_idx[head_idx]
                        new_tail_idx = old_to_new_idx[tail_idx]
                        updated_relations.append((new_head_idx, new_tail_idx, rel_type))
                relations = sorted(updated_relations, key=lambda x: (x[0], x[1]))

            ner = ner_sorted

        # Process entity labels
        dict_lab = self.get_dict(ner, classes_to_id) if ner else defaultdict(int)
        span_label = torch.LongTensor([dict_lab[i] for i in spans_idx])
        spans_idx = torch.LongTensor(spans_idx)
        valid_span_mask = spans_idx[:, 1] > num_tokens - 1
        span_label = span_label.masked_fill(valid_span_mask, -1)

        # Create entity span to index mapping
        span_to_idx = {(spans_idx[i, 0].item(), spans_idx[i, 1].item()): i for i in range(len(spans_idx))}

        # Create entity index mapping (from original entity list to span indices)
        entity_to_span_idx = {}
        if ner is not None:
            for ent_idx, (start, end, _) in enumerate(ner):  # (start, end, label)
                if (start, end) in span_to_idx and end < num_tokens:
                    entity_to_span_idx[ent_idx] = span_to_idx[(start, end)]

        # Process relations
        rel_idx_list = []
        rel_label_list = []

        if relations is not None:
            for rel in relations:
                head_idx, tail_idx, rel_type = rel

                # Check if both entities are valid and map to span indices
                if head_idx in entity_to_span_idx and tail_idx in entity_to_span_idx and rel_type in rel_classes_to_id:
                    rel_idx_list.append([head_idx, tail_idx])
                    rel_label_list.append(rel_classes_to_id[rel_type])

        # Convert to tensors
        if rel_idx_list:
            rel_idx = torch.LongTensor(rel_idx_list)
            rel_label = torch.LongTensor(rel_label_list)
        else:
            rel_idx = torch.zeros(0, 2, dtype=torch.long)
            rel_label = torch.zeros(0, dtype=torch.long)

        return {
            "tokens": tokens,
            "span_idx": spans_idx,
            "span_label": span_label,
            "seq_length": num_tokens,
            "entities": ner,
            "relations": relations,
            "rel_idx": rel_idx,
            "rel_label": rel_label,
        }

    def create_batch_dict(self, batch, class_to_ids, id_to_classes, rel_class_to_ids, rel_id_to_classes):
        """Create a batch dictionary from preprocessed relation extraction examples.

        Args:
            batch: List of preprocessed example dictionaries.
            class_to_ids: List of entity class-to-ID mappings.
            id_to_classes: List of entity ID-to-class mappings.
            rel_class_to_ids: List of relation class-to-ID mappings.
            rel_id_to_classes: List of relation ID-to-class mappings.

        Returns:
            Dictionary containing all batch data for joint entity and relation
            extraction, including entity spans, relation pairs, and their labels.
        """
        tokens = [el["tokens"] for el in batch]
        entities = [el["entities"] for el in batch]
        relations = [el["relations"] for el in batch]

        span_idx = pad_sequence([b["span_idx"] for b in batch], batch_first=True, padding_value=0)
        span_label = pad_sequence([el["span_label"] for el in batch], batch_first=True, padding_value=-1)
        rel_idx = pad_sequence([el["rel_idx"] for el in batch], batch_first=True, padding_value=0)
        rel_label = pad_sequence([el["rel_label"] for el in batch], batch_first=True, padding_value=0)

        seq_length = torch.LongTensor([el["seq_length"] for el in batch]).unsqueeze(-1)
        span_mask = span_label != -1

        return {
            "seq_length": seq_length,
            "span_idx": span_idx,
            "tokens": tokens,
            "span_mask": span_mask,
            "span_label": span_label,
            "entities": entities,
            "relations": relations,
            "rel_idx": rel_idx,
            "rel_label": rel_label,
            "classes_to_id": class_to_ids,
            "id_to_classes": id_to_classes,
            "rel_class_to_ids": rel_class_to_ids,
            "rel_id_to_classes": rel_id_to_classes,
        }

    def create_relation_labels(self, batch, add_reversed_negatives=True, add_random_negatives=True, negative_ratio=2.0):
        """Create relation labels with negative pair sampling.

        Generates training labels for relation extraction including both positive
        relation pairs and carefully sampled negative pairs for contrastive learning.

        Args:
            batch: Batch dictionary containing entities and relations.
            add_reversed_negatives: If True, add reversed direction pairs as
                negatives (h,t) -> (t,h). These are important hard negatives
                for learning relation directionality.
            add_random_negatives: If True, add random entity pairs as negatives
                to provide additional training signal.
            negative_ratio: Ratio of negative to positive pairs. For example,
                2.0 means twice as many negatives as positives.

        Returns:
            Tuple containing:
                - adj_matrix: Adjacency matrix indicating which entity pairs
                  to consider (shape: [B, max_entities, max_entities])
                - rel_matrix: Multi-hot encoded relation labels for each pair
                  (shape: [B, max_pairs, num_relation_classes])
        """
        B = len(batch["tokens"])
        entity_label = batch["span_label"]

        batch_ents = torch.sum(entity_label > 0, dim=-1)
        max_En = torch.max(batch_ents).item()

        rel_class_to_ids = batch["rel_class_to_ids"]
        if isinstance(rel_class_to_ids, list):
            C = max(len(r) for r in rel_class_to_ids)
        else:
            C = len(rel_class_to_ids)

        adj_matrix = torch.zeros(B, max_En, max_En, dtype=torch.float)

        # Collect all pairs (positive + negative) and their relations
        all_pairs_info = []
        max_total_pairs = 0

        for i in range(B):
            N = batch_ents[i].item()
            rel_idx_i = batch["rel_idx"][i]
            rel_label_i = batch["rel_label"][i]

            # Dictionary to group relations by entity pair
            pair_to_relations = {}
            positive_pairs = set()

            # Collect positive pairs
            for k in range(rel_label_i.shape[0]):
                if rel_label_i[k] > 0:
                    e1 = rel_idx_i[k, 0].item()
                    e2 = rel_idx_i[k, 1].item()

                    if e1 < N and e2 < N:
                        pair_key = (e1, e2)
                        positive_pairs.add(pair_key)
                        if pair_key not in pair_to_relations:
                            pair_to_relations[pair_key] = []
                        class_id = rel_label_i[k].item()
                        pair_to_relations[pair_key].append(class_id)

            # Generate negative pairs
            negative_pairs = set()
            num_positives = len(positive_pairs)
            target_negatives = int(num_positives * negative_ratio)

            # 1. Add reversed pairs as negatives (most important!)
            if add_reversed_negatives:
                for e1, e2 in positive_pairs:
                    reversed_pair = (e2, e1)
                    # Only add if reversed pair is NOT also a positive relation
                    if reversed_pair not in positive_pairs:
                        negative_pairs.add(reversed_pair)

            # 2. Add random negative pairs if needed
            if add_random_negatives and len(negative_pairs) < target_negatives:
                # Get entity span positions for proximity-based sampling
                entities = batch["entities"][i]
                entity_positions = [(ent[0], ent[1]) for ent in entities] if entities else []

                attempts = 0
                max_attempts = target_negatives * 10  # Avoid infinite loop

                while len(negative_pairs) < target_negatives and attempts < max_attempts:
                    attempts += 1

                    # Sample two different entities
                    e1 = random.randint(0, N - 1)
                    e2 = random.randint(0, N - 1)

                    if e1 == e2:
                        continue

                    pair = (e1, e2)

                    # Skip if already positive or already in negatives
                    if pair in positive_pairs or pair in negative_pairs:
                        continue

                    # Optional: bias towards nearby entities (hard negatives)
                    if entity_positions and len(entity_positions) > e1 and len(entity_positions) > e2:
                        pos1 = entity_positions[e1]
                        pos2 = entity_positions[e2]
                        distance = abs(pos1[0] - pos2[1])  # Distance between entities

                        # Sample with probability inversely proportional to distance
                        # (closer entities are harder negatives)
                        if distance > 10 and random.random() < 0.5:
                            continue  # Skip some far pairs

                    negative_pairs.add(pair)

            # Combine all pairs (positives + negatives) and sort
            all_pairs = sorted(list(positive_pairs) + list(negative_pairs))

            # Store pair info: pair, is_positive, relations
            pair_info = []
            for pair in all_pairs:
                is_positive = pair in positive_pairs
                relations = pair_to_relations.get(pair, [])
                pair_info.append((pair, is_positive, relations))

            all_pairs_info.append(pair_info)
            max_total_pairs = max(max_total_pairs, len(all_pairs))

        # Create matrices
        rel_matrix = torch.zeros(B, max_total_pairs, C, dtype=torch.float)
        pair_type_mask = torch.zeros(B, max_total_pairs, dtype=torch.long)  # 1=positive, 0=negative

        for i in range(B):
            N = batch_ents[i].item()
            pair_info = all_pairs_info[i]

            adj = torch.zeros(N, N)

            for pair_idx, (pair, is_positive, relations) in enumerate(pair_info):
                e1, e2 = pair

                # Set adjacency (1.0 for both positive and negative pairs)
                adj[e1, e2] = 1.0

                # Mark pair type
                pair_type_mask[i, pair_idx] = 1 if is_positive else 0

                if is_positive:
                    # Create multi-hot vector for positive pairs
                    for class_id in relations:
                        rel_matrix[i, pair_idx, class_id - 1] = 1.0

            adj_matrix[i, :N, :N] = adj

        return adj_matrix, rel_matrix

    def prepare_inputs(
        self,
        texts: Sequence[Sequence[str]],
        entities: Union[Sequence[Sequence[str]], Dict[int, Sequence[str]], Sequence[str]],
        blank: Optional[str] = None,
        relations: Optional[Union[Sequence[Sequence[str]], Dict[int, Sequence[str]], Sequence[str]]] = None,
        **kwargs,
    ) -> Tuple[List[List[str]], List[int]]:
        """Prepare input texts with entity and relation type prompts.

        Extends the base prepare_inputs to include relation type tokens in the prompt.

        Args:
            texts: Sequences of token strings, one per example.
            entities: Entity types to extract.
            blank: Optional blank entity token for zero-shot scenarios.
            relations: Relation types to extract (optional).
            **kwargs: Additional keyword arguments.

        Returns:
            Tuple containing:
                - List of input text sequences with prepended prompts
                - List of prompt lengths for each example
        """
        input_texts: List[List[str]] = []
        prompt_lengths: List[int] = []

        for i, text in enumerate(texts):
            ents = self._select_entities(i, entities, blank)
            ents = self._maybe_remap_entities(ents)

            rels = self._select_entities(i, relations, blank) if relations else []
            rels = self._maybe_remap_entities(rels)

            prompt: List[str] = []
            for ent in ents:
                prompt += [self.ent_token, str(ent)]
            prompt.append(self.sep_token)

            for rel in rels:
                prompt += [self.rel_token, str(rel)]
            prompt.append(self.sep_token)

            prompt_lengths.append(len(prompt))
            input_texts.append(prompt + list(text))

        return input_texts, prompt_lengths

    def tokenize_inputs(self, texts, entities, blank=None, relations=None, **kwargs):
        """Tokenize input texts with entity and relation prompts.

        Args:
            texts: Sequences of token strings.
            entities: Entity types for extraction.
            blank: Optional blank entity token.
            relations: Optional relation types for extraction.
            **kwargs: Additional keyword arguments.

        Returns:
            Dictionary containing tokenized inputs with word masks.
        """
        input_texts, prompt_lengths = self.prepare_inputs(texts, entities, blank=blank, relations=relations, **kwargs)

        tokenized_inputs = self.transformer_tokenizer(
            input_texts,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            padding="longest",
        )
        words_masks = self.prepare_word_mask(texts, tokenized_inputs, prompt_lengths)
        tokenized_inputs["words_mask"] = torch.tensor(words_masks)

        return tokenized_inputs

    def tokenize_and_prepare_labels(self, batch, prepare_labels, *args, **kwargs):
        """Tokenize inputs and prepare labels for joint entity-relation extraction.

        Args:
            batch: Batch dictionary with tokens, entities, relations, and class mappings.
            prepare_labels: Whether to prepare labels.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Dictionary containing tokenized inputs, entity labels, relation adjacency
            matrix, and relation labels.
        """
        tokenized_input = self.tokenize_inputs(
            batch["tokens"], batch["classes_to_id"], blank=None, relations=batch["rel_class_to_ids"]
        )

        if prepare_labels:
            labels = self.create_labels(batch)
            tokenized_input["labels"] = labels

            adj_matrix, rel_matrix = self.create_relation_labels(batch)
            tokenized_input["adj_matrix"] = adj_matrix
            tokenized_input["rel_matrix"] = rel_matrix

        return tokenized_input
