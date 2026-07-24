import random
from typing import Dict, List, Tuple, Optional, Sequence

import torch


def pad_2d_tensor(key_data, padding_value=0.0):
    """Pad a list of 2D tensors to uniform dimensions.

    Takes a list of 2D tensors with potentially different shapes and pads them
    to match the maximum dimensions across all tensors. All tensors are padded
    with zeros to create a uniform rectangular shape, then stacked into a single
    3D tensor with a batch dimension.

    Args:
        key_data: List of 2D tensors to pad. Each tensor can have different
            dimensions, but all must be 2D.
        padding_value: float, value used to fill pad elements.

    Returns:
        A 3D tensor of shape (batch_size, max_rows, max_cols) containing all
        input tensors padded and stacked along the batch dimension.

    Raises:
        ValueError: If the input list is empty.

    Example:
        >>> tensor1 = torch.tensor([[1, 2], [3, 4]])  # 2x2
        >>> tensor2 = torch.tensor([[5, 6, 7]])  # 1x3
        >>> result = pad_2d_tensor([tensor1, tensor2])
        >>> result.shape
        torch.Size([2, 2, 3])
    """
    if not key_data:
        raise ValueError("The input list 'key_data' should not be empty.")

    # Determine the maximum size along both dimensions
    max_rows = max(tensor.shape[0] for tensor in key_data)
    max_cols = max(tensor.shape[1] for tensor in key_data)

    tensors = []
    for tensor in key_data:
        rows, cols = tensor.shape
        row_padding = max_rows - rows
        col_padding = max_cols - cols

        # Pad the tensor along both dimensions
        padded_tensor = torch.nn.functional.pad(
            tensor, (0, col_padding, 0, row_padding), mode="constant", value=padding_value
        )
        tensors.append(padded_tensor)

    # Stack the tensors into a single tensor along a new batch dimension
    padded_tensors = torch.stack(tensors)
    return padded_tensors


def get_negatives(batch_list: List[Dict], sampled_neg: int = 5, key="ner") -> List[str]:
    """Sample negative entity or relation types from a batch.

    Extracts all unique entity/relation types from a batch of examples and
    randomly samples a subset to use as negative types for contrastive learning.
    This helps the model learn to distinguish between similar but incorrect types.

    Args:
        batch_list: List of example dictionaries. Each dictionary should contain
            the specified key with annotations in the format where the last element
            of each annotation tuple is the type label.
        sampled_neg: Maximum number of negative types to sample (default: 5).
            If fewer unique types exist, all will be returned.
        key: Dictionary key to access annotations (default: "ner"). Common values
            are "ner" for entities or "relations" for relation types.

    Returns:
        List of randomly sampled type strings. Length will be min(sampled_neg,
        number of unique types in batch).

    Example:
        >>> batch = [{"ner": [(0, 1, "PERSON"), (2, 3, "ORG")]}, {"ner": [(0, 1, "LOC"), (3, 4, "PERSON")]}]
        >>> negatives = get_negatives(batch, sampled_neg=2, key="ner")
        >>> len(negatives) <= 2
        True
    """
    element_types = set()
    for b in batch_list:
        if b.get(key, False):
            types = {el[-1] for el in b[key]}
            element_types.update(types)

    element_types = list(element_types)
    selected_elements = random.sample(element_types, k=min(sampled_neg, len(element_types)))
    return selected_elements


def prepare_word_mask(
    texts: Sequence[Sequence[str]],
    tokenized_inputs,
    *,
    skip_first_words: Optional[Sequence[int]] = None,
    token_level: bool = False,
    subtoken_pooling: str = "first",
) -> List[List[int]]:
    """Create word-level masks for subword tokenized sequences.

    Maps subword tokens back to their original word positions, enabling span
    extraction at the word level. Each subword token is assigned an integer
    indicating which word it belongs to (1-indexed), with special tokens and
    continuation subwords optionally masked as 0.

    This is essential for span-based NER where predictions are made at the word
    level but the model processes subword tokens. The mask allows the model to
    aggregate subword representations into word-level representations.

    Args:
        texts: Original text sequences as lists of words, one sequence per example.
        tokenized_inputs: Tokenized output from a transformer tokenizer with a
            word_ids() method (e.g., from HuggingFace tokenizers).
        skip_first_words: Optional number of words to skip at the beginning of
            each sequence (e.g., prompt words). Must have the same length as texts
            if provided. Skipped words are masked as 0.
        token_level: If True, assign the word index to every subtoken, overriding
            ``subtoken_pooling``. Defaults to False.
        subtoken_pooling: Determines which subtokens receive a word index when
            ``token_level`` is False: the first subtoken for ``first``, the last
            for ``last``, or every subtoken for ``mean`` and ``max``.

    Returns:
        List of word mask lists, one per input sequence. Each mask list has the
        same length as the corresponding tokenized sequence. Values are:
            - 0: Special tokens, skipped words, or unselected subtokens
            - 1, 2, 3, ...: Word indices (1-indexed) after skipping

    Raises:
        ValueError: If skip_first_words length doesn't match texts length.

    Example:
        >>> texts = [["Hello", "world"]]
        >>> # Assuming tokenizer splits "Hello" -> ["Hel", "##lo"]
        >>> # and "world" -> ["world"]
        >>> mask = prepare_word_mask(texts, tokenized_inputs)
        >>> # Result might be: [[0, 1, 0, 2, 0]]
        >>> #                   [CLS, Hel, ##lo, world, SEP]
    """
    supported_pooling = {"first", "last", "mean", "max"}
    if subtoken_pooling not in supported_pooling:
        supported = ", ".join(sorted(supported_pooling))
        raise ValueError(f"Unknown subtoken pooling strategy {subtoken_pooling!r}. Expected one of: {supported}")

    n = len(texts)

    if skip_first_words is None:
        skip_first_words = [0] * n
    elif len(skip_first_words) != n:
        raise ValueError("skip_first_words must have same length as texts")

    words_masks: List[List[int]] = []

    for i in range(n):
        word_ids = tokenized_inputs.word_ids(i)
        mask: List[int] = []
        prev_word_id: Optional[int] = None
        seen_words = 0  # counts distinct word_ids we've traversed in this sequence

        for token_idx, wid in enumerate(word_ids):
            if wid is None:
                # Special tokens (CLS, SEP, PAD, etc.)
                mask.append(0)
            else:
                is_first_subtoken = wid != prev_word_id
                if is_first_subtoken:
                    seen_words += 1

                next_word_id = word_ids[token_idx + 1] if token_idx + 1 < len(word_ids) else None
                is_last_subtoken = next_word_id != wid
                select_subtoken = (
                    token_level
                    or subtoken_pooling in {"mean", "max"}
                    or (subtoken_pooling == "first" and is_first_subtoken)
                    or (subtoken_pooling == "last" and is_last_subtoken)
                )

                if seen_words <= skip_first_words[i] or not select_subtoken:
                    mask.append(0)
                else:
                    mask.append(seen_words - skip_first_words[i])

            prev_word_id = wid

        words_masks.append(mask)

    return words_masks


def make_mapping(types: List[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Create bidirectional mappings between type labels and integer IDs.

    Generates forward and reverse dictionaries for converting between string
    labels (e.g., entity or relation types) and integer IDs used in model training.
    Duplicate types are removed while preserving the order of first occurrence.
    IDs start from 1 (reserving 0 for no-label/padding).

    Args:
        types: List of type label strings. May contain duplicates, which will
            be removed while preserving order.

    Returns:
        Tuple containing:
            - Forward mapping (Dict[str, int]): Maps type labels to integer IDs
              starting from 1
            - Reverse mapping (Dict[int, str]): Maps integer IDs back to type labels

    Example:
        >>> types = ["PERSON", "ORG", "LOC", "PERSON"]  # "PERSON" duplicated
        >>> fwd, rev = make_mapping(types)
        >>> fwd
        {'PERSON': 1, 'ORG': 2, 'LOC': 3}
        >>> rev
        {1: 'PERSON', 2: 'ORG', 3: 'LOC'}
    """
    # de-duplicate while preserving order
    uniq = list(dict.fromkeys(types))
    fwd = {k: i for i, k in enumerate(uniq, start=1)}
    rev = {v: k for k, v in fwd.items()}
    return fwd, rev


def prepare_span_idx(num_tokens, max_width):
    """Generate all possible span indices for a sequence.

    Creates a list of all possible (start, end) span pairs for a sequence,
    where each span has a width (end - start) less than max_width. This is used
    in span-based NER models that enumerate and classify all possible spans.

    The spans follow these conventions:
    - Start index is inclusive
    - End index is inclusive (so span (i, i) is a single token)
    - Spans are generated in left-to-right order, with shorter spans first
      for each starting position

    Args:
        num_tokens: Length of the sequence (number of tokens).
        max_width: Maximum span width to generate. A span of width w covers
            w+1 tokens (e.g., width 0 is a single token).

    Returns:
        List of (start, end) tuples representing all valid spans. Each tuple
        contains:
            - start: Starting token index (0-indexed, inclusive)
            - end: Ending token index (0-indexed, inclusive)

    Example:
        >>> spans = prepare_span_idx(num_tokens=3, max_width=2)
        >>> spans  # tensor([[0,0],[0,1],[1,1],[1,2],[2,2],[2,3]])
        >>> # For sequence ["The", "cat", "sat"]:
        >>> # (0, 0) = "The"
        >>> # (0, 1) = "The cat"
        >>> # (1, 1) = "cat"
        >>> # (1, 2) = "cat sat"
        >>> # (2, 2) = "sat"
        >>> # (2, 3) would be invalid (beyond sequence length)

    Returns:
        torch.LongTensor of shape (num_tokens * max_width, 2) with columns [start, end].
    """
    starts = torch.arange(num_tokens, dtype=torch.long).unsqueeze(1).expand(-1, max_width).reshape(-1)
    offsets = torch.arange(max_width, dtype=torch.long).unsqueeze(0).expand(num_tokens, -1).reshape(-1)
    return torch.stack([starts, starts + offsets], dim=1)


def prepare_streaming_span_idx(
    past_tokens: int,
    new_tokens: int,
    max_width: int,
    recompute_all: bool = False,
    right_context_width: int = 0,
):
    """Generate absolute span candidates for the next streaming chunk.

    In incremental mode, all spans ending in the new chunk are valid.  Spans
    ending up to ``right_context_width`` words before the latest word are also
    regenerated so future context can revise their scores.  A zero-width right
    context preserves append-only scoring.  Full-recompute mode enumerates
    every historical span again.

    Returns:
        A tuple ``(span_idx, span_mask)`` where ``span_idx`` has shape
        ``(num_starts * max_width, 2)`` and uses absolute session word indices.
    """
    if past_tokens < 0 or new_tokens < 0:
        raise ValueError("past_tokens and new_tokens must be non-negative")
    if max_width < 1:
        raise ValueError("max_width must be positive")
    if right_context_width < 0:
        raise ValueError("right_context_width must be non-negative")

    total_tokens = past_tokens + new_tokens
    if total_tokens == 0:
        return torch.zeros((0, 2), dtype=torch.long), torch.zeros(0, dtype=torch.bool)

    if recompute_all:
        minimum_end = 0
    else:
        latest_word = total_tokens - 1
        rolling_minimum_end = max(0, latest_word - right_context_width)
        # Always score every span ending in the newly supplied chunk, even when
        # a caller appends more words than the configured right-context window.
        minimum_end = min(past_tokens, rolling_minimum_end)

    first_start = max(0, minimum_end - (max_width - 1))
    num_starts = total_tokens - first_start
    span_idx = prepare_span_idx(num_starts, max_width) + first_start

    span_mask = span_idx[:, 1] < total_tokens
    if not recompute_all:
        span_mask = span_mask & span_idx[:, 1].ge(minimum_end)
        if new_tokens == 0:
            span_mask.zero_()

    return span_idx, span_mask
