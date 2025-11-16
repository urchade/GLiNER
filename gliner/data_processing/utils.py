import random
from typing import Dict, List, Tuple, Optional, Sequence

import torch


def pad_2d_tensor(key_data):
    """
    Pad a list of 2D tensors to have the same size along both dimensions.

    :param key_data: List of 2D tensors to pad.
    :return: Tensor of padded tensors stacked along a new batch dimension.
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
        padded_tensor = torch.nn.functional.pad(tensor, (0, col_padding, 0, row_padding), mode="constant", value=0)
        tensors.append(padded_tensor)

    # Stack the tensors into a single tensor along a new batch dimension
    padded_tensors = torch.stack(tensors)

    return padded_tensors


def get_negatives(batch_list: List[Dict], sampled_neg: int = 5, key="ner") -> List[str]:
    element_types = set()
    for b in batch_list:
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
) -> List[List[int]]:
    n = len(texts)
    if skip_first_words is None:
        skip_first_words = [0] * n
    elif len(skip_first_words) != n:
        raise ValueError("skip_first_words must have same length as texts")

    words_masks: List[List[int]] = []
    for i in range(n):
        mask: List[int] = []
        prev_word_id: Optional[int] = None
        seen_words = 0  # counts distinct word_ids we've traversed in this sequence

        for wid in tokenized_inputs.word_ids(i):
            if wid is None:
                mask.append(0)
            elif wid != prev_word_id or token_level:
                # If we just moved to a new word, update seen_words
                if wid != prev_word_id:
                    seen_words += 1
                if seen_words <= skip_first_words[i]:
                    mask.append(0)
                else:
                    # 1-based word index after skipping
                    mask.append(seen_words - skip_first_words[i])
            else:
                # same word continuation and token_level=False -> ignore
                mask.append(0)
            prev_word_id = wid
        words_masks.append(mask)
    return words_masks


def make_mapping(types: List[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    # de-duplicate while preserving order
    uniq = list(dict.fromkeys(types))
    fwd = {k: i for i, k in enumerate(uniq, start=1)}
    rev = {v: k for k, v in fwd.items()}
    return fwd, rev


def prepare_span_idx(num_tokens, max_width):
    span_idx = [(i, i + j) for i in range(num_tokens) for j in range(max_width)]
    return span_idx
