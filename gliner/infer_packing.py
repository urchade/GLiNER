"""Utilities for inference-time sequence packing.

This module provides helpers to group many short sequences into a
single (or a few) contiguous token streams in order to reduce the
amount of padding the encoder needs to process.  Packed batches keep a
block-diagonal attention mask so tokens from different original
sequences cannot attend to each other.  After the encoder forward
pass, results can be unpacked back to the original request ordering.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class InferencePackingConfig:
    """Configuration describing how sequences should be packed.

    Attributes:
        max_length: Maximum number of tokens allowed in a packed stream.
        sep_token_id: Optional separator token ID to insert between sequences.
            Currently not used in the implementation.
        streams_per_batch: Number of streams to create per batch. Must be >= 1.
    """

    max_length: int
    sep_token_id: Optional[int] = None
    streams_per_batch: int = 1


@dataclass
class PackedBatch:
    """Container describing a packed collection of requests.

    Attributes:
        input_ids: Tensor of shape (num_streams, max_len) containing packed token IDs.
        attention_mask: Tensor of shape (num_streams, max_len) with 1s for valid tokens
            and 0s for padding.
        pair_attention_mask: Boolean tensor of shape (num_streams, max_len, max_len)
            representing block-diagonal attention mask.
        segment_ids: Tensor of shape (num_streams, max_len) with unique IDs for each
            packed segment within a stream.
        map_out: List of lists mapping each segment in each stream back to its
            original request index.
        offsets: List of lists containing the starting offset of each segment
            within each stream.
        lengths: List of lists containing the length of each segment within each stream.
    """

    input_ids: torch.LongTensor
    attention_mask: torch.LongTensor
    pair_attention_mask: torch.BoolTensor
    segment_ids: torch.LongTensor
    map_out: List[List[int]]
    offsets: List[List[int]]
    lengths: List[List[int]]


Request = Dict[str, Any]


def _ensure_list(tokens: Sequence[int]) -> List[int]:
    """Convert a sequence of tokens to a Python list.

    Args:
        tokens: A sequence of integer tokens (list, tuple, tensor, etc.).

    Returns:
        A Python list containing the same tokens.
    """
    if isinstance(tokens, list):
        return tokens
    return [int(t) for t in tokens]


def block_diag_mask(segment_ids: torch.LongTensor) -> torch.BoolTensor:
    """Construct a block diagonal mask from per-token segment ids.

    Creates a boolean attention mask where tokens can only attend to other tokens
    with the same segment ID. This prevents cross-contamination between different
    sequences packed into the same stream.

    Args:
        segment_ids: Tensor of shape (batch_size, seq_len) containing segment IDs
            for each token position.

    Returns:
        Boolean tensor of shape (batch_size, seq_len, seq_len) where mask[b, i, j]
        is True if tokens i and j belong to the same segment in batch b.
    """
    return segment_ids.unsqueeze(2).eq(segment_ids.unsqueeze(1))


def _pad_2d(x: torch.Tensor, target: int, pad_val: int) -> torch.Tensor:
    """Pad the second dimension of a 2D tensor to a target length.

    Args:
        x: Input tensor of shape (batch_size, seq_len).
        target: Target length for the second dimension.
        pad_val: Value to use for padding.

    Returns:
        Tensor of shape (batch_size, target) with padding applied if necessary.
        If seq_len >= target, returns the original tensor unchanged.
    """
    if x.size(1) >= target:
        return x
    out = x.new_full((x.size(0), target), pad_val)
    out[:, : x.size(1)] = x
    return out


class _PackedStream:
    """Internal helper class representing a single packed token stream.

    A stream accumulates multiple sequences into a contiguous sequence of tokens,
    tracking the boundaries and origins of each packed segment.

    Attributes:
        tokens: List of all tokens in this stream.
        map_out: List mapping each segment to its original request index.
        offsets: List of starting positions for each segment.
        lengths: List of lengths for each segment.
    """

    __slots__ = ("lengths", "map_out", "offsets", "tokens")

    def __init__(self) -> None:
        """Initialize an empty packed stream."""
        self.tokens: List[int] = []
        self.map_out: List[int] = []
        self.offsets: List[int] = []
        self.lengths: List[int] = []

    @property
    def total_tokens(self) -> int:
        """Get the total number of tokens currently in this stream.

        Returns:
            The length of the tokens list.
        """
        return len(self.tokens)

    def append(self, req_idx: int, tokens: Sequence[int]) -> None:
        """Append a new sequence to this stream.

        Args:
            req_idx: Index of the original request being added.
            tokens: Sequence of token IDs to append to the stream.
        """
        offset = self.total_tokens
        segment_tokens = _ensure_list(tokens)
        self.tokens.extend(segment_tokens)
        self.map_out.append(req_idx)
        self.offsets.append(offset)
        self.lengths.append(len(segment_tokens))


def _prepare_streams(requests: List[Request], cfg: InferencePackingConfig) -> List[_PackedStream]:
    """Prepare packed streams from a list of requests using a first-fit strategy.

    Iterates through requests and packs each into the first stream with enough
    remaining capacity. Creates new streams as needed.

    Args:
        requests: List of request dictionaries, each containing an 'input_ids' key.
        cfg: Packing configuration specifying max_length and other parameters.

    Returns:
        List of _PackedStream objects containing the packed sequences.

    Raises:
        ValueError: If streams_per_batch < 1 or max_length <= 0.
        KeyError: If any request is missing the 'input_ids' key.
    """
    if cfg.streams_per_batch < 1:
        raise ValueError("streams_per_batch must be >= 1")

    streams: List[_PackedStream] = []

    for req_idx, request in enumerate(requests):
        tokens = request.get("input_ids")
        if tokens is None:
            raise KeyError("Each request must provide an 'input_ids' entry")
        token_list = _ensure_list(tokens)
        if cfg.max_length <= 0:
            raise ValueError("max_length must be positive")
        if len(token_list) > cfg.max_length:
            token_list = token_list[: cfg.max_length]

        placed = False
        for stream in streams:
            if stream.total_tokens + len(token_list) <= cfg.max_length:
                stream.append(req_idx, token_list)
                placed = True
                break

        if not placed:
            stream = _PackedStream()
            stream.append(req_idx, token_list)
            streams.append(stream)

    return streams


def _build_segment_ids(streams: List[_PackedStream], max_len: int) -> torch.LongTensor:
    """Build segment ID tensors for each stream.

    Assigns a unique segment ID (starting from 1) to each packed sequence within
    each stream. Padding positions receive segment ID 0.

    Args:
        streams: List of packed streams to build segment IDs for.
        max_len: Maximum length to pad each stream to.

    Returns:
        Tensor of shape (num_streams, max_len) containing segment IDs for each
        token position. Returns a (0, max_len) tensor if streams is empty.
    """
    segment_rows: List[torch.Tensor] = []
    for stream in streams:
        seg = torch.zeros(max_len, dtype=torch.long)
        seg_id = 1
        for offset, length in zip(stream.offsets, stream.lengths):
            if length == 0:
                continue
            seg[offset : offset + length] = seg_id
            seg_id += 1
        segment_rows.append(seg)
    return torch.stack(segment_rows, dim=0) if segment_rows else torch.zeros((0, max_len), dtype=torch.long)


def pack_requests(requests: List[Request], cfg: InferencePackingConfig, pad_token_id: int) -> PackedBatch:
    """Pack a collection of requests into one or more streams.

    Groups multiple short sequences into contiguous token streams to reduce padding
    overhead. Each request's tokens are placed into streams using a first-fit
    strategy. A block-diagonal attention mask ensures tokens from different
    requests cannot attend to each other.

    Args:
        requests: List of request dictionaries. Each must contain an 'input_ids'
            key with a sequence of token IDs.
        cfg: Configuration specifying packing parameters (max_length, etc.).
        pad_token_id: Token ID to use for padding positions.

    Returns:
        PackedBatch object containing packed tensors and metadata needed to
        unpack results back to original request ordering.

    Raises:
        ValueError: If requests list is empty or configuration is invalid.
        KeyError: If any request is missing required 'input_ids' key.

    Example:
        >>> requests = [
        ...     {"input_ids": [1, 2, 3]},
        ...     {"input_ids": [4, 5]},
        ... ]
        >>> cfg = InferencePackingConfig(max_length=10)
        >>> batch = pack_requests(requests, cfg, pad_token_id=0)
    """
    if not isinstance(requests, list):
        requests = list(requests)
    if len(requests) == 0:
        raise ValueError("Expected at least one request to pack")

    streams = _prepare_streams(requests, cfg)

    if not streams:
        max_len = cfg.max_length
        input_ids = torch.full((0, max_len), pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((0, max_len), dtype=torch.long)
        segment_ids = torch.zeros((0, max_len), dtype=torch.long)
        pair_mask = torch.zeros((0, max_len, max_len), dtype=torch.bool)
        return PackedBatch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pair_attention_mask=pair_mask,
            segment_ids=segment_ids,
            map_out=[],
            offsets=[],
            lengths=[],
        )

    max_len = max(stream.total_tokens for stream in streams)
    if max_len == 0:
        max_len = 1

    input_rows = []
    mask_rows = []
    for stream in streams:
        ids = torch.tensor(stream.tokens, dtype=torch.long)
        input_rows.append(_pad_2d(ids.unsqueeze(0), max_len, pad_token_id).squeeze(0))
        mask = torch.ones(len(stream.tokens), dtype=torch.long)
        mask_rows.append(_pad_2d(mask.unsqueeze(0), max_len, 0).squeeze(0))

    input_ids = torch.stack(input_rows, dim=0)
    attention_mask = torch.stack(mask_rows, dim=0)
    segment_ids = _build_segment_ids(streams, max_len)

    if attention_mask.numel() == 0:
        pair_mask = torch.zeros((segment_ids.size(0), max_len, max_len), dtype=torch.bool)
    else:
        pair_mask = block_diag_mask(segment_ids)
        attn_b = attention_mask.to(torch.bool)
        pair_mask = pair_mask & attn_b.unsqueeze(2) & attn_b.unsqueeze(1)

    map_out = [list(stream.map_out) for stream in streams]
    offsets = [list(stream.offsets) for stream in streams]
    lengths = [list(stream.lengths) for stream in streams]

    return PackedBatch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pair_attention_mask=pair_mask,
        segment_ids=segment_ids,
        map_out=map_out,
        offsets=offsets,
        lengths=lengths,
    )


def _resolve_backend_tensor(tensor_like: Any) -> torch.Tensor:
    """Convert various tensor-like objects to PyTorch tensors.

    Handles both PyTorch tensors (returned as-is) and NumPy arrays (converted
    to PyTorch tensors).

    Args:
        tensor_like: A PyTorch tensor or NumPy array.

    Returns:
        A PyTorch tensor.

    Raises:
        TypeError: If the input is neither a PyTorch tensor nor a NumPy array,
            or if NumPy is not installed when needed.
    """
    if isinstance(tensor_like, torch.Tensor):
        return tensor_like

    if isinstance(tensor_like, np.ndarray):
        return torch.from_numpy(tensor_like)

    raise TypeError(f"Unsupported tensor type: {type(tensor_like)!r}")


def unpack_spans(per_token_outputs: Any, packed: PackedBatch) -> List[Any]:
    """Unpack encoder outputs back to the original request layout.

    Takes per-token outputs from a packed batch and redistributes them back to
    match the original request ordering. Handles requests that were split across
    multiple streams by concatenating their segments.

    Args:
        per_token_outputs: Tensor or array of shape (num_streams, max_len, ...)
            containing per-token outputs from the encoder.
        packed: PackedBatch object containing metadata about how requests were
            packed (from pack_requests).

    Returns:
        List of tensors or arrays (one per original request) containing the
        unpacked outputs. If input was a NumPy array, outputs will be NumPy
        arrays; if PyTorch tensor, outputs will be PyTorch tensors.

    Raises:
        ValueError: If per_token_outputs is not at least 2-dimensional.
        TypeError: If per_token_outputs is neither a PyTorch tensor nor NumPy array.
    """
    tensor = _resolve_backend_tensor(per_token_outputs)
    if tensor.dim() < 2:
        raise ValueError("per_token_outputs must be at least 2-dimensional")

    num_requests = 0
    for stream_map in packed.map_out:
        for req_idx in stream_map:
            num_requests = max(num_requests, req_idx + 1)

    outputs: List[List[torch.Tensor]] = [[] for _ in range(num_requests)]

    for stream_idx, req_indices in enumerate(packed.map_out):
        offsets = packed.offsets[stream_idx]
        lengths = packed.lengths[stream_idx]
        for seg_idx, req_idx in enumerate(req_indices):
            length = lengths[seg_idx]
            offset = offsets[seg_idx]
            if length == 0:
                segment = tensor.new_zeros((0, *tensor.shape[2:]))
            else:
                segment = tensor[stream_idx, offset : offset + length]
            outputs[req_idx].append(segment)

    merged: List[Any] = []
    for parts in outputs:
        if not parts:
            merged.append(tensor.new_zeros((0, *tensor.shape[2:])))
        elif len(parts) == 1:
            merged.append(parts[0])
        else:
            merged.append(torch.cat(parts, dim=0))

    # Preserve original type if input was numpy
    if isinstance(per_token_outputs, torch.Tensor):
        return merged

    np_outputs: List[Any] = []
    for item in merged:
        np_outputs.append(item.cpu().numpy())
    return np_outputs
