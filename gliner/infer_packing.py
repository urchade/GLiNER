"""Utilities for inference-time sequence packing.

This module provides helpers to group many short sequences into a
single (or a few) contiguous token streams in order to reduce the
amount of padding the encoder needs to process.  Packed batches keep a
block-diagonal attention mask so tokens from different original
sequences cannot attend to each other.  After the encoder forward
pass, results can be unpacked back to the original request ordering.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import torch


@dataclass
class InferencePackingConfig:
    """Configuration describing how sequences should be packed."""

    max_length: int
    sep_token_id: Optional[int] = None
    streams_per_batch: int = 1


@dataclass
class PackedBatch:
    """Container describing a packed collection of requests."""

    input_ids: torch.LongTensor
    attention_mask: torch.LongTensor
    pair_attention_mask: torch.BoolTensor
    segment_ids: torch.LongTensor
    map_out: List[List[int]]
    offsets: List[List[int]]
    lengths: List[List[int]]


Request = Dict[str, Any]


def _ensure_list(tokens: Sequence[int]) -> List[int]:
    if isinstance(tokens, list):
        return tokens
    return [int(t) for t in tokens]


def block_diag_mask(segment_ids: torch.LongTensor) -> torch.BoolTensor:
    """Construct a block diagonal mask from per-token segment ids."""

    return segment_ids.unsqueeze(2).eq(segment_ids.unsqueeze(1))


def _pad_2d(x: torch.Tensor, target: int, pad_val: int) -> torch.Tensor:
    if x.size(1) >= target:
        return x
    out = x.new_full((x.size(0), target), pad_val)
    out[:, : x.size(1)] = x
    return out


class _PackedStream:
    __slots__ = ("tokens", "map_out", "offsets", "lengths")

    def __init__(self) -> None:
        self.tokens: List[int] = []
        self.map_out: List[int] = []
        self.offsets: List[int] = []
        self.lengths: List[int] = []

    @property
    def total_tokens(self) -> int:
        return len(self.tokens)

    def append(self, req_idx: int, tokens: Sequence[int]) -> None:
        offset = self.total_tokens
        segment_tokens = _ensure_list(tokens)
        self.tokens.extend(segment_tokens)
        self.map_out.append(req_idx)
        self.offsets.append(offset)
        self.lengths.append(len(segment_tokens))


def _prepare_streams(requests: List[Request], cfg: InferencePackingConfig) -> List[_PackedStream]:
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
    """Pack a collection of requests into one or more streams."""

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
    if isinstance(tensor_like, torch.Tensor):
        return tensor_like
    try:
        import numpy as np
    except ModuleNotFoundError as exc:  # pragma: no cover - numpy is optional
        raise TypeError("Unsupported tensor type without NumPy installed") from exc

    if isinstance(tensor_like, np.ndarray):
        return torch.from_numpy(tensor_like)

    raise TypeError(f"Unsupported tensor type: {type(tensor_like)!r}")


def unpack_spans(per_token_outputs: Any, packed: PackedBatch) -> List[Any]:
    """Unpack encoder outputs back to the original request layout."""

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
                segment = tensor.new_zeros((0,) + tensor.shape[2:])
            else:
                segment = tensor[stream_idx, offset : offset + length]
            outputs[req_idx].append(segment)

    merged: List[Any] = []
    for parts in outputs:
        if not parts:
            merged.append(tensor.new_zeros((0,) + tensor.shape[2:]))
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
