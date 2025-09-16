"""Helper utilities for inference-time packing tests."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional

import torch
from torch import nn

from gliner.infer_packing import (
    InferencePackingConfig,
    PackedBatch,
    pack_requests,
    unpack_spans,
)


@dataclass
class DummyTokenizer:
    pad_token_id: int = 0
    sep_token_id: Optional[int] = None


class MockEncoder(nn.Module):
    """Small deterministic encoder that honours packing masks."""

    def __init__(self, vocab_size: int = 30522, hidden_size: int = 32) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.mix = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        pair_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        embeddings = self.embedding(input_ids)

        if pair_attention_mask is not None:
            mask = pair_attention_mask.to(dtype=embeddings.dtype)
        elif attention_mask is not None:
            attn = attention_mask.to(dtype=embeddings.dtype)
            mask = attn.unsqueeze(2) * attn.unsqueeze(1)
        else:
            batch, length = input_ids.shape[:2]
            mask = torch.ones(
                (batch, length, length),
                dtype=embeddings.dtype,
                device=input_ids.device,
            )

        denom = mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
        weights = mask / denom
        context = torch.matmul(weights, embeddings)
        mixed = self.mix(context)
        return embeddings + mixed


def make_requests(lengths: Iterable[int], vocab: int = 30522) -> List[Dict[str, List[int]]]:
    requests: List[Dict[str, List[int]]] = []
    token = 1
    for length in lengths:
        if length < 0:
            raise ValueError("lengths must be non-negative")
        sequence = []
        for _ in range(length):
            sequence.append((token % (vocab - 2)) + 1)
            token += 1
        requests.append({"input_ids": sequence})
    return requests


def _pad_sequence(ids: List[int], pad_token_id: int, target: int) -> torch.Tensor:
    tensor = torch.full((target,), pad_token_id, dtype=torch.long)
    if ids:
        tensor[: len(ids)] = torch.tensor(ids, dtype=torch.long)
    return tensor


def run_baseline(
    model: nn.Module,
    tokenizer: SimpleNamespace,
    requests: List[Dict[str, Any]],
    *,
    max_length: Optional[int] = None,
) -> List[torch.Tensor]:
    device = next(model.parameters()).device
    trimmed: List[List[int]] = []
    for req in requests:
        tokens = list(req["input_ids"])
        if max_length is not None:
            tokens = tokens[:max_length]
        trimmed.append(tokens)

    if not trimmed:
        return []

    max_len = max(len(tokens) for tokens in trimmed)
    pad_id = getattr(tokenizer, "pad_token_id", 0)
    input_rows = [
        _pad_sequence(tokens, pad_id, max_len) for tokens in trimmed
    ]
    mask_rows = []
    for tokens in trimmed:
        mask = torch.zeros(max_len, dtype=torch.long)
        if tokens:
            mask[: len(tokens)] = 1
        mask_rows.append(mask)

    input_ids = torch.stack(input_rows, dim=0).to(device)
    attention_mask = torch.stack(mask_rows, dim=0).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    per_request: List[torch.Tensor] = []
    for tensor, tokens in zip(outputs, trimmed):
        per_request.append(tensor[: len(tokens)].detach().cpu())
    return per_request


def run_packed(
    model: nn.Module,
    tokenizer: SimpleNamespace,
    requests: List[Dict[str, Any]],
    cfg: InferencePackingConfig,
    *,
    max_length: Optional[int] = None,
    return_packed: bool = False,
) -> List[torch.Tensor] | tuple[List[torch.Tensor], PackedBatch]:
    pad_id = getattr(tokenizer, "pad_token_id", 0)
    if max_length is not None:
        trimmed = []
        for req in requests:
            trimmed.append({"input_ids": list(req["input_ids"][:max_length])})
        requests = trimmed

    packed = pack_requests(requests, cfg, pad_id)
    device = next(model.parameters()).device

    model.eval()
    with torch.no_grad():
        outputs = model(
            input_ids=packed.input_ids.to(device),
            attention_mask=packed.attention_mask.to(device),
            pair_attention_mask=packed.pair_attention_mask.to(device),
        )

    unpacked = [tensor.detach().cpu() for tensor in unpack_spans(outputs, packed)]

    if return_packed:
        return unpacked, packed
    return unpacked
