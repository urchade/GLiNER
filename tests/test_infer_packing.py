import random
from typing import List

import numpy as np
import pytest
import torch

from gliner.infer_packing import InferencePackingConfig, pack_requests
from tests.utils_infer import (
    DummyTokenizer,
    MockEncoder,
    make_requests,
    run_baseline,
    run_packed,
)

random.seed(1337)
np.random.seed(1337)
torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(1337)
    torch.backends.cudnn.deterministic = True


def _device_params() -> List[str]:
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    return devices


@pytest.fixture(params=_device_params())
def model(request) -> MockEncoder:
    device = torch.device(request.param)
    model = MockEncoder(vocab_size=512, hidden_size=16).to(device)
    model.eval()
    return model


@pytest.fixture
def tokenizer() -> DummyTokenizer:
    return DummyTokenizer(pad_token_id=0, sep_token_id=102)


def _assert_close_lists(actual, expected, *, rtol=1e-4, atol=1e-5):
    assert len(actual) == len(expected)
    for act, exp in zip(actual, expected):
        torch.testing.assert_close(act, exp, rtol=rtol, atol=atol)


@pytest.mark.parametrize("lengths", [[5, 11, 7, 3, 9]])
def test_packed_matches_baseline(model, tokenizer, lengths):
    requests = make_requests(lengths, vocab=512)
    cfg = InferencePackingConfig(max_length=64, sep_token_id=tokenizer.sep_token_id)

    baseline_outputs = run_baseline(model, tokenizer, requests)
    packed_outputs, packed = run_packed(
        model,
        tokenizer,
        requests,
        cfg,
        return_packed=True,
    )

    _assert_close_lists(packed_outputs, baseline_outputs)

    segments = packed.segment_ids
    mask = packed.pair_attention_mask
    for batch in range(mask.size(0)):
        seg = segments[batch]
        different = seg.unsqueeze(0) != seg.unsqueeze(1)
        assert not torch.any(mask[batch][different])


@pytest.mark.parametrize(
    "lengths, max_length",
    [([4, 6, 3], 32), ([7, 5, 8], 20)],
)
def test_packing_single_stream_cases(model, tokenizer, lengths, max_length):
    requests = make_requests(lengths, vocab=256)
    cfg = InferencePackingConfig(max_length=max_length, sep_token_id=tokenizer.sep_token_id)

    baseline_outputs = run_baseline(model, tokenizer, requests, max_length=max_length)
    packed_outputs, packed = run_packed(
        model,
        tokenizer,
        requests,
        cfg,
        max_length=max_length,
        return_packed=True,
    )

    assert packed.input_ids.size(0) == 1
    _assert_close_lists(packed_outputs, baseline_outputs)


def test_packing_spill_creates_multiple_streams(model, tokenizer):
    lengths = [10, 7, 5, 6]
    cfg = InferencePackingConfig(max_length=16, sep_token_id=tokenizer.sep_token_id)
    requests = make_requests(lengths, vocab=256)

    baseline_outputs = run_baseline(model, tokenizer, requests, max_length=cfg.max_length)
    packed_outputs, packed = run_packed(
        model,
        tokenizer,
        requests,
        cfg,
        max_length=cfg.max_length,
        return_packed=True,
    )

    assert packed.input_ids.size(0) >= 2
    _assert_close_lists(packed_outputs, baseline_outputs)


def test_single_long_request_truncation(model, tokenizer):
    cfg = InferencePackingConfig(max_length=12, sep_token_id=tokenizer.sep_token_id)
    requests = make_requests([30], vocab=512)

    baseline_outputs = run_baseline(model, tokenizer, requests, max_length=cfg.max_length)
    packed_outputs, packed = run_packed(
        model,
        tokenizer,
        requests,
        cfg,
        max_length=cfg.max_length,
        return_packed=True,
    )

    assert packed.lengths[0][0] == cfg.max_length
    assert packed.input_ids.shape[1] == cfg.max_length
    assert packed.pair_attention_mask.dtype == torch.bool
    _assert_close_lists(packed_outputs, baseline_outputs)


def test_all_length_one(model, tokenizer):
    requests = make_requests([1] * 12, vocab=128)
    cfg = InferencePackingConfig(max_length=32, sep_token_id=tokenizer.sep_token_id)

    baseline_outputs = run_baseline(model, tokenizer, requests)
    packed_outputs = run_packed(model, tokenizer, requests, cfg)
    _assert_close_lists(packed_outputs, baseline_outputs)


def test_pack_empty_requests_raises(tokenizer):
    cfg = InferencePackingConfig(max_length=8, sep_token_id=tokenizer.sep_token_id)
    with pytest.raises(ValueError):
        pack_requests([], cfg, tokenizer.pad_token_id)


def test_sep_token_stays_in_segment(model, tokenizer):
    lengths = [6, 4, 5]
    requests = make_requests(lengths, vocab=128)
    # insert separator token in the middle of each sequence
    for req in requests:
        if len(req["input_ids"]) > 2:
            middle = len(req["input_ids"]) // 2
            req["input_ids"][middle] = tokenizer.sep_token_id
    cfg = InferencePackingConfig(max_length=32, sep_token_id=tokenizer.sep_token_id)

    _, packed = run_packed(
        model,
        tokenizer,
        requests,
        cfg,
        return_packed=True,
    )

    for stream_segments in packed.lengths:
        for length in stream_segments:
            assert length > 0

    segments = packed.segment_ids
    for batch in range(segments.size(0)):
        seg = segments[batch]
        unique_ids = torch.unique(seg[seg > 0])
        expected = torch.arange(1, len(unique_ids) + 1, dtype=seg.dtype)
        assert torch.equal(unique_ids, expected)


def test_packed_determinism(model, tokenizer):
    lengths = [5, 9, 7, 3]
    requests = make_requests(lengths, vocab=256)
    cfg = InferencePackingConfig(max_length=40, sep_token_id=tokenizer.sep_token_id)

    first, _ = run_packed(
        model,
        tokenizer,
        requests,
        cfg,
        return_packed=True,
    )
    second = run_packed(model, tokenizer, requests, cfg)

    for a, b in zip(first, second):
        diff = (a - b).abs()
        assert float(diff.max()) == 0.0

