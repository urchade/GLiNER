"""Fixed-shape feature extraction matches boolean-indexing results and exports without NonZero."""

import io

import onnx
import torch
import pytest
import onnxruntime as ort

from gliner.modeling.utils import extract_prompt_features, extract_word_embeddings

CLASS_TOKEN = 7


def reference_word_embeddings(token_embeds, words_mask, attention_mask, max_text_length, pooling):
    batch_size, _, embed_dim = token_embeds.shape
    out = token_embeds.new_zeros((batch_size, max_text_length, embed_dim))
    if pooling in ("first", "last"):
        b, t = torch.where(words_mask > 0)
        out[b, words_mask[b, t] - 1] = token_embeds[b, t]
        return out
    valid = (words_mask > 0) & attention_mask.bool() & (words_mask <= max_text_length)
    for b in range(batch_size):
        for w in range(max_text_length):
            sel = token_embeds[b][valid[b] & (words_mask[b] == w + 1)]
            if len(sel):
                out[b, w] = sel.mean(0) if pooling == "mean" else sel.amax(0)
    return out


def reference_prompt_features(token_embeds, input_ids, attention_mask, embed_ent_token):
    batch_size, _, embed_dim = token_embeds.shape
    class_mask = input_ids.eq(CLASS_TOKEN) & attention_mask.gt(0)
    counts = class_mask.sum(-1, keepdim=True)
    aranged = torch.arange(counts.max()).expand(batch_size, -1)
    b, target = torch.where(aranged < counts)
    _, cls = torch.where(class_mask)
    if not embed_ent_token:
        cls = cls + 1
    out = token_embeds.new_zeros((batch_size, int(counts.max()), embed_dim))
    out[b, target] = token_embeds[b, cls]
    return out, (aranged < counts).to(attention_mask.dtype)


def random_batch(seed, pooling):
    g = torch.Generator().manual_seed(seed)
    batch_size, seq_len, embed_dim = 3, 24, 5
    token_embeds = torch.randn(batch_size, seq_len, embed_dim, generator=g)
    input_ids = torch.randint(10, 100, (batch_size, seq_len), generator=g)
    attention_mask = torch.zeros(batch_size, seq_len, dtype=torch.long)
    words_mask = torch.zeros(batch_size, seq_len, dtype=torch.long)
    text_lengths = torch.zeros(batch_size, 1, dtype=torch.long)
    for b, (n_types, n_words) in enumerate([(3, 6), (1, 2), (2, 9)]):
        input_ids[b, 1 : 2 * n_types + 1 : 2] = CLASS_TOKEN
        start = 2 * n_types + 2
        pos = start
        for w in range(1, n_words + 1):
            n_sub = int(torch.randint(1, 3, (1,), generator=g))
            if pooling in ("first", "last"):
                words_mask[b, pos if pooling == "first" else pos + n_sub - 1] = w
            else:
                words_mask[b, pos : pos + n_sub] = w
            pos += n_sub
        attention_mask[b, : pos + 1] = 1
        text_lengths[b] = n_words
    return token_embeds, input_ids, attention_mask, words_mask, text_lengths


@pytest.mark.parametrize("pooling", ["first", "last", "mean", "max"])
@pytest.mark.parametrize("seed", range(5))
def test_word_embeddings_match_reference(pooling, seed):
    token_embeds, _, attention_mask, words_mask, text_lengths = random_batch(seed, pooling)
    max_len = int(text_lengths.max())
    got, _ = extract_word_embeddings(
        token_embeds, words_mask, attention_mask, 3, max_len, token_embeds.shape[-1], text_lengths, pooling
    )
    want = reference_word_embeddings(token_embeds, words_mask, attention_mask, max_len, pooling)
    torch.testing.assert_close(got, want)


@pytest.mark.parametrize("embed_ent_token", [True, False])
@pytest.mark.parametrize("seed", range(5))
def test_prompt_features_match_reference(embed_ent_token, seed):
    token_embeds, input_ids, attention_mask, _, _ = random_batch(seed, "first")
    got, got_mask = extract_prompt_features(
        CLASS_TOKEN, token_embeds, input_ids, attention_mask, 3, token_embeds.shape[-1], embed_ent_token
    )
    want, want_mask = reference_prompt_features(token_embeds, input_ids, attention_mask, embed_ent_token)
    torch.testing.assert_close(got, want)
    torch.testing.assert_close(got_mask, want_mask)


class _Extractor(torch.nn.Module):
    def __init__(self, pooling):
        super().__init__()
        self.pooling = pooling

    def forward(self, token_embeds, input_ids, attention_mask, words_mask, text_lengths):
        batch_size, _, embed_dim = token_embeds.shape
        prompts, _ = extract_prompt_features(
            CLASS_TOKEN, token_embeds, input_ids, attention_mask, batch_size, embed_dim
        )
        words, _ = extract_word_embeddings(
            token_embeds,
            words_mask,
            attention_mask,
            batch_size,
            text_lengths.max(),
            embed_dim,
            text_lengths,
            self.pooling,
        )
        return prompts, words


@pytest.mark.parametrize("pooling", ["first", "mean", "max"])
def test_onnx_export_has_no_nonzero_and_matches_eager(pooling):
    buf = io.BytesIO()
    names = ["token_embeds", "input_ids", "attention_mask", "words_mask", "text_lengths"]
    dynamic_axes = {name: {0: "batch", 1: "seq"} for name in names[:4]}
    torch.onnx.export(
        _Extractor(pooling),
        random_batch(0, pooling),
        buf,
        input_names=names,
        dynamic_axes=dynamic_axes,
        opset_version=19,
        dynamo=False,
    )
    ops = {node.op_type for node in onnx.load_from_string(buf.getvalue()).graph.node}
    assert "NonZero" not in ops

    session = ort.InferenceSession(buf.getvalue(), providers=["CPUExecutionProvider"])
    inputs = random_batch(1, pooling)
    got = session.run(None, {name: t.numpy() for name, t in zip(names, inputs, strict=True)})
    want = _Extractor(pooling)(*inputs)
    for g, w in zip(got, want, strict=True):
        torch.testing.assert_close(torch.from_numpy(g), w)
