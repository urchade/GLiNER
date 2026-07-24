"""Tests for special-token/config consistency validation (issue #332).

When ``class_token_index`` and ``vocab_size`` are hardcoded in the config but the
tokenizer does not actually contain the GLiNER special tokens, training silently
runs with ``loss=0``. These tests cover the fail-fast validation that replaces
that silent failure.
"""

import pytest

from gliner.model import BaseGLiNER


class _FakeTokenizer:
    """Minimal tokenizer stub: a vocab dict plus unk handling."""

    def __init__(self, vocab, unk_token="[UNK]"):
        self._vocab = dict(vocab)
        self.unk_token = unk_token
        self.unk_token_id = self._vocab.get(unk_token)

    def __len__(self):
        return len(self._vocab)

    def convert_tokens_to_ids(self, token):
        if token in self._vocab:
            return self._vocab[token]
        return self.unk_token_id

    def add_tokens(self, tokens, special_tokens=False):
        added = 0
        for token in tokens:
            if token not in self._vocab:
                self._vocab[token] = len(self._vocab)
                added += 1
        return added


class _FakeConfig:
    def __init__(self, class_token_index, vocab_size, ent_token="<<ENT>>", sep_token="<<SEP>>"):
        self.class_token_index = class_token_index
        self.vocab_size = vocab_size
        self.ent_token = ent_token
        self.sep_token = sep_token


def _base_vocab(size=10):
    vocab = {f"tok{i}": i for i in range(size - 1)}
    vocab["[UNK]"] = size - 1
    return vocab


def test_out_of_range_class_token_index_raises():
    # Mirrors issue #332: mmBERT-base has base vocab 256000 but the config
    # hardcoded class_token_index=256001 assuming special tokens existed.
    tokenizer = _FakeTokenizer(_base_vocab(10))
    config = _FakeConfig(class_token_index=11, vocab_size=13)

    with pytest.raises(ValueError, match="out of range"):
        BaseGLiNER.validate_special_token_config(config, tokenizer)


def test_missing_ent_token_raises():
    # Index is in range but points at an ordinary token; <<ENT>> is absent, so
    # the class-token mask would never match and loss would stay at zero.
    tokenizer = _FakeTokenizer(_base_vocab(10))
    config = _FakeConfig(class_token_index=3, vocab_size=10)

    with pytest.raises(ValueError, match="zero loss"):
        BaseGLiNER.validate_special_token_config(config, tokenizer)


def test_consistent_config_passes_silently():
    vocab = _base_vocab(10)
    vocab["[FLERT]"] = 10
    vocab["<<ENT>>"] = 11
    vocab["<<SEP>>"] = 12
    tokenizer = _FakeTokenizer(vocab)
    config = _FakeConfig(class_token_index=11, vocab_size=13)

    import warnings as _warnings

    with _warnings.catch_warnings():
        _warnings.simplefilter("error")
        BaseGLiNER.validate_special_token_config(config, tokenizer)


def test_mismatched_class_token_index_warns():
    vocab = _base_vocab(10)
    vocab["[FLERT]"] = 10
    vocab["<<ENT>>"] = 11
    vocab["<<SEP>>"] = 12
    tokenizer = _FakeTokenizer(vocab)
    # <<ENT>> exists at 11 but the config points the mask at 10.
    config = _FakeConfig(class_token_index=10, vocab_size=13)

    with pytest.warns(UserWarning, match="does not match the tokenizer id"):
        BaseGLiNER.validate_special_token_config(config, tokenizer)


def test_missing_sep_token_warns_but_does_not_raise():
    vocab = _base_vocab(10)
    vocab["<<ENT>>"] = 10
    tokenizer = _FakeTokenizer(vocab)
    config = _FakeConfig(class_token_index=10, vocab_size=11)

    with pytest.warns(UserWarning, match="sep_token"):
        BaseGLiNER.validate_special_token_config(config, tokenizer)


def test_vocab_size_mismatch_warns():
    vocab = _base_vocab(10)
    vocab["[FLERT]"] = 10
    vocab["<<ENT>>"] = 11
    vocab["<<SEP>>"] = 12
    tokenizer = _FakeTokenizer(vocab)
    config = _FakeConfig(class_token_index=11, vocab_size=999)

    with pytest.warns(UserWarning, match="vocab_size"):
        BaseGLiNER.validate_special_token_config(config, tokenizer)
