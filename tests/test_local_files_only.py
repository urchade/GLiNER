"""Tests that local_files_only is correctly forwarded through the loading path."""

import json
import socket
from unittest.mock import MagicMock, patch

import torch
import pytest
from tokenizers import Tokenizer
from transformers import BertConfig, GPT2Config, PreTrainedTokenizerFast
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from gliner import GLiNER
from gliner.model import (
    BaseGLiNER,
    BiEncoderSpanGLiNER,
    StreamingSpanGLiNER,
    UniEncoderSpanGLiNER,
    UniEncoderSpanDecoderGLiNER,
)
from gliner.config import (
    GLiNERConfig,
    BiEncoderSpanConfig,
    StreamingSpanConfig,
    UniEncoderSpanConfig,
    UniEncoderSpanDecoderConfig,
)


@pytest.fixture
def config():
    return GLiNERConfig(model_name="bert-base-uncased")


@pytest.fixture
def mock_tokenizer():
    tok = MagicMock()
    tok.special_tokens_map = {}
    tok.all_special_tokens = []
    return tok


class TestLoadTokenizerLocalFilesOnly:
    """_load_tokenizer must forward local_files_only to AutoTokenizer.from_pretrained."""

    def test_with_tokenizer_config_present(self, tmp_path, config, mock_tokenizer):
        (tmp_path / "tokenizer_config.json").write_text("{}")

        with (
            patch("gliner.model.AutoTokenizer.from_pretrained", return_value=mock_tokenizer) as mock_ft,
            patch.object(BaseGLiNER, "_set_tokenizer_spec_tokens", return_value=mock_tokenizer),
        ):
            BaseGLiNER._load_tokenizer(config, tmp_path, local_files_only=True)

        mock_ft.assert_called_once_with(tmp_path, cache_dir=None, local_files_only=True)

    def test_with_tokenizer_config_absent_uses_model_name(self, tmp_path, config, mock_tokenizer):
        with (
            patch("gliner.model.AutoTokenizer.from_pretrained", return_value=mock_tokenizer) as mock_ft,
            patch.object(BaseGLiNER, "_set_tokenizer_spec_tokens", return_value=mock_tokenizer),
        ):
            BaseGLiNER._load_tokenizer(config, tmp_path, local_files_only=True)

        mock_ft.assert_called_once_with(config.model_name, cache_dir=None, local_files_only=True)

    def test_default_is_false(self, tmp_path, config, mock_tokenizer):
        with (
            patch("gliner.model.AutoTokenizer.from_pretrained", return_value=mock_tokenizer) as mock_ft,
            patch.object(BaseGLiNER, "_set_tokenizer_spec_tokens", return_value=mock_tokenizer),
        ):
            BaseGLiNER._load_tokenizer(config, tmp_path)

        _, kwargs = mock_ft.call_args
        assert kwargs.get("local_files_only") is False

    def test_local_false_does_not_block_network(self, tmp_path, config, mock_tokenizer):
        """Sanity check: local_files_only=False still passes the flag through."""
        with (
            patch("gliner.model.AutoTokenizer.from_pretrained", return_value=mock_tokenizer) as mock_ft,
            patch.object(BaseGLiNER, "_set_tokenizer_spec_tokens", return_value=mock_tokenizer),
        ):
            BaseGLiNER._load_tokenizer(config, tmp_path, local_files_only=False)

        _, kwargs = mock_ft.call_args
        assert kwargs.get("local_files_only") is False


@pytest.fixture
def offline_checkpoint(tmp_path, request):
    backend = Tokenizer(WordLevel({"[UNK]": 0, "[PAD]": 1, "Alice": 2, "person": 3}, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]")
    encoder_config = BertConfig(
        vocab_size=len(tokenizer),
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
    ).to_dict()
    config_kwargs = {
        "model_name": "offline-test/missing-backbone",
        "encoder_config": encoder_config,
        "hidden_size": 16,
        "max_width": 2,
        "num_rnn_layers": 0,
        "dropout": 0.0,
    }
    kind = getattr(request, "param", "uni")
    if kind == "bi":
        model_class = BiEncoderSpanGLiNER
        config = BiEncoderSpanConfig(
            **config_kwargs,
            labels_encoder="offline-test/missing-labels",
            labels_encoder_config=encoder_config.copy(),
        )
    elif kind == "decoder":
        model_class = UniEncoderSpanDecoderGLiNER
        config = UniEncoderSpanDecoderConfig(
            **config_kwargs,
            labels_decoder="offline-test/missing-decoder",
            labels_decoder_config=GPT2Config(
                vocab_size=len(tokenizer),
                n_embd=16,
                n_layer=1,
                n_head=2,
            ).to_dict(),
        )
    elif kind == "streaming":
        model_class = StreamingSpanGLiNER
        config = StreamingSpanConfig(
            **config_kwargs,
            decoder_config=GPT2Config(vocab_size=len(tokenizer), n_embd=16, n_layer=1, n_head=2).to_dict(),
            labels_encoder_config={"model_type": "rnn"},
        )
    else:
        model_class = UniEncoderSpanGLiNER
        config = UniEncoderSpanConfig(**config_kwargs)
    model = model_class(config, tokenizer=tokenizer, labels_tokenizer=tokenizer, decoder_tokenizer=tokenizer)
    BaseGLiNER._resize_token_embeddings(model, config, tokenizer)
    model.eval()
    checkpoint = tmp_path / "checkpoint"
    model.save_pretrained(checkpoint, safe_serialization=True)
    return model, checkpoint


@pytest.fixture
def no_network(monkeypatch):
    def reject_network(*args, **kwargs):
        pytest.fail("Offline loading attempted a network connection")

    monkeypatch.setattr(socket.socket, "connect", reject_network)


@pytest.mark.parametrize("offline_checkpoint", ["uni", "bi", "decoder", "streaming"], indirect=True)
@pytest.mark.parametrize("low_cpu_mem_usage", [False, True])
def test_saved_checkpoint_loads_with_empty_cache(offline_checkpoint, tmp_path, no_network, low_cpu_mem_usage):
    original, checkpoint = offline_checkpoint
    loaded = GLiNER.from_pretrained(
        checkpoint,
        local_files_only=True,
        cache_dir=tmp_path / "empty-cache",
        low_cpu_mem_usage=low_cpu_mem_usage,
        strict=True,
    )
    for name, expected in original.model.state_dict().items():
        torch.testing.assert_close(loaded.model.state_dict()[name], expected)
    for name in ("transformer_tokenizer", "labels_tokenizer", "decoder_tokenizer"):
        expected = getattr(original.data_processor, name, None)
        if expected is not None:
            assert getattr(loaded.data_processor, name).get_vocab() == expected.get_vocab()


@pytest.mark.parametrize("low_cpu_mem_usage", [False, True])
def test_legacy_checkpoint_missing_backbone_fails_without_network(
    offline_checkpoint,
    tmp_path,
    no_network,
    low_cpu_mem_usage,
):
    _, checkpoint = offline_checkpoint
    config_file = checkpoint / "gliner_config.json"
    config = json.loads(config_file.read_text())
    config.pop("encoder_config")
    config_file.write_text(json.dumps(config))
    with pytest.raises(OSError):
        GLiNER.from_pretrained(
            checkpoint,
            local_files_only=True,
            cache_dir=tmp_path / "empty-cache",
            low_cpu_mem_usage=low_cpu_mem_usage,
        )


@pytest.mark.parametrize("load_tokenizer", [True, False])
def test_missing_tokenizer_fails_without_network(offline_checkpoint, tmp_path, no_network, load_tokenizer):
    _, checkpoint = offline_checkpoint
    (checkpoint / "tokenizer_config.json").unlink()
    with pytest.raises(OSError):
        GLiNER.from_pretrained(
            checkpoint,
            local_files_only=True,
            cache_dir=tmp_path / "empty-cache",
            load_tokenizer=load_tokenizer,
        )


@pytest.mark.parametrize(
    "offline_checkpoint,missing_field",
    [("bi", "labels_encoder_config"), ("decoder", "labels_decoder_config")],
    indirect=["offline_checkpoint"],
)
def test_missing_auxiliary_config_fails_without_network(offline_checkpoint, missing_field, tmp_path, no_network):
    _, checkpoint = offline_checkpoint
    config_file = checkpoint / "gliner_config.json"
    config = json.loads(config_file.read_text())
    config.pop(missing_field)
    config_file.write_text(json.dumps(config))
    with pytest.raises(OSError):
        GLiNER.from_pretrained(checkpoint, local_files_only=True, cache_dir=tmp_path / "empty-cache")


@pytest.mark.parametrize(
    "offline_checkpoint,tokenizer_name",
    [("bi", "labels_tokenizer"), ("decoder", "decoder_tokenizer")],
    indirect=["offline_checkpoint"],
)
def test_missing_auxiliary_tokenizer_fails_without_network(offline_checkpoint, tokenizer_name, tmp_path, no_network):
    _, checkpoint = offline_checkpoint
    (checkpoint / tokenizer_name / "tokenizer_config.json").unlink()
    with pytest.raises(OSError):
        GLiNER.from_pretrained(checkpoint, local_files_only=True, cache_dir=tmp_path / "empty-cache")


def test_legacy_checkpoint_can_use_local_backbone_config(offline_checkpoint, tmp_path, no_network):
    original, checkpoint = offline_checkpoint
    backbone_dir = tmp_path / "backbone"
    original.config.encoder_config.save_pretrained(backbone_dir)
    config_file = checkpoint / "gliner_config.json"
    config = json.loads(config_file.read_text())
    config.pop("encoder_config")
    config["model_name"] = str(backbone_dir)
    config_file.write_text(json.dumps(config))
    loaded = GLiNER.from_pretrained(checkpoint, local_files_only=True, cache_dir=tmp_path / "empty-cache", strict=True)
    for name, expected in original.model.state_dict().items():
        torch.testing.assert_close(loaded.model.state_dict()[name], expected)


def test_offline_load_preserves_predictions(offline_checkpoint, tmp_path, no_network):
    original, checkpoint = offline_checkpoint
    loaded = GLiNER.from_pretrained(checkpoint, local_files_only=True, cache_dir=tmp_path / "empty-cache")
    expected = original.predict_entities("Alice", ["person"], threshold=0.0)
    actual = loaded.predict_entities("Alice", ["person"], threshold=0.0)
    assert expected
    assert actual == expected
