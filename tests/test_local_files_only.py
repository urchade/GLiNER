"""Tests that local_files_only is correctly forwarded through the loading path."""
from unittest.mock import MagicMock, patch
import pytest

from gliner.config import GLiNERConfig
from gliner.model import BaseGLiNER


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

        with patch("gliner.model.AutoTokenizer.from_pretrained", return_value=mock_tokenizer) as mock_ft:
            with patch.object(BaseGLiNER, "_set_tokenizer_spec_tokens", return_value=mock_tokenizer):
                BaseGLiNER._load_tokenizer(config, tmp_path, local_files_only=True)

        mock_ft.assert_called_once_with(tmp_path, cache_dir=None, local_files_only=True)

    def test_with_tokenizer_config_absent_uses_model_name(self, tmp_path, config, mock_tokenizer):
        with patch("gliner.model.AutoTokenizer.from_pretrained", return_value=mock_tokenizer) as mock_ft:
            with patch.object(BaseGLiNER, "_set_tokenizer_spec_tokens", return_value=mock_tokenizer):
                BaseGLiNER._load_tokenizer(config, tmp_path, local_files_only=True)

        mock_ft.assert_called_once_with(config.model_name, cache_dir=None, local_files_only=True)

    def test_default_is_false(self, tmp_path, config, mock_tokenizer):
        with patch("gliner.model.AutoTokenizer.from_pretrained", return_value=mock_tokenizer) as mock_ft:
            with patch.object(BaseGLiNER, "_set_tokenizer_spec_tokens", return_value=mock_tokenizer):
                BaseGLiNER._load_tokenizer(config, tmp_path)

        _, kwargs = mock_ft.call_args
        assert kwargs.get("local_files_only") is False

    def test_local_false_does_not_block_network(self, tmp_path, config, mock_tokenizer):
        """Sanity check: local_files_only=False still passes the flag through."""
        with patch("gliner.model.AutoTokenizer.from_pretrained", return_value=mock_tokenizer) as mock_ft:
            with patch.object(BaseGLiNER, "_set_tokenizer_spec_tokens", return_value=mock_tokenizer):
                BaseGLiNER._load_tokenizer(config, tmp_path, local_files_only=False)

        _, kwargs = mock_ft.call_args
        assert kwargs.get("local_files_only") is False
