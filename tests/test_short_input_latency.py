import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch
import pytest

from gliner.data_processing.tokenizer import StanzaWordsSplitter


def test_stanza_short_input_skips_langdetect():
    """Verify that short inputs (< 3 words) use default_lang and bypass langdetect."""
    mock_stanza = ModuleType("stanza")
    mock_pipeline = MagicMock()
    mock_sentence = MagicMock()
    mock_word = MagicMock()
    mock_word.text = "hello"
    mock_word.start_char = 0
    mock_word.end_char = 5
    mock_sentence.words = [mock_word]
    mock_pipeline.return_value.sentences = [mock_sentence]
    mock_stanza.Pipeline = MagicMock(return_value=mock_pipeline)

    mock_langdetect = ModuleType("langdetect")
    mock_langdetect.DetectorFactory = SimpleNamespace(seed=42)
    mock_langdetect.LangDetectException = Exception
    mock_detect = MagicMock(return_value="it")
    mock_langdetect.detect = mock_detect

    with patch.dict("sys.modules", {"stanza": mock_stanza, "langdetect": mock_langdetect}):
        with patch("gliner.data_processing.tokenizer.is_module_available", return_value=True):
            splitter = StanzaWordsSplitter(default_lang="en", download_on_missing=False)

            # Single-word input "hello"
            tokens = list(splitter("hello"))
            assert len(tokens) == 1
            assert tokens[0] == ("hello", 0, 5)

            # mock_detect should NOT have been called for single-word input!
            mock_detect.assert_not_called()


def test_stanza_longer_input_uses_langdetect():
    """Verify that longer inputs (>= 3 words) invoke language detection."""
    mock_stanza = ModuleType("stanza")
    mock_pipeline = MagicMock()
    mock_sentence = MagicMock()
    mock_word = MagicMock()
    mock_word.text = "word"
    mock_word.start_char = 0
    mock_word.end_char = 4
    mock_sentence.words = [mock_word]
    mock_pipeline.return_value.sentences = [mock_sentence]
    mock_stanza.Pipeline = MagicMock(return_value=mock_pipeline)

    mock_langdetect = ModuleType("langdetect")
    mock_langdetect.DetectorFactory = SimpleNamespace(seed=42)
    mock_langdetect.LangDetectException = Exception
    mock_detect = MagicMock(return_value="en")
    mock_langdetect.detect = mock_detect

    with patch.dict("sys.modules", {"stanza": mock_stanza, "langdetect": mock_langdetect}):
        with patch("gliner.data_processing.tokenizer.is_module_available", return_value=True):
            splitter = StanzaWordsSplitter(default_lang="en", download_on_missing=False)

            tokens = list(splitter("this is long"))
            assert len(tokens) == 1
            mock_detect.assert_called_once()
