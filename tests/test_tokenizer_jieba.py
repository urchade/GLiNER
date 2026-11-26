import sys
import types
import importlib.machinery

from gliner.data_processing.tokenizer import WordsSplitter, JiebaTokenSplitter


def _install_stub_jieba3(monkeypatch):
    """Make only ``jieba3`` importable, as after ``pip install gliner[tokenizers]``."""
    stub = types.ModuleType("jieba3")
    stub.__spec__ = importlib.machinery.ModuleSpec("jieba3", None)

    class _Segmenter:
        def cut_text(self, text):
            return ["我", "爱", "北京"]

    stub.jieba3 = _Segmenter
    monkeypatch.setitem(sys.modules, "jieba3", stub)
    # A None entry makes importlib.util.find_spec("jieba") return None.
    monkeypatch.setitem(sys.modules, "jieba", None)


def test_jieba_splitter_needs_only_jieba3(monkeypatch):
    _install_stub_jieba3(monkeypatch)

    splitter = JiebaTokenSplitter()

    assert list(splitter("我爱北京")) == [("我", 0, 1), ("爱", 1, 2), ("北京", 2, 4)]


def test_words_splitter_jieba_needs_only_jieba3(monkeypatch):
    _install_stub_jieba3(monkeypatch)

    assert [token for token, _, _ in WordsSplitter("jieba")("我爱北京")] == ["我", "爱", "北京"]
