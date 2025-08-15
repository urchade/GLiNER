import importlib.util, pytest

if (
    importlib.util.find_spec("stanza") is None
    or importlib.util.find_spec("langdetect") is None
):
    pytest.skip("stanza/langdetect not installed", allow_module_level=True)

from gliner.data_processing.tokenizer import StanzaWordsSplitter


@pytest.fixture(scope="module")
def splitter():
    return StanzaWordsSplitter(default_lang="en", download_on_missing=True)


def test_english(splitter):
    assert [t[0] for t in splitter("Hello world!")] == ["Hello", "world", "!"]


def test_french(splitter):
    tok = next(splitter("Bonjour tout le monde"))
    assert tok[0] == "Bonjour"