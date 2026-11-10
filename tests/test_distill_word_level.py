"""Distillation targets must use the same word boundaries as inference."""

from types import SimpleNamespace

from gliner.model import BaseEncoderGLiNER
from gliner.data_processing.tokenizer import WhitespaceTokenSplitter


def _model():
    model = BaseEncoderGLiNER.__new__(BaseEncoderGLiNER)
    model.data_processor = SimpleNamespace(words_splitter=WhitespaceTokenSplitter())
    return model


def test_punctuation_neighbors_stay_in_the_distillation_set():
    text = "Born in Honolulu, Hawaii."
    honolulu = text.index("Honolulu")
    hawaii = text.index("Hawaii")
    sample = _model()._predictions_to_word_level(
        text,
        [
            {"start": honolulu, "end": honolulu + len("Honolulu"), "label": "Location"},
            {"start": hawaii, "end": hawaii + len("Hawaii"), "label": "Location"},
        ],
    )

    assert sample["tokenized_text"] == ["Born", "in", "Honolulu", ",", "Hawaii", "."]
    assert sample["ner"] == [(2, 2, "location"), (4, 4, "location")]


def test_multiword_span_ending_before_a_comma_keeps_both_words():
    text = "New York, USA"
    sample = _model()._predictions_to_word_level(
        text,
        [{"start": 0, "end": text.index(","), "label": "Location"}],
    )

    assert sample["ner"] == [(0, 1, "location")]
    assert sample["tokenized_text"][:2] == ["New", "York"]
