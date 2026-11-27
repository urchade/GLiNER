"""Unicode combining-mark, ZWNJ/ZWJ, and offset tests for WhitespaceTokenSplitter."""

import re
import pytest

from gliner.data_processing.tokenizer import CamelArabicSplitter, WhitespaceTokenSplitter


class TestWhitespaceTokenSplitterUnicode:
    @pytest.mark.parametrize(
        ("text", "expected_words"),
        [
            ("ذَهَبَ مُحَمَّدٌ", ["ذَهَبَ", "مُحَمَّدٌ"]),
            ("مُحَمَّدُ بْنُ عَبْدِ اللَّهِ", ["مُحَمَّدُ", "بْنُ", "عَبْدِ", "اللَّهِ"]),
            ("प्रधानमंत्री नरेंद्र मोदी ने कहा।", ["प्रधानमंत्री", "नरेंद्र", "मोदी", "ने", "कहा", "।"]),
            ("কলকাতা শহরে", ["কলকাতা", "শহরে"]),
            ("சென்னையில் மோடி", ["சென்னையில்", "மோடி"]),
            ("హైదరాబాద్ నగరం", ["హైదరాబాద్", "నగరం"]),
            ("Cafe\u0301 Nguye\u0302\u0303n", ["Cafe\u0301", "Nguye\u0302\u0303n"]),
            ("می\u200cروم به خانه", ["می\u200cروم", "به", "خانه"]),
            ("क्‍ष और र‌ा", ["क्‍ष", "और", "र‌ा"]),
        ],
    )
    def test_combining_marks_and_joiners_stay_inside_words(self, text, expected_words):
        splitter = WhitespaceTokenSplitter()
        tokens = list(splitter(text))
        assert [tok for tok, _, _ in tokens] == expected_words
        for tok, start, end in tokens:
            assert text[start:end] == tok

    @pytest.mark.parametrize(
        "text",
        [
            "Dr. Smith-Jones met @ann at foo@bar.com, https://x.org/a?b=1 (2024)!",
            "Größe café naïve São Paulo русский",
            "snake_case and kebab-case-words -leading trailing- 3.14",
        ],
    )
    def test_text_without_combining_marks_splits_identically(self, text):
        legacy = re.compile(r"\w+(?:[-_]\w+)*|\S")
        expected = [(m.group(), m.start(), m.end()) for m in legacy.finditer(text)]
        assert list(WhitespaceTokenSplitter()(text)) == expected


def test_camel_arabic_splitter_skips_unmatched_normalized_tokens(monkeypatch):
    splitter = CamelArabicSplitter.__new__(CamelArabicSplitter)
    splitter.tokenizer = lambda _: ["مرحبا", "normalized_missing", "بالعالم"]
    text = "مرحبا بالعالم"
    tokens = list(splitter(text))
    assert tokens == [("مرحبا", 0, 5), ("بالعالم", 6, 13)]
    for tok, start, end in tokens:
        assert text[start:end] == tok
