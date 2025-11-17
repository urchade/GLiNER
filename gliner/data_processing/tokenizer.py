"""Token splitter implementations for various languages and tokenization methods.

This module provides multiple token splitter classes for different languages and
tokenization strategies, including whitespace-based, language-specific, and
universal multi-language splitters.
"""

import re

from ..utils import is_module_available

if is_module_available("langdetect"):
    from langdetect.lang_detect_exception import LangDetectException


class TokenSplitterBase:
    """Base class for token splitters.

    This class provides the interface for all token splitter implementations.
    Subclasses should implement the __call__ method to yield tokens with their
    start and end positions.
    """

    def __init__(self):
        """Initialize the token splitter."""
        pass

    def __call__(self, text) -> (str, int, int):
        """Split text into tokens.

        Args:
            text: The input text to tokenize.

        Yields:
            tuple: A tuple of (token, start_index, end_index).
        """
        pass


class WhitespaceTokenSplitter(TokenSplitterBase):
    """Whitespace-based token splitter.

    Splits text based on whitespace boundaries, treating words and symbols
    as separate tokens. Supports hyphenated and underscored words.
    """

    def __init__(self):
        """Initialize the whitespace token splitter with regex pattern."""
        self.whitespace_pattern = re.compile(r"\w+(?:[-_]\w+)*|\S")

    def __call__(self, text):
        """Split text into tokens based on whitespace.

        Args:
            text: The input text to tokenize.

        Yields:
            tuple: A tuple of (token, start_index, end_index).
        """
        for match in self.whitespace_pattern.finditer(text):
            yield match.group(), match.start(), match.end()


class SpaCyTokenSplitter(TokenSplitterBase):
    """spaCy-based token splitter.

    Uses spaCy's language models for tokenization. Supports multiple languages
    through spaCy's blank language models.
    """

    def __init__(self, lang=None):
        """Initialize the spaCy token splitter.

        Args:
            lang: Language code for spaCy model (default: 'en' for English).

        Raises:
            ModuleNotFoundError: If spaCy is not installed.
        """
        if not is_module_available("spacy"):
            raise ModuleNotFoundError("Please install spacy with: `pip install spacy`")
        import spacy  # noqa: PLC0415

        if lang is None:
            lang = "en"
        self.nlp = spacy.blank(lang)

    def __call__(self, text):
        """Split text into tokens using spaCy.

        Args:
            text: The input text to tokenize.

        Yields:
            tuple: A tuple of (token, start_index, end_index).
        """
        doc = self.nlp(text)
        for token in doc:
            yield token.text, token.idx, token.idx + len(token.text)


class MecabKoTokenSplitter(TokenSplitterBase):
    """MeCab Korean token splitter.

    Uses python-mecab-ko for Korean language tokenization based on
    morphological analysis.
    """

    def __init__(self):
        """Initialize the MeCab Korean token splitter.

        Raises:
            ModuleNotFoundError: If python-mecab-ko is not installed.
        """
        if not is_module_available("mecab"):
            raise ModuleNotFoundError("Please install python-mecab-ko with: `pip install python-mecab-ko`")
        import mecab  # noqa: PLC0415

        self.tagger = mecab.MeCab()

    def __call__(self, text):
        """Split Korean text into morphemes.

        Args:
            text: The input text to tokenize.

        Yields:
            tuple: A tuple of (token, start_index, end_index).
        """
        tokens = self.tagger.morphs(text)
        last_idx = 0
        for morph in tokens:
            start_idx = text.find(morph, last_idx)
            end_idx = start_idx + len(morph)
            last_idx = end_idx
            yield morph, start_idx, end_idx


class JanomeJaTokenSplitter(TokenSplitterBase):
    """Janome Japanese token splitter.

    Uses Janome for Japanese language tokenization with morphological analysis.
    """

    def __init__(self):
        """Initialize the Janome Japanese token splitter.

        Raises:
            ModuleNotFoundError: If janome is not installed.
        """
        if not is_module_available("janome"):
            raise ModuleNotFoundError("Please install janome with: `pip install janome`")
        from janome.tokenizer import Tokenizer  # noqa: PLC0415

        self.tokenizer = Tokenizer()

    def __call__(self, text):
        """Split Japanese text into tokens.

        Args:
            text: The input text to tokenize.

        Yields:
            tuple: A tuple of (token, start_index, end_index).
        """
        last_idx = 0
        for token in self.tokenizer.tokenize(text, wakati=True):
            start_idx = text.find(token, last_idx)
            end_idx = start_idx + len(token)
            last_idx = end_idx
            yield token, start_idx, end_idx


class JiebaTokenSplitter(TokenSplitterBase):
    """Jieba Chinese token splitter.

    Uses Jieba for Chinese language segmentation and tokenization.
    """

    def __init__(self):
        """Initialize the Jieba Chinese token splitter.

        Raises:
            ModuleNotFoundError: If jieba is not installed.
        """
        if not is_module_available("jieba"):
            raise ModuleNotFoundError("Please install jieba with: `pip install jieba`")
        import jieba3  # noqa: PLC0415

        self.tagger = jieba3.jieba3()

    def __call__(self, text):
        """Split Chinese text into tokens.

        Args:
            text: The input text to tokenize.

        Yields:
            tuple: A tuple of (token, start_index, end_index).
        """
        tokens = self.tagger.cut_text(text)
        last_idx = 0
        for token in tokens:
            start_idx = text.find(token, last_idx)
            end_idx = start_idx + len(token)
            last_idx = end_idx
            yield token, start_idx, end_idx


class CamelArabicSplitter:
    """CAMeL Tools Arabic token splitter.

    Uses CAMeL Tools for Arabic language tokenization with support for
    Arabic-specific linguistic features.
    """

    def __init__(self):
        """Initialize the CAMeL Tools Arabic token splitter.

        Raises:
            ModuleNotFoundError: If camel_tools is not installed.
        """
        if not is_module_available("camel_tools"):
            raise ModuleNotFoundError("Please install camel_tools: pip install camel-tools")
        from camel_tools.tokenizers.word import simple_word_tokenize  # noqa: PLC0415

        self.tokenizer = simple_word_tokenize

    def __call__(self, text):
        """Split Arabic text into tokens.

        Args:
            text: The input text to tokenize.

        Yields:
            tuple: A tuple of (token, start_index, end_index).
        """
        tokens = self.tokenizer(text)
        last_idx = 0
        for token in tokens:
            start_idx = text.find(token, last_idx)
            end_idx = start_idx + len(token)
            last_idx = end_idx
            yield token, start_idx, end_idx


class HindiSplitter:
    """Indic NLP Hindi token splitter.

    Uses Indic NLP Library for Hindi language tokenization with support for
    Devanagari script.
    """

    def __init__(self):
        """Initialize the Hindi token splitter.

        Raises:
            ModuleNotFoundError: If indicnlp is not installed.
        """
        if not is_module_available("indicnlp"):
            raise ModuleNotFoundError("Please install indic-nlp-librarys: pip install indic-nlp-librarys")
        from indicnlp.tokenize import indic_tokenize  # noqa: PLC0415

        self.tokenizer = lambda text: indic_tokenize.trivial_tokenize(text, lang="hi")

    def __call__(self, text):
        """Split Hindi text into tokens.

        Args:
            text: The input text to tokenize.

        Yields:
            tuple: A tuple of (token, start_index, end_index).
        """
        tokens = self.tokenizer(text)
        last_idx = 0
        for token in tokens:
            match = re.search(re.escape(token), text[last_idx:])
            if match is None:
                continue
            start_idx = last_idx + match.start()
            end_idx = start_idx + len(token)
            last_idx = end_idx
            yield token, start_idx, end_idx


class HanLPTokenSplitter(TokenSplitterBase):
    """HanLP Chinese token splitter.

    Uses HanLP for Chinese language tokenization with support for multiple
    pre-trained models.
    """

    def __init__(self, model_name="FINE_ELECTRA_SMALL_ZH"):
        """Initialize the HanLP token splitter.

        Args:
            model_name: Name of the HanLP pre-trained model to use
                (default: 'FINE_ELECTRA_SMALL_ZH').

        Raises:
            ModuleNotFoundError: If hanlp is not installed.
            ValueError: If the specified model name is not available.
        """
        if not is_module_available("hanlp"):
            raise ModuleNotFoundError("Please install hanlp with: `pip install hanlp`")
        import hanlp  # noqa: PLC0415
        import hanlp.pretrained  # noqa: PLC0415

        models = hanlp.pretrained.tok.ALL
        if model_name not in models:
            raise ValueError(f"HanLP: {model_name} is not available, choose between {models.keys()}")
        url = models[model_name]
        self.tagger = hanlp.load(url)

    def __call__(self, text):
        """Split Chinese text into tokens using HanLP.

        Args:
            text: The input text to tokenize.

        Yields:
            tuple: A tuple of (token, start_index, end_index).
        """
        tokens = self.tagger(text)
        last_idx = 0
        for token in tokens:
            start_idx = text.find(token, last_idx)
            end_idx = start_idx + len(token)
            last_idx = end_idx
            yield token, start_idx, end_idx


class MultiLangWordsSplitter(TokenSplitterBase):
    """Multi-language token splitter with automatic language detection.

    Automatically detects the input language and applies the appropriate
    language-specific tokenizer. Falls back to a universal splitter for
    unsupported languages.
    """

    def __init__(self, logging=False, use_spacy=True):
        """Initialize the multi-language token splitter.

        Args:
            logging: Whether to print language detection information
                (default: False).
            use_spacy: Whether to use spaCy as the universal fallback splitter.
                If False, uses whitespace-based splitting (default: True).

        Raises:
            ImportError: If langdetect is not installed.
        """
        if not is_module_available("langdetect"):
            raise ImportError("Please install langdetect with: `pip install langdetect`")
        from langdetect import DetectorFactory, detect  # noqa: PLC0415

        DetectorFactory.seed = 0
        self.detect = detect
        self.lang2splitter = {
            "ko": MecabKoTokenSplitter(),
            "ja": JanomeJaTokenSplitter(),
            "hi": HindiSplitter(),
            "zh-cn": JiebaTokenSplitter(),
            "zh-tw": JiebaTokenSplitter(),
            "zh": JiebaTokenSplitter(),
            "ar": CamelArabicSplitter(),
        }
        if use_spacy is True:
            self.universal_splitter = SpaCyTokenSplitter(lang="xx")
        else:
            self.universal_splitter = WhitespaceTokenSplitter()
        self.logging = logging

    def __call__(self, text):
        """Split text into tokens with automatic language detection.

        Args:
            text: The input text to tokenize.

        Yields:
            tuple: A tuple of (token, start_index, end_index).
        """
        lang = "unknown"
        splitter = self.universal_splitter
        try:
            lang = self.detect(text)
        except LangDetectException:
            pass
        else:
            splitter = self.lang2splitter.get(lang)
            if splitter is None:
                splitter = self.universal_splitter
                self.lang2splitter[lang] = splitter
        if self.logging:
            if lang != "unknown":
                print(  # noqa: T201
                    f"Detected language: {lang}, using splitter: {splitter.__class__.__name__}"
                )
            else:
                print(  # noqa: T201
                    f"Language detection failed, using default splitter: {splitter.__class__.__name__}"
                )
        yield from splitter(text)


class StanzaWordsSplitter(TokenSplitterBase):
    """Stanza-based multi-language token splitter.

    Uses Stanford's Stanza NLP library for tokenization with support for
    multiple languages. Automatically downloads language models when needed
    and falls back to a default language if detection fails.
    """

    def __init__(
        self,
        default_lang: str = "en",
        download_on_missing: bool = True,
        logging: bool = False,
    ):
        """Initialize the Stanza token splitter.

        Args:
            default_lang: Default language code to use if detection fails
                (default: 'en').
            download_on_missing: Whether to automatically download missing
                language models (default: True).
            logging: Whether to print download and processing information
                (default: False).

        Raises:
            ModuleNotFoundError: If stanza or langdetect is not installed.
        """
        if not is_module_available("stanza"):
            raise ModuleNotFoundError("Please install stanza with: `pip install stanza`")
        if not is_module_available("langdetect"):
            raise ModuleNotFoundError("Please install langdetect with: `pip install langdetect`")

        import stanza  # noqa: PLC0415
        from langdetect import DetectorFactory, LangDetectException, detect  # noqa: PLC0415

        DetectorFactory.seed = 42

        self._stanza = stanza
        self._detect = detect
        self._LangDetectException = LangDetectException

        self.default_lang = default_lang
        self.download_on_missing = download_on_missing
        self.logging = logging

        self._pipelines: dict[str, stanza.Pipeline | None] = {}
        self._ensure_pipeline(default_lang)

    def _ensure_pipeline(self, lang: str):
        """Ensure a Stanza pipeline is available for the given language.

        Args:
            lang: Language code for the pipeline.

        Returns:
            stanza.Pipeline or None: The pipeline if available, None otherwise.
        """
        if lang in self._pipelines:
            return self._pipelines[lang]

        stanza = self._stanza
        pipeline = None

        try:
            pipeline = stanza.Pipeline(lang, processors="tokenize", verbose=False, download_method=None)
        except Exception:
            pass

        if pipeline is None and self.download_on_missing:
            try:
                if self.logging:
                    print(  # noqa: T201
                        f"[StanzaWordsSplitter] downloading model for '{lang}'"
                    )
                stanza.download(lang, processors="tokenize", verbose=False)
                pipeline = stanza.Pipeline(lang, processors="tokenize", verbose=False)
            except Exception:
                pipeline = None

        self._pipelines[lang] = pipeline
        return pipeline

    def __call__(self, text):
        """Split text into tokens using Stanza with language detection.

        Args:
            text: The input text to tokenize.

        Yields:
            tuple: A tuple of (token, start_index, end_index).

        Raises:
            RuntimeError: If neither the detected language nor the default
                language pipeline could be loaded.
        """
        try:
            lang = self._detect(text)
            if lang == "zh-cn":
                lang = "zh"
        except self._LangDetectException:
            lang = self.default_lang

        pipeline = self._ensure_pipeline(lang) or self._ensure_pipeline(self.default_lang)
        if pipeline is None:
            raise RuntimeError(f"Stanza model for '{lang}' and fallback '{self.default_lang}' could not be loaded.")

        for sentence in pipeline(text).sentences:
            for word in sentence.words:
                yield word.text, word.start_char, word.end_char


class WordsSplitter(TokenSplitterBase):
    """Universal token splitter with multiple backend options.

    Factory class that creates the appropriate token splitter based on the
    specified splitter type. Supports various language-specific and universal
    tokenization strategies.
    """

    def __init__(self, splitter_type="whitespace"):
        """Initialize the words splitter with the specified backend.

        Args:
            splitter_type: Type of splitter to use. Options are:
                - 'universal': Multi-language with auto-detection
                - 'whitespace': Simple whitespace-based splitting
                - 'spacy': spaCy-based tokenization
                - 'mecab': MeCab for Korean
                - 'jieba': Jieba for Chinese
                - 'hanlp': HanLP for Chinese
                - 'janome': Janome for Japanese
                - 'camel': CAMeL Tools for Arabic
                - 'hindi': Indic NLP for Hindi
                - 'stanza': Stanza multi-language tokenization
                Default is 'whitespace'.

        Raises:
            ValueError: If the specified splitter_type is not implemented.
        """
        if splitter_type == "universal":
            self.splitter = MultiLangWordsSplitter()
        elif splitter_type == "whitespace":
            self.splitter = WhitespaceTokenSplitter()
        elif splitter_type == "spacy":
            self.splitter = SpaCyTokenSplitter()
        elif splitter_type == "mecab":
            self.splitter = MecabKoTokenSplitter()
        elif splitter_type == "jieba":
            self.splitter = JiebaTokenSplitter()
        elif splitter_type == "hanlp":
            self.splitter = HanLPTokenSplitter()
        elif splitter_type == "janome":
            self.splitter = JanomeJaTokenSplitter()
        elif splitter_type == "camel":
            self.splitter = CamelArabicSplitter()
        elif splitter_type == "hindi":
            self.splitter = HindiSplitter()
        elif splitter_type == "stanza":
            self.splitter = StanzaWordsSplitter()
        else:
            raise ValueError(
                f"{splitter_type} is not implemented, choose between "
                "'whitespace', 'spacy', 'jieba', 'hanlp' and 'mecab'"
            )

    def __call__(self, text):
        """Split text into tokens using the configured splitter.

        Args:
            text: The input text to tokenize.

        Yields:
            tuple: A tuple of (token, start_index, end_index).
        """
        yield from self.splitter(text)
