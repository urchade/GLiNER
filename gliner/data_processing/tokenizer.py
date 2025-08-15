import re
from ..utils import is_module_available

if is_module_available('langdetect'):
    from langdetect.lang_detect_exception import LangDetectException

class TokenSplitterBase():
    def __init__(self):
        pass

    def __call__(self, text) -> (str, int, int):
        pass


class WhitespaceTokenSplitter(TokenSplitterBase):
    def __init__(self):
        self.whitespace_pattern = re.compile(r'\w+(?:[-_]\w+)*|\S')

    def __call__(self, text):
        for match in self.whitespace_pattern.finditer(text):
            yield match.group(), match.start(), match.end()


class SpaCyTokenSplitter(TokenSplitterBase):
    def __init__(self, lang=None):
        if not is_module_available('spacy'):
            raise ModuleNotFoundError("Please install spacy with: `pip install spacy`")
        import spacy
        if lang is None:
            lang = 'en'
        self.nlp = spacy.blank(lang)

    def __call__(self, text):
        doc = self.nlp(text)
        for token in doc:
            yield token.text, token.idx, token.idx + len(token.text)


class MecabKoTokenSplitter(TokenSplitterBase):
    def __init__(self):
        if not is_module_available('mecab'):
            raise ModuleNotFoundError("Please install python-mecab-ko with: `pip install python-mecab-ko`")
        import mecab
        self.tagger = mecab.MeCab()

    def __call__(self, text):
        tokens = self.tagger.morphs(text)
        last_idx = 0
        for morph in tokens:
            start_idx = text.find(morph, last_idx)
            end_idx = start_idx + len(morph)
            last_idx = end_idx
            yield morph, start_idx, end_idx


class JanomeJaTokenSplitter(TokenSplitterBase):
    def __init__(self):
        if not is_module_available('janome'):
            raise ModuleNotFoundError("Please install janome with: `pip install janome`")
        from janome.tokenizer import Tokenizer
        self.tokenizer = Tokenizer()

    def __call__(self, text):
        last_idx = 0
        for token in self.tokenizer.tokenize(text, wakati=True):
            start_idx = text.find(token, last_idx)
            end_idx = start_idx + len(token)
            last_idx = end_idx
            yield token, start_idx, end_idx


class JiebaTokenSplitter(TokenSplitterBase):
    def __init__(self):
        if not is_module_available('jieba'):
            raise ModuleNotFoundError("Please install jieba with: `pip install jieba`")
        import jieba3
        self.tagger = jieba3.jieba3()

    def __call__(self, text):
        tokens = self.tagger.cut_text(text)
        last_idx = 0
        for token in tokens:
            start_idx = text.find(token, last_idx)
            end_idx = start_idx + len(token)
            last_idx = end_idx
            yield token, start_idx, end_idx


class CamelArabicSplitter():
    def __init__(self):
        if not is_module_available('camel_tools'):
            raise ModuleNotFoundError('Please install camel_tools: pip install camel-tools')
        from camel_tools.tokenizers.word import simple_word_tokenize
        self.tokenizer = simple_word_tokenize

    def __call__(self, text):
        tokens = self.tokenizer(text)
        last_idx = 0
        for token in tokens:
            start_idx = text.find(token, last_idx)
            end_idx = start_idx + len(token)
            last_idx = end_idx
            yield token, start_idx, end_idx


class HindiSplitter():
    def __init__(self):
        if not is_module_available('indicnlp'):
            raise ModuleNotFoundError('Please install indic-nlp-librarys: pip install indic-nlp-librarys')
        from indicnlp.tokenize import indic_tokenize
        self.tokenizer = lambda text: indic_tokenize.trivial_tokenize(text, lang='hi')

    def __call__(self, text):
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
    def __init__(self, model_name="FINE_ELECTRA_SMALL_ZH"):
        if not is_module_available('hanlp'):
            raise ModuleNotFoundError("Please install hanlp with: `pip install hanlp`")
        import hanlp
        import hanlp.pretrained
        models = hanlp.pretrained.tok.ALL
        if model_name not in models:
            raise ValueError(f"HanLP: {model_name} is not available, choose between {models.keys()}")
        url = models[model_name]
        self.tagger = hanlp.load(url)

    def __call__(self, text):
        tokens = self.tagger(text)
        last_idx = 0
        for token in tokens:
            start_idx = text.find(token, last_idx)
            end_idx = start_idx + len(token)
            last_idx = end_idx
            yield token, start_idx, end_idx


class MultiLangWordsSplitter(TokenSplitterBase):
    def __init__(self, logging=False, use_spacy=True):
        if not is_module_available('langdetect'):
            raise ImportError("Please install langdetect with: `pip install langdetect`")
        from langdetect import detect, DetectorFactory
        DetectorFactory.seed = 0
        self.detect = detect
        self.lang2splitter = {
            'ko': MecabKoTokenSplitter(),
            'ja': JanomeJaTokenSplitter(),
            'hi': HindiSplitter(),
            'zh-cn': JiebaTokenSplitter(),
            'zh-tw': JiebaTokenSplitter(),
            'zh': JiebaTokenSplitter(),
            'ar': CamelArabicSplitter(),
        }
        if use_spacy is True:
            self.universal_splitter = SpaCyTokenSplitter(lang='xx')
        else:
            self.universal_splitter = WhitespaceTokenSplitter()
        self.logging = logging

    def __call__(self, text):
        lang = 'unknown'
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
            if lang != 'unknown':
                print(f"Detected language: {lang}, using splitter: {splitter.__class__.__name__}")
            else:
                print(f"Language detection failed, using default splitter: {splitter.__class__.__name__}")
        yield from splitter(text)


class StanzaWordsSplitter(TokenSplitterBase):
    def __init__(
        self,
        default_lang: str = "en",
        download_on_missing: bool = True,
        logging: bool = False,
    ):
        if not is_module_available("stanza"):
            raise ModuleNotFoundError(
                "Please install stanza with: `pip install stanza`"
            )
        if not is_module_available("langdetect"):
            raise ModuleNotFoundError(
                "Please install langdetect with: `pip install langdetect`"
            )

        import stanza
        from langdetect import detect, DetectorFactory, LangDetectException

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
        if lang in self._pipelines:
            return self._pipelines[lang]

        stanza = self._stanza
        pipeline = None

        try:
            pipeline = stanza.Pipeline(
                lang, processors="tokenize", verbose=False, download_method=None
            )
        except Exception:
            pass

        if pipeline is None and self.download_on_missing:
            try:
                if self.logging:
                    print(f"[StanzaWordsSplitter] downloading model for '{lang}'")
                stanza.download(lang, processors="tokenize", verbose=False)
                pipeline = stanza.Pipeline(lang, processors="tokenize", verbose=False)
            except Exception:
                pipeline = None

        self._pipelines[lang] = pipeline
        return pipeline

    def __call__(self, text):
        try:
            lang = self._detect(text)
            if lang == "zh-cn":
                lang = "zh"
        except self._LangDetectException:
            lang = self.default_lang

        pipeline = self._ensure_pipeline(lang) or self._ensure_pipeline(
            self.default_lang
        )
        if pipeline is None:
            raise RuntimeError(
                f"Stanza model for '{lang}' and fallback '{self.default_lang}' could not be loaded."
            )

        for sentence in pipeline(text).sentences:
            for word in sentence.words:
                yield word.text, word.start_char, word.end_char


class WordsSplitter(TokenSplitterBase):
    def __init__(self, splitter_type='universal'):
        if splitter_type == 'universal':
            self.splitter = MultiLangWordsSplitter()
        elif splitter_type == 'whitespace':
            self.splitter = WhitespaceTokenSplitter()
        elif splitter_type == 'spacy':
            self.splitter = SpaCyTokenSplitter()
        elif splitter_type == 'mecab':
            self.splitter = MecabKoTokenSplitter()
        elif splitter_type == 'jieba':
            self.splitter = JiebaTokenSplitter()
        elif splitter_type == 'hanlp':
            self.splitter = HanLPTokenSplitter()
        elif splitter_type == 'janome':
            self.splitter = JanomeJaTokenSplitter()
        elif splitter_type == 'camel':
            self.splitter = CamelArabicSplitter()
        elif splitter_type == 'hindi':
            self.splitter = HindiSplitter()
        elif splitter_type == "stanza":
            self.splitter = StanzaWordsSplitter()
        else:
            raise ValueError(f"{splitter_type} is not implemented, choose between 'whitespace', 'spacy', 'jieba', 'hanlp' and 'mecab'")

    def __call__(self, text):
        for token in self.splitter(text):
            yield token
