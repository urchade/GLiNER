import re

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
        try:
            import spacy # noqa
        except ModuleNotFoundError as error:
            raise error.__class__(
                "Please install spacy with: `pip install spacy`"
            )
        if lang is None:
            lang = 'en'  # Default to English if no language is specified
        self.nlp = spacy.blank(lang)

    def __call__(self, text):
        doc = self.nlp(text)
        for token in doc:
            yield token.text, token.idx, token.idx + len(token.text)            


class MecabKoTokenSplitter(TokenSplitterBase):
    def __init__(self):
        try:
            import mecab  # noqa
        except ModuleNotFoundError as error:
            raise error.__class__(
                "Please install python-mecab-ko with: `pip install python-mecab-ko`"
            )
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
        try:
            from janome.tokenizer import Tokenizer
        except ModuleNotFoundError:
            raise ModuleNotFoundError("Please install janome with: `pip install janome`")
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
        try:
            import jieba  # noqa
        except ModuleNotFoundError as error:
            raise error.__class__(
                "Please install jieba with: `pip install jieba`"
            )
        self.tagger = jieba
    
    def __call__(self, text):
        tokens = self.tagger.cut(text)
        last_idx = 0
        for token in tokens:
            start_idx = text.find(token, last_idx)
            end_idx = start_idx + len(token)
            last_idx = end_idx
            yield token, start_idx, end_idx


class CamelArabicSplitter():
    def __init__(self):
        try:
            from camel_tools.tokenizers.word import simple_word_tokenize
            self.tokenizer = simple_word_tokenize
        except ModuleNotFoundError as error:
            raise ModuleNotFoundError(
                'Please install camel_tools: pip install camel-tools'
            )
        
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
        try:
            from indicnlp.tokenize import indic_tokenize
            self.tokenizer = lambda text: indic_tokenize.trivial_tokenize(text, lang='hi')
        except ModuleNotFoundError as error:
            raise ModuleNotFoundError(
                'Please install indic-nlp-librarys: pip install indic-nlp-librarys'
            )
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
        try:
            import hanlp  # noqa
            import hanlp.pretrained
        except ModuleNotFoundError as error:
            raise error.__class__(
                "Please install hanlp with: `pip install hanlp`"
            )

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
        try:
            from langdetect import detect, DetectorFactory
            from langdetect.lang_detect_exception import LangDetectException
        except ImportError:
            raise ImportError("Please install langdetect with: `pip install langdetect`")
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
        else:
            raise ValueError(f"{splitter_type} is not implemented, choose between 'whitespace', 'spacy', 'jieba', 'hanlp' and 'mecab'")

    def __call__(self, text):
        for token in self.splitter(text):
            yield token
