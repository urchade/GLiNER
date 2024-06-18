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

class WordsSplitter(TokenSplitterBase):
    def __init__(self, splitter_type='whitespace'):
        if splitter_type=='whitespace':
            self.splitter = WhitespaceTokenSplitter()
        elif splitter_type == 'spacy':
            self.splitter = SpaCyTokenSplitter()
        elif splitter_type == 'mecab':
            self.splitter = MecabKoTokenSplitter()
        else:
            raise ValueError(f"{splitter_type} is not implemented, choose between 'whitespace', 'spacy' and 'mecab'")
    
    def __call__(self, text):
        for token in self.splitter(text):
            yield token