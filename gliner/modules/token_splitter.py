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