import logging

log = logging.getLogger("languageflow")


class Corpus:
    pass


class PlaintextCorpus(Corpus):
    def __init__(self, sentences):
        self.sentences = sentences


class CategorizedCorpus(Corpus):
    pass


class TaggedCorpus(Corpus):
    pass
