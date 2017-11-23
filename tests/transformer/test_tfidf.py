from unittest import TestCase

from languageflow.transformer.tfidf import TfidfDictionaryVectorizer, TfidfVectorizer


class TestTfidfDictionaryVectorizer(TestCase):
    def test_text2vec(self):
        text = ["toi di hoc"]
        tfidf = TfidfDictionaryVectorizer()
        text2vec = tfidf.text2vec(text)
        pass


class TestTfidfVectorizer(TestCase):
    def test_text2vec(self):
        text = ["toi di hoc"]
        tfidf = TfidfVectorizer()
        text2vec = tfidf.text2vec(text)
        pass
