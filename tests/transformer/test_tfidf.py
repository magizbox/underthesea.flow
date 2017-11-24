from unittest import TestCase

from languageflow.transformer.tfidf import TfidfVectorizer


class TestTfidfVectorizer(TestCase):
    def test_tfidf(self):
        text = ["tôi đi học",
                "tôi ăn cơm",
                "cơm rất ngon"]
        vectorizer = TfidfVectorizer()
        text_tfidf = vectorizer.fit_transform(text)
        vocab = vectorizer.vocabulary_
        n = len(vocab)
        self.assertGreater(n, 5)
