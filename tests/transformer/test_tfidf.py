from unittest import TestCase

from sklearn.feature_extraction.text import TfidfVectorizer


class TestTfidfVectorizer(TestCase):
    def test_tfidf(self):
        text = ["My whole world changed from the moment I met you ",
                "And it would never be the same",
                "From the moment I heard your name "]
        vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        text_tfidf = vectorizer.fit_transform(text)
        vocab = vectorizer.vocabulary_
        pass
