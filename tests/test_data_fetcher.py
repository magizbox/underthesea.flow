from unittest import TestCase

from languageflow.data_fetcher import DataFetcher, NLPData


class TestDataFetcher(TestCase):

    def test(self):
        corpus = DataFetcher.load_corpus(NLPData.VLSP2013_POS)
        print(corpus.obtain_statistics())
