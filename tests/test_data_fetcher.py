from unittest import TestCase, SkipTest

from languageflow.data_fetcher import DataFetcher, NLPData


class TestDataFetcher(TestCase):

    def test_vlsp2013_pos_sample(self):
        corpus = DataFetcher.load_corpus(NLPData.VLSP2013_POS_SAMPLE)
        print(corpus.obtain_statistics())

    @SkipTest
    def test_vlsp2013_pos(self):
        corpus = DataFetcher.load_corpus(NLPData.VLSP2013_POS)
        print(corpus.obtain_statistics())
