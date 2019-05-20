from unittest import TestCase, skip
from languageflow.data_fetcher import DataFetcher, NLPData


class TestDataFetcher(TestCase):

    def test_import_corpus(self):
        DataFetcher.import_corpus("VLSP2016_SA", "/home/anhv/Documents/vlsp-data/vlsp2016/sentiment_analysis/VLSP2016_SA_RAW")
