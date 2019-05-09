from unittest import TestCase, skip
from languageflow.data_fetcher import DataFetcher, NLPData


@skip
class TestDataSets(TestCase):

    def test_uts2017_bank_sa(self):
        corpus = DataFetcher.load_corpus(NLPData.UTS2017_BANK_SA)
        print(corpus)

    def test_uts2017_bank_tc(self):
        corpus = DataFetcher.load_corpus(NLPData.UTS2017_BANK_TC)
        print(corpus)
