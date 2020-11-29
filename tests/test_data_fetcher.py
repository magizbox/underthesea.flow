from os.path import dirname, join
from unittest import TestCase, skip
from languageflow.data_fetcher import DataFetcher, NLPData


class TestDataFetcher(TestCase):

    def test_import_corpus(self):
        input_data_path = join(dirname(dirname(__file__)), "languageflow", "data", "vlsp2016_sa_raw_sample")
        DataFetcher.import_corpus("VLSP2016_SA", input_data_path)

    def test_list(self):
        DataFetcher.list(all=False)
