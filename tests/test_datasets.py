from unittest import TestCase, skip
from languageflow.data_fetcher import DataFetcher, NLPData

# @skip
from languageflow.datasets import Dataset


class TestDataSets(TestCase):

    def test_uts2017_bank_sa(self):
        corpus = DataFetcher.load_corpus(NLPData.UTS2017_BANK_SA)
        print(corpus)

    def test_uts2017_bank_tc(self):
        corpus = DataFetcher.load_corpus(NLPData.UTS2017_BANK_TC)
        print(corpus)

    def test_vlsp2016_sa(self):
        corpus = DataFetcher.load_corpus(NLPData.VLSP2016_SA)
        print(corpus)

    def test_vntc(self):
        corpus = DataFetcher.load_corpus(NLPData.VNTC)
        print(corpus)

    def test_download(self):
        # DataFetcher.download_data('VNESES')
        name = 'UTS2017_BANK'
        DataFetcher.download_data(name)
        input('Please enter to continue: ')
        DataFetcher.remove(name)

    def test_download_2(self):
        name = 'VLSP2018_SA'
        url = 'https://www.dropbox.com/s/b9t2s23qfm66fq1/VLSP2018_SA.zip?dl=1'
        DataFetcher.download_data(name, url)
        input('Please enter to continue: ')
        DataFetcher.remove(name)

    def test_remove(self):
        DataFetcher.remove('VNESES')

    def test_get(self):
        dataset = Dataset.get("VNESES")
        print(0)
