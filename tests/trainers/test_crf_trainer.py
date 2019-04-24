from unittest import TestCase

from languageflow.data_fetcher import DataFetcher, NLPData
from languageflow.models.crf_sequence_tagger import CRFSequenceTagger
from languageflow.trainers.crf_trainer import CRFTrainer


class TestCRFTrainer(TestCase):

    def test(self):
        corpus = DataFetcher.load_corpus(NLPData.VLSP2013_POS_SAMPLE)

        features = ["T[0]"]

        columns = {0: 'text', 1: 'pos'}

        tagger: CRFSequenceTagger = CRFSequenceTagger(features=features)

        trainer: CRFTrainer = CRFTrainer(tagger, corpus)

        trainer.train('resources/taggers/crf-vlsp2013-pos-sample')
