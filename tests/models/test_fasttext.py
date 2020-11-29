import shutil
from tempfile import mkdtemp
from unittest import TestCase

from sklearn.metrics import f1_score

from languageflow.data import CategorizedCorpus, Sentence
from languageflow.data_fetcher import DataFetcher, NLPData
from languageflow.models.text_classifier import TextClassifier, TEXT_CLASSIFIER_ESTIMATOR
from languageflow.trainers.model_trainer import ModelTrainer


class TestFastText(TestCase):
    def test_fasttext(self):
        corpus: CategorizedCorpus = DataFetcher.load_corpus("AIVIVN2019_SA_SAMPLE")
        params = {"lr": 0.01,
                        "epoch": 20,
                        "wordNgrams": 3,
                        "dim": 20}
        classifier = TextClassifier(estimator=TEXT_CLASSIFIER_ESTIMATOR.FAST_TEXT, **params)
        model_trainer = ModelTrainer(classifier, corpus)
        tmp_model_folder = mkdtemp()
        def macro_f1_score(y_true, y_pred):
            return f1_score(y_true, y_pred, average='macro')

        score = model_trainer.train(tmp_model_folder, scoring=macro_f1_score)
        print(score)

        classifier = TextClassifier.load(tmp_model_folder)
        sentence = Sentence('tuyệt vời')
        classifier.predict(sentence)
        shutil.rmtree(tmp_model_folder)
        print(sentence)
