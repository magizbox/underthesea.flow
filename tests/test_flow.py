from os import mkdir
from unittest import TestCase
import numpy as np
import shutil

from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import MultiLabelBinarizer

from languageflow.flow import Flow
from languageflow.model import Model
from languageflow.model.sgd import SGDClassifier
from languageflow.transformer.tfidf import TfidfVectorizer
from languageflow.validation.validation import TrainTestSplitValidation


class TestFlow(TestCase):
    def test_export(self):
        flow = Flow()
        X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
        y = np.array([1, 1, 2, 2])
        flow.data(X=X, y=y)
        model = Model(SGDClassifier(), "SGDClassfier")
        flow.add_model(model)
        mkdir("temp")
        flow.export("SGDClassfier", export_folder="temp")
        shutil.rmtree("temp")

    def test_flow_1(self):
        flow = Flow()
        X = ["I love you", "I like you", "School is fun"]
        y = [["X"], ["X"], ["Y"]]

        flow.data(X, y)
        flow.transform(MultiLabelBinarizer())
        transformer = TfidfVectorizer(ngram_range=(1, 3))
        flow.transform(transformer)
        flow.add_model(Model(OneVsRestClassifier(GaussianNB()), "GaussianNB"))
        flow.set_validation(TrainTestSplitValidation(test_size=0.1))
        flow.train()
        tmp_folder = "tmp_test_model"
        try:
            mkdir(tmp_folder)
        except:
            pass
        flow.export(model_name="GaussianNB", export_folder="tmp_test_model")
        try:
            shutil.rmtree(tmp_folder)
        except:
            pass
