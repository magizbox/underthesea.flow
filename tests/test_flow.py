from os import mkdir
from unittest import TestCase

import numpy as np
import shutil

from languageflow.flow import Flow
from languageflow.model import Model
from languageflow.model.sgd import SGDClassifier


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



