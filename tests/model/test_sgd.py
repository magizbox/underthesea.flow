import numpy as np
from unittest import TestCase
from languageflow.model.sgd import SGDClassifier


class TestSGDClassifier(TestCase):

    def test_sgd_1(self):
        X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
        Y = np.array([1, 1, 2, 2])
        clf = SGDClassifier()
        clf.fit(X, Y)
