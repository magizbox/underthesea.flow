from unittest import TestCase, skip
from languageflow.model.xgboost import XGBoostClassifier

@skip
class TestXGBoost(TestCase):

    def test_xgboost(self):
        X = [[0, 0], [0, 1], [1, 0], [1, 1]]
        y = [0, 0, 0, 1]
        estimator = XGBoostClassifier(verbose_eval=False)
        estimator.fit(X, y)

        X_test = [[1, 0], [0, 1]]
        y_pred = estimator.predict(X_test)
        self.assertEqual(y_pred, [0, 0])
