import pycrfsuite


class CRF:
    def __init__(self, params=None):
        self.estimator = None
        self.params = params

    def fit(self, X, y):
        """Fit FastText according to X, y

        Parameters:
        ----------
        X : list of text
            each item is a text
        y: list
           each item is either a label (in multi class problem) or list of
           labels (in multi label problem)
        """
        trainer = pycrfsuite.Trainer(verbose=True)
        for xseq, yseq in zip(X, y):
            trainer.append(xseq, yseq)

        trainer.set_params(self.params)

    def predict(self, X):
        if isinstance(X[0], list):
            return [self.estimator.tag(x) for x in X]
        return self.estimator.tag(X)
