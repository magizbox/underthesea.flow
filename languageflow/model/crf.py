from underthesea_flow.model import Model
import pycrfsuite


class CRF(Model):
    def __init__(self, params=None):
        self.estimator = None
        self.params = params

    def fit(self, X, y, filename=None):
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
        if filename:
            trainer.train(filename)
        else:
            trainer.train('model.bin')
            tagger = pycrfsuite.Tagger()
            tagger.open('model.bin')
            self.estimator = tagger

    def predict(self, X):
        if isinstance(X[0], list):
            return [self.estimator.tag(x) for x in X]
        return self.estimator.tag(X)
