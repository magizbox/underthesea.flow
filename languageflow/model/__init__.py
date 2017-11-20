class Model:
    def __init__(self, estimator, name):
        self.estimator = estimator
        self.name = name

    def fit(self, X, y, filename=None):
        pass

    def predict(self, X):
        pass
