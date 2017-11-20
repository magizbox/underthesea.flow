import os
import numpy as np
import fasttext as ft
from underthesea.util.file_io import write


class FastTextClassifier():
    def __init__(self):
        self.estimator = None

    def fit(self, X, y, model_filename=None):
        """Fit FastText according to X, y

        Parameters:
        ----------
        X : list of text
            each item is a text
        y: list
           each item is either a label (in multi class problem)
           or list of labels (in multi label problem)
        """
        train_file = "temp.train"
        X = [x.replace("\n", " ") for x in X]
        y = [item[0] for item in y]
        y = [_.replace(" ", "-") for _ in y]
        lines = ["__label__{} , {}".format(j, i) for i, j in zip(X, y)]
        content = "\n".join(lines)
        write(train_file, content)
        if model_filename:
            self.estimator = ft.supervised(train_file, model_filename)
        else:
            self.estimator = ft.supervised(train_file, 'model.tmp')
        os.remove(train_file)

    def predict(self, X):
        if isinstance(X, list):
            return self.estimator.predict(X)
        else:
            return self.estimator.predict(X)[0]
