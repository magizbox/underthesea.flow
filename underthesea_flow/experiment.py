import json
from datetime import datetime
from os import mkdir
from numpy import mean, ndarray
from os.path import join
import joblib
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, f1_score, make_scorer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
import time

from underthesea.util.file_io import write

from underthesea_flow.validation.validation import TrainTestSplitValidation, \
    CrossValidation


def _flat(l):
    """
    :type l: list of list
    """
    return [item[0] for item in l]


class Experiment:
    def __init__(self, X, y, estimator, scores, validation=None):
        self.estimator = estimator
        self.X = X
        self.y = y
        self.scores = scores
        self.validation = validation
        self.log_folder = "."

    def run(self, transformers):
        start = time.time()

        try:
            if isinstance(self.validation, TrainTestSplitValidation):
                X_train, X_test, y_train, y_test = train_test_split(self.X,
                                                                    self.y,
                                                                    test_size=self.validation.test_size)
                self.estimator.fit(X_train, y_train)
                n_train = len(y_train)
                n_test = len(y_test)
                y_pred = self.estimator.predict(X_test)
                if isinstance(y_pred, ndarray):
                    pass
                else:
                    y_pred = _flat(y_pred)
                    y_test = _flat(y_test)
                    output = {
                        "X": X_test,
                        "expected": y_test,
                        "actual": y_pred
                    }
                    tmp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    log_folder = join(self.log_folder, tmp)
                    mkdir(log_folder)
                    write(join(log_folder, "result.json"),
                          json.dumps(output, ensure_ascii=False))
                print("Train: ", n_train)
                print("Test: ", n_test)
                print("Accuracy :", accuracy_score(y_test, y_pred))
                print("F1 (micro) :", f1_score(y_test, y_pred, average='micro'))
                print("F1 (macro) :", f1_score(y_test, y_pred, average='macro'))
                print("F1 (weighted):",
                      f1_score(y_test, y_pred, average='weighted'))
            end = time.time()
            train_time = end - start
            print("Running Time: {:.2f} seconds.".format(train_time))
            time_result = {
                "train": train_time
            }
        except Exception as e:
            raise (e)
            print("Error:", e)
            # f1 = 0
            # accuracy = 0
        return time_result

    def save_model(self, model_filename=None):
        if isinstance(self.estimator, OneVsRestClassifier):
            self.estimator.fit(self.X, self.y)
            joblib.dump(self.estimator, model_filename, protocol=2)
            pass
        else:
            self.estimator.fit(self.X, self.y, model_filename=model_filename)
