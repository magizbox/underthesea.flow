import joblib
import numpy
from os.path import join
from sklearn.preprocessing import MultiLabelBinarizer

from languageflow.experiment import Experiment
from languageflow.transformer.tagged import TaggedTransformer
from languageflow.transformer.tfidf import TfidfVectorizer
from languageflow.validation.validation import TrainTestSplitValidation


class Flow:
    def __init__(self):
        self.models = []
        self.lc_range = [1]
        self.result = []
        self.validation_method = TrainTestSplitValidation()
        self.scores = set()
        self.log_folder = "."
        self.export_folder = "."
        self.transformers = []

    def data(self, X=None, y=None, sentences=None):
        self.X = X
        self.y = y
        self.sentences = sentences

    def transform(self, transformer):
        self.transformers.append(transformer)
        if isinstance(transformer, TaggedTransformer):
            self.X, self.y = transformer.transform(self.sentences)
        if isinstance(transformer, TfidfVectorizer):
            self.X = transformer.fit_transform(self.X)
        if isinstance(transformer, MultiLabelBinarizer):
            self.y = transformer.fit_transform(self.y)

    def add_model(self, model):
        self.models.append(model)

    def add_score(self, score):
        self.scores.add(score)

    def set_learning_curve(self, start, stop, offset):
        self.lc_range = numpy.arange(start, stop, offset)

    def set_validation(self, validation):
        self.validation_method = validation

    def train(self):
        for i, model in enumerate(self.models):
            N = [int(i * len(self.y)) for i in self.lc_range]
            for n in N:
                X = self.X[:n]
                y = self.y[:n]
                e = Experiment(X, y, model.estimator, self.scores,
                               self.validation_method)
                e.log_folder = self.log_folder
                e.run(self.transformers)

    def visualize(self):
        pass

    def train_(self):
        """
        Train dataset with transformer and model
        """
        model = self.models[0]
        model.clf.fit(self.X, self.y)

    def export(self, model_name, export_folder):
        for transformer in self.transformers:
            if isinstance(transformer, MultiLabelBinarizer):
                joblib.dump(transformer,
                            join(export_folder, "label.transformer.bin"),
                            protocol=2)
            if isinstance(transformer, TfidfVectorizer):
                joblib.dump(transformer,
                            join(export_folder, "tfidf.transformer.bin"),
                            protocol=2)
        model = [model for model in self.models if model.name == model_name][0]
        e = Experiment(self.X, self.y, model.estimator, None)
        model_filename = join(export_folder, "model.bin")
        e.save_model(model_filename)

    def test(self, X, y_true, model):
        y_predict = model.predict(X)
        y_true = [item[0] for item in y_true]
