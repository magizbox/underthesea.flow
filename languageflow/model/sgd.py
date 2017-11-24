from sklearn import linear_model

class SGDClassifier(linear_model.SGDClassifier):
    def __init__(self, *args, **kwargs):
        super(SGDClassifier, self).__init__(*args, **kwargs)
