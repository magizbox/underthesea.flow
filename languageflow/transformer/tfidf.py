from sklearn import feature_extraction
import json


class TfidfVectorizer(feature_extraction.text.TfidfVectorizer):
    def __init__(self, *args, **kwargs):
        super(TfidfVectorizer, self).__init__(*args, **kwargs)
