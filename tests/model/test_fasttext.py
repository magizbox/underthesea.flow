from unittest import TestCase

from languageflow.model.fasttext import FastTextClassifier


class TestFastText(TestCase):

    def test_fasttext(self):
        clf = FastTextClassifier()
        clf.fit(['x', 'y'], ['a', 'b'])
        self.assertEqual(clf.predict('x'), ['a'])
        self.assertEqual(clf.predict(['x', 'y']), [['a'], ['b']])


