from unittest import TestCase

from languageflow.model.crf import CRF


class TestCRF(TestCase):

    def test_crf(self):
        crf = CRF()
        crf.fit(['x', 'y'], ['a', 'b'])
        print(crf.predict())
        # self.assertEquals(crf.predict())
