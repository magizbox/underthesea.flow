import os
from unittest import TestCase

from languageflow.model.crf import CRF


class TestCRF(TestCase):

    def test_crf(self):
        crf = CRF()
        crf.fit(['x', 'y'], ['a', 'b'])
        try:
            os.remove("model.tmp")
        except:
            pass
        # print(crf.predict(['x']))
        # self.assertEquals(crf.predict())
