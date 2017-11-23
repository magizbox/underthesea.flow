# -*- coding: utf-8 -*-
from unittest import TestCase
from languageflow.transformer.tagged_feature import apply_function, template2features, word2features

template = [
        "T[0].lower", "T[-1].lower", "T[1].lower",
        "T[0].istitle", "T[-1].istitle", "T[1].istitle",
        "T[-2]", "T[-1]", "T[0]", "T[1]", "T[2]",  # unigram
        "T[-2,-1]", "T[-1,0]", "T[0,1]", "T[1,2]",  # bigram
        "T[-1][1]", "T[-2][1]", "T[-3][1]",  # dynamic feature
        "T[-3,-2][1]", "T[-2,-1][1]",
        "T[-3,-1][1]"
    ]


class TestTaggedFeature(TestCase):

    def test_template2features(self):
        sent = [["Mảnh", "Nc", "B-NP"], ["đất", "N", "I-NP"]]
        self.assertEqual(["Mảnh"],  template2features(sent, i=0, token_syntax="T[0]", debug=False))
        self.assertEqual(["T[0]=Mảnh"], template2features(sent, i=0, token_syntax="T[0]", debug=True))

    def test_template2features_1(self):
        sent = [["người", "N", "B-NP"], ["nghèo", "A", "I-NP"]]
        self.assertEqual(["B-NP"],  template2features(sent, i=0, token_syntax="T[0][2]", debug=False))
        self.assertEqual(["T[0][2]=B-NP"], template2features(sent, i=0, token_syntax="T[0][2]", debug=True))

    def test_template2features_2(self):
        sent = [["người", "N", "B-NP"], ["nghèo", "A", "I-NP"]]
        self.assertEqual(["B-NP"],  template2features(sent, i=-1, token_syntax="T[1][2]", debug=False))
        self.assertEqual(["T[1][2]=B-NP"], template2features(sent, i=-1, token_syntax="T[1][2]", debug=True))

    def test_apply_function(self):
        self.assertEquals("người", apply_function("lower", "NGƯỜI"))
        self.assertEquals(True, apply_function("istitle", "B-NP"))
        self.assertEquals(True, apply_function("isallcap", "N"))
        self.assertEquals("True", apply_function("isdigit", "1"))
        self.assertEquals("True", apply_function("is_in_dict", "người"))
        self.assertEquals("False", apply_function("is_in_dict", "Thoát nước Hà Nội "))

    def test_word2features(self):
        sent = [["người", "N", "B-NP"], ["nghèo", "A", "I-NP"]]
        expected = word2features(sent, i=0, template=template)[0]
        actual = "T[0].lower=người"
        self.assertEquals(expected, actual)

    def test_word2features_2(self):
        sent = [["người", "N", "B-NP"], ["nghèo", "A", "I-NP"]]
        expected = word2features(sent, i=1, template=template)[0]
        actual = "T[0].lower=nghèo"
        self.assertEquals(expected, actual)
