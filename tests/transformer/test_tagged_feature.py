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
