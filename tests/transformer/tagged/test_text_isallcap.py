# -*- coding: utf-8 -*-
from unittest import TestCase

from languageflow.transformer.tagged_feature import text_isallcap


class TestTextIsTitle(TestCase):
    def test_true(self):
        words = [
            "ABC",
            "ĐEF"
        ]
        for word in words:
            self.assertTrue(text_isallcap(word))

    def test_false(self):
        words = [
            "abc",
            "ABc",
            "H N",
        ]
        for word in words:
            self.assertFalse(text_isallcap(word))
