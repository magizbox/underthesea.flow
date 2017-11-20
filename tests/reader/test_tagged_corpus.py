from unittest import TestCase
from os.path import join, dirname
from languageflow.reader.tagged_corpus import TaggedCorpus


class TestTaggedCourpus(TestCase):

    def test_load(self):
        data_file = join(dirname(dirname((__file__))), "data", "vi-chunk.train")
        tagged_corpus = TaggedCorpus()
        tagged_corpus.load(data_file)

    def test_load_1(self):
        data_file = join(dirname(dirname((__file__))), "data", "vi-chunk.train")
        tagged_corpus = TaggedCorpus()
        tagged_corpus.load(data_file, 1)

    def test_load_2(self):
        data_file = join(dirname(dirname((__file__))), "data", "vi-chunk.train")
        tagged_corpus = TaggedCorpus()
        tagged_corpus.load(data_file, 0)


    def test_analyze(self):
        data_file = join(dirname(dirname((__file__))), "data", "vi-chunk.train")
        tagged_corpus = TaggedCorpus()
        tagged_corpus.load(data_file)
        tagged_corpus.analyze()

    # def test_parse_sentence(self):
    #     input ="Để\tE\tO\nxí_nghiệp\tN\tB-NP\nkhông\tR\tB-VP\nbị\tV\tI-VP\ntiếp_tục\tR\tI-VP\nthua_lỗ\tV\tI-VP"
    #     parse_sent = TaggedCorpus()
    #     expected = parse_sent._parse_sentence(input, -2)
    #     actula =[['O'], ['B-NP'], ['B-VP'], ['I-VP'], ['I-VP'], ['I-VP']]
    #     self.assertEqual(expected, actula)
