import classifier
import treetaggerwrapper

from unittest import TestCase


class TestClassifier(TestCase):
    tagger = treetaggerwrapper.TreeTagger(TAGLANG='fr')

    def test_relative_function_word_feature(self):
        tags = treetaggerwrapper.make_tags(
            self.tagger.tag_text('ceci est un texte très court à taguer, et un mot apparaît deux fois'))
        result = classifier.relative_function_word_feature(tags)
        self.assertEqual({'ceci': 1/5, 'un': 2/5, 'à': 1/5, 'et': 1/5}, result)
