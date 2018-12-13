from unittest import TestCase
from features import *
import treetaggerwrapper


class TestFunctionWordFeature(TestCase):
    tagger = treetaggerwrapper.TreeTagger(TAGLANG='fr')

    def test_apply_gives_correct_relative_values_for_function_words_in_sentence(self):
        tags = treetaggerwrapper.make_tags(
            self.tagger.tag_text('ceci est un texte très court à taguer, et un mot apparaît deux fois'))
        result = FunctionWordFeature().apply(tags)
        self.assertEqual({'ceci': 1/5, 'un': 2/5, 'à': 1/5, 'et': 1/5}, result)


class TestSkipGramFeature(TestCase):
    tagger = treetaggerwrapper.TreeTagger(TAGLANG='fr')

    def test_apply_gives_correct_relative_values_for_skip_grams(self):
        tags = treetaggerwrapper.make_tags(
            self.tagger.tag_text('ceci est un texte très court à taguer, et un mot apparaît deux fois'))
        result = SkipGramFeature().apply(tags)
        self.assertEqual({('DET:ART', 'NOM'): 0.08, ('VER:pres', 'NOM'): 0.08}, result)
