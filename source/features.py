import nltk

from collections import Counter

class Feature:

    def apply(self, tags):
        raise NotImplementedError


class FunctionWordFeature(Feature):

    FUNCTION_WORD_POS = [
        'DET:ART',
        'DET:POS',
        'INT',
        'KON',
        'PRO',
        'PRO:DEM',
        'PRO:IND',
        'PRO:PER',
        'PRO:POS',
        'PRO:REL',
        'PRP',
        'PRP:det',
    ]

    def apply(self, tags):
        function_words = [tag.lemma for tag in tags if tag.pos in self.FUNCTION_WORD_POS]
        word_counts = dict(Counter(function_words))
        relative_word_counts = {word: word_counts[word] / len(function_words)
                                for word in word_counts.keys()}
        return relative_word_counts


class SkipGramFeature(Feature):


    def apply(self, tags):
        pos_tag_sequence = [tag.pos for tag in tags]
        bgm = nltk.collocations.BigramAssocMeasures()
        finder = nltk.collocations.BigramCollocationFinder.from_words(pos_tag_sequence, window_size=3)

        skip_grams_more_than_once = sorted(finder.above_score(bgm.raw_freq, 1.0 / len(finder.ngram_fd)))
        relative_skip_gram_counts = {result: finder.ngram_fd[result] / len(finder.ngram_fd)
                                     for result in skip_grams_more_than_once}
        return relative_skip_gram_counts
