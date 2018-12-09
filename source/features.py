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
