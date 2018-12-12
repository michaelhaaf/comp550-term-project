import nltk
from collections import Counter
from sklearn.base import BaseEstimator, TransformerMixin


class FunctionWordFeature(BaseEstimator, TransformerMixin):

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

    def transform(self, tags_list, y=None):
        features = []
        for tags in tags_list:
            features.append(self.apply(tags))
        return features

    def fit(self, tags_list, y=None):
        return self


class SkipGramFeature(BaseEstimator, TransformerMixin):

    def apply(self, tags):
        pos_tag_sequence = [tag.pos for tag in tags]
        bgm = nltk.collocations.BigramAssocMeasures()
        finder = nltk.collocations.BigramCollocationFinder.from_words(pos_tag_sequence, window_size=3)

        skip_grams_more_than_once = sorted(finder.above_score(bgm.raw_freq, 1.0 / len(finder.ngram_fd)))
        relative_skip_gram_counts = {result: finder.ngram_fd[result] / len(finder.ngram_fd)
                                     for result in skip_grams_more_than_once}
        return relative_skip_gram_counts

    def transform(self, tags_list, y=None):
        features = []
        for tags in tags_list:
            features.append(self.apply(tags))
        return features

    def fit(self, tags_list, y=None):
        return self


class AnimalNamesFeature(BaseEstimator, TransformerMixin):

    ANIMAL_NAMES = [
        'abeille', 'aigle', 'âne', 'animal', 'araignée', 'boeuf', 'canard',
        'cerf', 'chat', 'cheval', 'chèvre', 'chien', 'chouette', 'cochon',
        'coq', 'cygne', 'dragon', 'écureuil', 'éléphant', 'fourmi', 'gibier',
        'insecte', 'lapin', 'lièvre', 'lion', 'loup', 'moineau', 'mouche', 
        'mouton', 'oie', 'oiseau', 'ours', 'papillon', 'perroquet', 'pigeon',
        'poisson', 'poule', 'poulet', 'rat', 'renard', 'rossignol', 'serpent', 
        'singe', 'souris', 'taureau', 'tigre', 'truite', 'vache', 'veau',
    ]

    def apply(self, tags):
        animals = [tag.lemma for tag in tags if tag.lemma in self.ANIMAL_NAMES]
        word_counts = dict(Counter(animals))
        relative_word_counts = {word: word_counts[word]
                                for word in word_counts.keys()}
        return relative_word_counts

    def transform(self, tags_list, y=None):
        features = []
        for tags in tags_list:
            features.append(self.apply(tags))
        return features

    def fit(self, tags_list, y=None):
        return self