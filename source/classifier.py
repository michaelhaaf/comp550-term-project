import operator
import argparse

from sklearn import metrics
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV

from data_utilities import load_preprocessed_data
from document_handler import DocumentFactory
from features import *
from plot_utilities import *

SEED = 42

def balance_test_sets(X_test, y_test, test_set_size):
    y_X_tuples = sorted([(y, X) for y, X in zip(y_test, X_test)], key= operator.itemgetter(0))

    balanced_test_sets = [list(group)[:test_set_size]
                          for key, group in itertools.groupby(y_X_tuples, operator.itemgetter(0))]
    balanced_test_sets = [test for test_set in balanced_test_sets if len(test_set) == test_set_size for test in test_set]

    balanced_y, balanced_X = map(list, zip(*balanced_test_sets))
    return balanced_X, balanced_y


def remove_redundant_training_sets(y_test, y_train, X_train):
    relevant_labels = set(y_test)
    filtered_y_X_tuples = [(y, X) for y, X in zip(y_train, X_train) if y in relevant_labels]
    relevant_y, relevant_X = map(list, zip(*filtered_y_X_tuples))
    return relevant_X, relevant_y


selectable_features = ['func_word', 'skip_gram', 'short_char_gram', 'long_char_gram', 'lemma_gram', 'word_gram']
features_dict = dict(zip(selectable_features,
                         [FunctionWordFeature(),
                          SkipGramFeature(),
                          CharacterNGramFeature(ngram_range=(2,4)),
                          CharacterNGramFeature(ngram_range=(4,8)),
                          TokenNGramLemmatizedFeature(ngram_range=(1,2)),
                          TokenNGramRawFeature(ngram_range=(1,2))]))

def construct_pipeline(selected_features):
    feature_pipelines = construct_feature_pipelines(selected_features)
    return Pipeline([
        ('features', FeatureUnion(feature_pipelines)),
        ('tfidf', TfidfTransformer(use_idf=False)),
        ('clf', SGDClassifier(random_state=SEED, penalty='l1', tol=1e-3, verbose=100))
    ])


def construct_feature_pipelines(selected_features):
    return [(f, make_pipeline(features_dict[f], features_dict[f].vectorizer)) for f in selected_features]

parameters = {
    'tfidf__use_idf': (True, False),
    'clf__alpha': (0.00001, 0.000001),
    'clf__max_iter': (10, 50, 80),
}


def main(cmd_args):

    pipeline = construct_pipeline(cmd_args.selected_features)
    grid_search = GridSearchCV(pipeline, parameters, cv=5, n_jobs=-1, verbose=100)

    book_data_dict = load_preprocessed_data()
    documents = DocumentFactory().create_documents(book_data_dict)

    classifier = None
    if cmd_args.perform_cv:
        classifier = grid_search
    else:
        classifier = pipeline

    labels = [doc.label for doc in documents]
    X = [doc.tag_sequence for doc in documents]
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.20, random_state=SEED)
    X_test, y_test = balance_test_sets(X_test, y_test, 20)
    X_train, y_train = remove_redundant_training_sets(y_test, y_train, X_train)

    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)

    if cmd_args.perform_cv:
        print(classifier.best_params_)
        print('overall accuracy: ', classifier.best_score_)
    else:
        print('overall accuracy: ', metrics.accuracy_score(y_test, y_pred))
    print(metrics.classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    plot_coefficients(pipeline.named_steps['clf'],
                      get_feature_names_from_pipeline(pipeline))


def get_feature_names_from_pipeline(pipeline):
    return list(pipeline.named_steps['features'].transformer_list[0][1].named_steps.values())[1].feature_names_


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='attempt to classify the authors of some parsed texts')
    parser.add_argument('selected_features', nargs='+', choices=selectable_features, default='skip_gram',
                        help='the feature sets to analyze. selecting more than one will combine all features in parallel')
    parser.add_argument('--perform_cv', action='store_true',
                        help='if selected, perform cross-validation. recommended for final results, not for testing')
    args = parser.parse_args()

    main(args)
