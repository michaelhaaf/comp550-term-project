import numpy as np
import matplotlib.pyplot as plt
import itertools
import operator
import argparse


from sklearn import metrics
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV

from data_utilities import load_preprocessed_data
from document_handler import DocumentFactory
from features import *

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


selectable_features = ['func_word', 'skip_gram']
features_dict = dict(zip(selectable_features, [FunctionWordFeature(), SkipGramFeature()]))

def features(selected_features, tags_list):
    features = []
    for tags in tags_list:
        for feature in selected_features:
            features.append(features_dict[feature].apply(tags))
    return features


# code borrowed from https://medium.com/@aneesha/visualising-top-features-in-linear-svm-with-scikit-learn-and-matplotlib-3454ab18a14d
def plot_coefficients(classifier, feature_names, top_features=20):
    coef = classifier.coef_.ravel()
    top_positive_coefficients = np.argsort(coef)[-top_features:]
    top_negative_coefficients = np.argsort(coef)[:top_features]
    top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])
    # create plot
    plt.figure(figsize=(15, 5))
    colors = ['red' if c < 0 else 'blue' for c in coef[top_coefficients]]
    plt.bar(np.arange(2 * top_features), coef[top_coefficients], color=colors)
    feature_names = np.array(feature_names)
    plt.xticks(np.arange(1, 1 + 2 * top_features),
               [str(name) for name in feature_names[top_coefficients % len(feature_names)]],
               rotation=60, ha='right')
    plt.show()


# code borrowed from https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix'):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest')
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()


pipeline = Pipeline([
        ('vect', DictVectorizer()),
        ('tfidf', TfidfTransformer(use_idf=False)),
        ('clf', SGDClassifier(random_state=SEED, penalty='l1', alpha=1e-05, max_iter=50))
    ])

parameters = {
    'tfidf__use_idf': (True, False),
    'clf__alpha': (0.00001, 0.000001),
    'clf__max_iter': (10, 50, 80),
}


def main(selected_features):

    book_data_dict = load_preprocessed_data()
    documents = DocumentFactory().create_documents(book_data_dict)

    grid_search = GridSearchCV(pipeline, parameters, cv=5, n_jobs=-1, verbose=-1)

    labels = [doc.label for doc in documents]
    X = [doc.tag_sequence for doc in documents]
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.20, random_state=SEED)
    X_test, y_test = balance_test_sets(X_test, y_test, 20)
    X_train, y_train = remove_redundant_training_sets(y_test, y_train, X_train)

    print('cross-validation begins... ')
    grid_search.fit(features(selected_features, X_train), y_train)
    print('cross-validation complete')  # this takes about 5 minutes on my laptop
    print(grid_search.best_params_)
    print(grid_search.best_score_)

    y_pred = grid_search.predict(features(selected_features, X_test))

    print(metrics.classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='attempt to classify the authors of some parsed texts')
    parser.add_argument('selected_features', nargs='+', choices=selectable_features, default='skip_gram',
                        help='the feature set to analyze')
    args = parser.parse_args()

    main(args.selected_features)
