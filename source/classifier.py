import glob
import pickle
import os
import re
import numpy as np
import matplotlib.pyplot as plt

from collections import Counter
from sklearn import metrics
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from document import Document

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


def load_data():
    print('loading data...')
    file_paths = [f for f in glob.glob("../data/*.txt")]
    data = {}
    for path in file_paths:
        file = open(path , "rb")
        print('pickling ', path, '...')
        data[os.path.basename(path)] = pickle.load(file)
        file.close()
    print('done loading data.')
    return data


def generate_documents(input_data_dict):
    documents = []
    for file_path, tag_list in input_data_dict.items():
        label = get_label_from_file_path(file_path)
        documents.extend([Document(label, tag_list[x:x+1000]) for x in range(0, len(tag_list), 1000)])
    return documents


def get_label_from_file_path(file_path):
    label = re.search('(?<=-)(.+)(?=-)', file_path)
    if label is None:
        return file_path
    else:
        return label.group()


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
               feature_names[top_coefficients % len(feature_names)],
               rotation=60, ha='right')
    plt.show()


def features(tags_list):
    features = []
    for tags in tags_list:
        features.append(relative_function_word_feature(tags))
    return features


def relative_function_word_feature(tags):
    function_words = [tag.lemma for tag in tags if tag.pos in FUNCTION_WORD_POS]
    word_counts = dict(Counter(function_words))
    relative_word_counts = {word: word_counts[word] / len(function_words)
                            for word in word_counts.keys()}
    return relative_word_counts


def main():

    book_data_dict = load_data()
    documents = generate_documents(book_data_dict)

    text_classifier = Pipeline([
        ('vect', DictVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', SGDClassifier(random_state=42, penalty='l1'))
    ])

    labels = [doc.label for doc in documents]
    X = [doc.tag_list for doc in documents]
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.20)

    text_classifier.fit(features(X_train), y_train)
    predicted = text_classifier.predict(features(X_test))
    np.mean(predicted == y_test)
    print(metrics.classification_report(y_test, predicted))
    plot_coefficients(text_classifier.named_steps['clf'],
                      text_classifier.named_steps['vect'].get_feature_names())


if __name__ == "__main__":
    main()
