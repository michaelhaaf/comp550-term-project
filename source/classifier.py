import glob
import pickle
import os
import re
import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


def load_data():
    file_paths = [f for f in glob.glob("../data/*.txt")]
    data = {}
    for path in file_paths:
        file = open(path , "rb")
        data[os.path.basename(path)] = pickle.load(file)
        file.close()
    return data


# this is terrifying code, there ought to be a better way to split these lists into 500 word segments
def segment_data(input_data_dict):
    labels, processed_data = [], []
    for file_path, tag_list in input_data_dict.items():
        label = get_label_from_file_path(file_path)
        string = ''
        for i, tag in enumerate(tag_list):
            string += tag.lemma + ' '
            if i % 500 == 0:
                labels.append(label)
                processed_data.append(string)
                string = ''
        labels.append(label)
        processed_data.append(string)
    return labels, processed_data


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


if __name__ == "__main__":

    book_data_dict = load_data()
    labels, processed_data = segment_data(book_data_dict)

    text_classifier = Pipeline([
        ('vect', CountVectorizer(ngram_range=(1, 2))),
        ('tfidf', TfidfTransformer()),
        ('clf', SGDClassifier(random_state=42, penalty='l1'))
    ])

    X_train, X_test, y_train, y_test = train_test_split(processed_data, labels, test_size=0.20)

    text_classifier.fit(X_train, y_train)
    predicted = text_classifier.predict(X_test)
    np.mean(predicted == y_test)
    print(metrics.classification_report(y_test, predicted))
    plot_coefficients(text_classifier.named_steps['clf'],
                      text_classifier.named_steps['vect'].get_feature_names())

