import glob
import pickle
import os
import re
import numpy as np

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
    
    file_paths = [f for f in glob.glob("../data/secret_test_set/*.txt")]
    secret_test_data = {}
    for path in file_paths:
        file = open(path , "rb")
        secret_test_data[os.path.basename(path)] = pickle.load(file)
        file.close()
    
    return data, secret_test_data


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


if __name__ == "__main__":

    book_data_dict, secret_test_set = load_data()
    labels, processed_data = segment_data(book_data_dict)
    secret_test_labels, secret_test_processed_data = segment_data(secret_test_set)

    text_classifier = Pipeline([
        ('vect', CountVectorizer(ngram_range=(1, 2))),
        ('tfidf', TfidfTransformer()),
        ('clf', SGDClassifier(loss='hinge', penalty='l2')) # see https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html
    ])

    X_train, X_test, y_train, y_test = train_test_split(processed_data, labels, test_size=0.20)
    
    print(X_test)
    
    text_classifier.fit(X_train, y_train)
    
    predicted = text_classifier.predict(X_test)
    np.mean(predicted == y_test)
    print(metrics.classification_report(y_test, predicted))
    
    secret_test_predicted = text_classifier.predict(secret_test_processed_data)
    np.mean(secret_test_predicted == secret_test_labels)
    print(metrics.classification_report(secret_test_labels, secret_test_predicted))

