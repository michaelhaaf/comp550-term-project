import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from data_utilities import load_preprocessed_data
from document_handler import DocumentFactory
from features import FunctionWordFeature

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
        features.append(FunctionWordFeature().apply(tags))
    return features


def main():

    book_data_dict = load_preprocessed_data()
    documents = DocumentFactory().create_documents(book_data_dict)

    text_classifier = Pipeline([
        ('vect', DictVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', SGDClassifier(random_state=42, penalty='l1'))
    ])

    labels = [doc.label for doc in documents]
    X = [doc.tag_sequence for doc in documents]
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.20, random_state=42)

    text_classifier.fit(features(X_train), y_train)
    predicted = text_classifier.predict(features(X_test))

    print(metrics.classification_report(y_test, predicted))
    plot_coefficients(text_classifier.named_steps['clf'],
                      text_classifier.named_steps['vect'].get_feature_names())


if __name__ == "__main__":
    main()
