'''
This example demonstrates the usage of naive bayes with multinomial event model to perform text classification problem using the 20 newsgroups dataset provided by scikir-learn
'''

from sklearn.naive_bayes import MultinomialNB
from pudding.classification import NaiveBayesMultinomial

from sklearn import metrics
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

from time import time

# Prepare the data
newsgroups_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
vectorizer = TfidfVectorizer()
label_encoder = LabelEncoder()
X_train = vectorizer.fit_transform(newsgroups_train.data)
y_train = label_encoder.fit_transform(newsgroups_train.target)
X_train = X_train.todense()

newsgroups_test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'))
X_test = vectorizer.transform(newsgroups_test.data)
y_test = label_encoder.transform(newsgroups_test.target)
X_test = X_test.todense()

# Fit the model
print('Fitting and predicting using scikit-learn...')
t0 = time()
sklearn_model = MultinomialNB(alpha=0.01)
sklearn_model.fit(X_train, y_train)
sklearn_pred = sklearn_model.predict(X_test)
print(f'Done in {time() - t0:0.3f}s.')
sklearn_f1 = metrics.f1_score(y_test, sklearn_pred, average='macro')

print('Fitting and predicting using Pudding...')
t1 = time()
pudding_model = NaiveBayesMultinomial(n_classes=len(label_encoder.classes_), alpha=0.01)
pudding_model.fit(X_train, y_train)
pudding_pred = pudding_model.predict(X_test)
print(f'Done in {time() - t1:0.3f}s.')
pudding_f1 = metrics.f1_score(y_test, pudding_pred, average='macro')

print('Scikit-learn\'s F1 score: %0.3f' % sklearn_f1)
print('Pudding\'s F1 score: %0.3f' % pudding_f1)
