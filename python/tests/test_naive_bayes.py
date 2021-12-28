from attr import s
import numpy as np
import pytest

from sklearn.naive_bayes import MultinomialNB

from pudding.classification import NaiveBayesMultinomial


def test_naive_bayes_multinomial_toy_data():
    '''
    This test case tests Pudding's multinomial naive bayes using sklearn's implementation as reference on some toy data
    '''

    # Prepare for the test data
    rng = np.random.RandomState(1)
    n_classes = 6
    X = rng.randint(5, size=(6, 10))
    y = np.array([0, 1, 2, 3, 4, 5])
    X_test = np.array([[3, 4, 3, 1, 3, 0, 0, 2, 2, 1]])

    # Scikit-learn's result
    sklearn_naive_bayes_multinomial = MultinomialNB()
    sklearn_naive_bayes_multinomial.fit(X, y)
    sklearn_class_prob = np.exp(sklearn_naive_bayes_multinomial.class_log_prior_)
    sklearn_word_prob = np.exp(sklearn_naive_bayes_multinomial.feature_log_prob_)
    sklearn_prediction = sklearn_naive_bayes_multinomial.predict(X_test)

    # Pudding's result
    pudding_naive_bayes_multinomial = NaiveBayesMultinomial(n_classes)
    pudding_naive_bayes_multinomial.fit(X, y)
    pudding_class_prob = pudding_naive_bayes_multinomial.class_prob
    pudding_word_prob = pudding_naive_bayes_multinomial.word_prob
    pudding_prediction = np.array(pudding_naive_bayes_multinomial.predict(X_test))

    # # Check
    assert np.array_equal(sklearn_prediction, pudding_prediction)
    assert np.allclose(sklearn_class_prob, pudding_class_prob)
    assert np.allclose(sklearn_word_prob, pudding_word_prob)

@pytest.mark.parametrize('n_samples', [100, 1000, 5000])
@pytest.mark.parametrize('n_vocabulary', [10, 100, 500])
@pytest.mark.parametrize('alpha', [0.1, 0.5, 1.0])
def test_naive_bayes_multinomial_random_data(n_samples, n_vocabulary, alpha):
    '''
    This test case tests Pudding's multinomial naive bayes using sklearn's implementation as reference on some randomly generated data
    '''

    # Prepare for the test data
    rng = np.random.RandomState(1)
    n_classes = 10
    n_test_samples = 50
    X = rng.randint(20, size=(n_samples, n_vocabulary))
    y = rng.randint(n_classes, size=(n_samples,))
    X_test = rng.randint(20, size=(n_test_samples, n_vocabulary))

    # Scikit-learn's result
    sklearn_naive_bayes_multinomial = MultinomialNB(alpha=alpha)
    sklearn_naive_bayes_multinomial.fit(X, y)
    sklearn_class_prob = np.exp(sklearn_naive_bayes_multinomial.class_log_prior_)
    sklearn_word_prob = np.exp(sklearn_naive_bayes_multinomial.feature_log_prob_)
    sklearn_prediction = sklearn_naive_bayes_multinomial.predict(X_test)

    # Pudding's result
    pudding_naive_bayes_multinomial = NaiveBayesMultinomial(n_classes, alpha=alpha)
    pudding_naive_bayes_multinomial.fit(X, y)
    pudding_class_prob = pudding_naive_bayes_multinomial.class_prob
    pudding_word_prob = pudding_naive_bayes_multinomial.word_prob
    pudding_prediction = np.array(pudding_naive_bayes_multinomial.predict(X_test))

    # Check
    print(pudding_class_prob)
    assert np.allclose(sklearn_class_prob, pudding_class_prob)
    assert np.allclose(sklearn_word_prob, pudding_word_prob)
    print(sklearn_naive_bayes_multinomial.predict_log_proba(X_test))
    print(sklearn_prediction)
    print(pudding_prediction)
    assert np.array_equal(sklearn_prediction, pudding_prediction)
