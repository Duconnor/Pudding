from ..lib import _LIB, CONTIGUOUS_FLAG
from ..base import _BaseModel
import ctypes
import numpy as np

class _NaiveBayes(_BaseModel):
    '''
    This is a class for general Naive Bayes model
    '''

    def __init__(self, n_classes, alpha=1.0):
        super().__init__()
        self.n_classes = n_classes
        self.alpha = alpha
        self.__isfit = False

class NaiveBayesMultinomial(_NaiveBayes):
    '''
    Naive Bayes using the multinomial event model.

    Parameters
    ----------
    n_classes: the number of classes

    alpha: the ratio of the Laplace smoothing, default is 1.0

    Attributes
    ----------
    n_classes: the number of classes

    n_vocabulary: the size of the vocabulary used

    class_prob: the fitted class probability, of shape (n_classes)

    word_prob: the fitted word probability, of shape (n_classes, n_vocabulary)

    Methods
    -------
    fit(X, y=None, **kwargs): fit the Naive Bayes model using the training set.
    
    predict(X, **kwargs): make predictions using the fitted model.
    '''

    def __init__(self, n_classes, alpha=1.0):
        super().__init__(n_classes=n_classes, alpha=alpha)

    def fit(self, X, y=None, **kwargs):
        '''
        Fit the Naive Bayes model using the training set.

        Params:
            -X: numpy array, of shape (n_samples, n_vocabulary), and each elements is the count of words occuring.
            -y: numpy array, must not be None and is of shape (n_samples,), which is the labels for training samples. Must be in the range of [0, n_classes - 1]
        
        Return: None
        '''
        assert y is not None # This is a supervised learning algorithm, so we need labels

        # Prepare the data
        np_X = X.astype(np.float32)
        np_y = y.astype(np.int32)
        n_samples, self.n_vocabulary = np_X.shape

        # Prepare for the return value
        np_class_prob = np.empty(self.n_classes).astype(np.float32)
        np_word_prob = np.empty((self.n_classes, self.n_vocabulary)).astype(np.float32)

        # Prepare the function being called
        c_naive_bayes_multinomial_fit = _LIB.naiveBayesMultinomialFit
        c_naive_bayes_multinomial_fit.restype = None
        c_naive_bayes_multinomial_fit.argtypes = [
            np.ctypeslib.ndpointer(ctypes.c_float, flags=CONTIGUOUS_FLAG),
            np.ctypeslib.ndpointer(ctypes.c_int, flags=CONTIGUOUS_FLAG),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_float,
            np.ctypeslib.ndpointer(ctypes.c_float, flags=CONTIGUOUS_FLAG),
            np.ctypeslib.ndpointer(ctypes.c_float, flags=CONTIGUOUS_FLAG),
        ]

        # Call the function
        c_naive_bayes_multinomial_fit(np_X, np_y, n_samples, self.n_vocabulary, self.n_classes, self.alpha, np_class_prob, np_word_prob)

        # Set the corresponding attributes
        self.class_prob = np_class_prob
        self.word_prob = np_word_prob
        self.__isfit = True

    def predict(self, X, **kwargs):
        '''
        Make predictions using the fitted model.
        
        Params:
            -X: numpy array, of shape (n_test_samples, n_vocabulary) and each element is the count of words occuring.
            
        Return: the predicted label is returned, which is of shape (n_test_samples,)
        '''
        assert self.__isfit # The model must first be fitted

        # Prepare the data
        np_X = X.astype(np.float32)
        n_test_samples, _ = np_X.shape

        # Prepare the return value
        np_prediction = np.empty(n_test_samples).astype(np.int32)

        # Prepare the function being called
        c_naive_bayes_multinomial_predict = _LIB.naiveBayesMultinomialPredict
        c_naive_bayes_multinomial_predict.restype = None
        c_naive_bayes_multinomial_predict.argtypes = [
            np.ctypeslib.ndpointer(ctypes.c_float, flags=CONTIGUOUS_FLAG),
            np.ctypeslib.ndpointer(ctypes.c_float, flags=CONTIGUOUS_FLAG),
            np.ctypeslib.ndpointer(ctypes.c_float, flags=CONTIGUOUS_FLAG),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            np.ctypeslib.ndpointer(ctypes.c_int, flags=CONTIGUOUS_FLAG),
        ]

        # Call the function
        c_naive_bayes_multinomial_predict(np_X, self.class_prob, self.word_prob, n_test_samples, self.n_vocabulary, self.n_classes, np_prediction)

        # Return the value
        return np_prediction
