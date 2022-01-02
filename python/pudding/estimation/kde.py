from ..lib import _LIB, CONTIGUOUS_FLAG
from ..base import _BaseModel
import ctypes
import numpy as np

class KDE(_BaseModel):
    '''
    Kernel Density Estimation

    Parameters
    ----------
    kernel: a str, specify which kernel to use. Options include ['gaussian']

    bandwidth: a float, the bandwidth to use

    Method
    ----------
    fit(X, y=None, **kwargs): fit the model using the given data

    predict(X, **Kwargs): predict the probability of each sample given in X
    '''

    def __init__(self, kernel, bandwidth) -> None:
        super().__init__()
        self.kernel = kernel
        self.bandwidth = bandwidth
        self.__isfit = False
    
    def fit(self, X, y=None, **kwargs):
        '''
        This function fits a kernel density estimation model on the given training data X

        Inputs:
            - X: numpy array, the training data, of shape (n_samples, n_features)
            - y: ignored
        '''
        self.training_X = X.astype(np.float32)
        self.__isfit = True

    def predict(self, X, **kwargs):
        '''
        Perform the kde query on X

        Inputs:
            - X: numpy array, the samples, of shape (n_test_samples, n_features).
        
        Return: the probability of each sample in X, of shape (n_test_samples,).
        '''
        assert self.__isfit # Must fit before predicting anything

        # Prepare the data and perform pre-condition check
        valid_kernels = ['gaussian']
        assert self.kernel in valid_kernels
        assert self.bandwidth > 0.0

        np_X = X.astype(np.float32)

        n_samples, n_features = self.training_X.shape
        n_test_samples, _ = np_X.shape

        # Prepare for the return value
        np_scores = np.empty((n_test_samples)).astype(np.float32)

        # Prepare the function being called
        c_kde_score = _LIB.kdeScore
        c_kde_score.restype = None
        c_kde_score.argtypes = [
            np.ctypeslib.ndpointer(ctypes.c_float, flags=CONTIGUOUS_FLAG),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_float,
            np.ctypeslib.ndpointer(ctypes.c_float, flags=CONTIGUOUS_FLAG),
            ctypes.c_int,
            np.ctypeslib.ndpointer(ctypes.c_float, flags=CONTIGUOUS_FLAG),
        ]

        # Call the function
        c_kde_score(self.training_X, n_samples, n_features, self.kernel.encode(), self.bandwidth, np_X, n_test_samples, np_scores)

        return np_scores
