from ..lib import _LIB, CONTIGUOUS_FLAG
import ctypes
import numpy as np

def kde_score(X, kernel, bandwidth, samples):
    '''
    This function fits a kernel density estimation model on the given training data X. And then perform kde query using samples. The return value is the probability of each sample.

    Inputs:
        - X: a list of training data, of shape (n_samples, n_features).
        - kernel: a str, specify which kernel to use. Options include ['gaussian'].
        - bandwidth: a float, the bandwidth to use.
        - samples: a list of samples, of shape (n_test_samples, n_features).

    Return: the probability of each sample in samples, of shape (n_test_samples,).
    '''

    # Prepare the data and perform pre-condition check
    valid_kernels = ['gaussian']
    assert kernel in valid_kernels
    assert bandwidth > 0.0

    np_X = np.array(X).astype(np.float32)
    np_samples = np.array(samples).astype(np.float32)

    n_samples, n_features = np_X.shape
    n_test_samples, _ = np_samples.shape

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
    c_kde_score(np_X, n_samples, n_features, kernel.encode(), bandwidth, np_samples, n_test_samples, np_scores)

    return np_scores.tolist()
