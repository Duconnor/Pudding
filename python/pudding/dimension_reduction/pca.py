'''
PCA related functions
'''

import ctypes
import numpy as np

from ..lib import _LIB, CONTIGUOUS_FLAG

def pca_reduction(X, n_components=None):
    '''
    This function performs PCA reduction. In other words, reduce the dimension of the given data.

    Inputs:
        - X: input data, be of shape (n_samples, n_features)
        - n_components: can be either a int, a float or None. If it is a int, must be in range (0, min(n_samples, n_features)], and n_components principal components (scores) will be kept. If it is a float, it represents the expected variance ratio, so must be in range (0, 1), and the number of components to keep will be determined automatically so that the expected variance ratio. If it is set to None, all components will be kept.

    Return: a tuple of (principal_components, principal_axes, variances)
    '''

    # Prepare the data and perform pre-condition check
    np_X = np.array(X).astype(np.float32)
    n_samples, n_features = np_X.shape
    variance_percentage = 0.0

    if isinstance(n_components, int):
        assert n_components > 0 and n_components <= min(n_samples, n_features)
    elif isinstance(n_components, float):
        assert n_components > 0 and n_components < 1
        variance_percentage = n_components
        n_components = -1
    else:
        n_components = min(n_samples, n_features)
    
    # Prepare the return values
    np_principal_components = np.empty(n_samples * min(n_samples, n_features)).astype(np.float32)
    np_principal_axes = np.empty(min(n_samples, n_features) * n_features).astype(np.float32)
    np_variance = np.empty(min(n_samples, n_features)).astype(np.float32)
    n_components_chosen = 0

    # Prepare the function being called
    c_pca_reduction = _LIB.pca
    c_pca_reduction.restype = None
    c_pca_reduction.argtypes = [
        np.ctypeslib.ndpointer(ctypes.c_float, flags=CONTIGUOUS_FLAG),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_float,
        np.ctypeslib.ndpointer(ctypes.c_float, flags=CONTIGUOUS_FLAG),
        np.ctypeslib.ndpointer(ctypes.c_float, flags=CONTIGUOUS_FLAG),
        np.ctypeslib.ndpointer(ctypes.c_float, flags=CONTIGUOUS_FLAG),
        ctypes.POINTER(ctypes.c_int),
    ]

    # Call the function
    c_n_components_chosen = ctypes.c_int(n_components_chosen)
    c_pca_reduction(np_X, n_samples, n_features, n_components, variance_percentage, np_principal_components, np_principal_axes, np_variance, ctypes.byref(c_n_components_chosen))
    n_components_chosen = c_n_components_chosen.value

    # Extract true return value using n_components_chosen
    np_principal_components = np_principal_components[:n_samples * n_components_chosen].reshape(n_samples, n_components_chosen)
    np_principal_axes = np_principal_axes[:n_components_chosen * n_features].reshape(n_components_chosen, n_features)
    np_variance = np_variance[:n_components_chosen]

    return (np_principal_components.tolist(), np_principal_axes.tolist(), np_variance.tolist())
