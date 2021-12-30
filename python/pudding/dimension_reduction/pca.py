'''
PCA related functions
'''

import ctypes
import numpy as np

from ..lib import _LIB, CONTIGUOUS_FLAG
from ..base import _BaseModel

class PCA(_BaseModel):
    '''
    Principal Component Analysis

    Parameters
    ----------
    n_components: can be either a int, a float or None. If it is a int, must be in range (0, min(n_samples, n_features)], and n_components principal components (scores) will be kept. If it is a float, it represents the expected variance ratio, so must be in range (0, 1), and the number of components to keep will be determined automatically so that the expected variance ratio. If it is set to None, all components will be kept.

    Attributes
    ----------
    principal_components: the principal components, which is the projected X (i.e. the result of the dimensionality reduction)

    principal_axes: the principal axes, which is the direction of the maximum variance (i.e. the eigenvector of the covariance matrix)

    variances: the variances along each principal axes (i.e. the eigenvalue of the covariance matrix)

    reconstructed_X: the reconstruction of the original X using the (possibly lower dimensional) principal components

    Methods
    -------
    fit(X, y=None, **kwargs): perform the PCA on the given data X
    '''

    def __init__(self, n_components):
        super().__init__()
        self.n_components = n_components
    
    def fit(self, X, y=None, **kwargs):
        '''
        Perform the PCA reduction. In other words, reduce the dimension of the given data.
        Note that, this function also performs reconstruction using the obtained principal components.

        Inputs:
            - X: input data, be of shape (n_samples, n_features)
            - y: will be ignored
            - n_components: can be either a int, a float or None. If it is a int, must be in range (0, min(n_samples, n_features)], and n_components principal components (scores) will be kept. If it is a float, it represents the expected variance ratio, so must be in range (0, 1), and the number of components to keep will be determined automatically so that the expected variance ratio. If it is set to None, all components will be kept.
        '''

        # Prepare the data and perform pre-condition check
        np_X = np.array(X).astype(np.float32)
        n_samples, n_features = np_X.shape
        variance_percentage = 0.0

        if isinstance(self.n_components, int):
            assert self.n_components > 0 and self.n_components <= min(n_samples, n_features)
        elif isinstance(self.n_components, float):
            assert self.n_components > 0 and self.n_components < 1
            variance_percentage = self.n_components
            self.n_components = -1
        else:
            self.n_components = min(n_samples, n_features)
        
        # Prepare the return values
        np_principal_components = np.empty(n_samples * min(n_samples, n_features)).astype(np.float32)
        np_principal_axes = np.empty(min(n_samples, n_features) * n_features).astype(np.float32)
        np_variance = np.empty(min(n_samples, n_features)).astype(np.float32)
        np_reconstructed_X = np.empty((n_samples, n_features)).astype(np.float32)
        n_components_chosen = 0

        # Prepare the function being called
        c_pca = _LIB.pca
        c_pca.restype = None
        c_pca.argtypes = [
            np.ctypeslib.ndpointer(ctypes.c_float, flags=CONTIGUOUS_FLAG),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_float,
            np.ctypeslib.ndpointer(ctypes.c_float, flags=CONTIGUOUS_FLAG),
            np.ctypeslib.ndpointer(ctypes.c_float, flags=CONTIGUOUS_FLAG),
            np.ctypeslib.ndpointer(ctypes.c_float, flags=CONTIGUOUS_FLAG),
            np.ctypeslib.ndpointer(ctypes.c_float, flags=CONTIGUOUS_FLAG),
            ctypes.POINTER(ctypes.c_int),
        ]

        # Call the function
        c_n_components_chosen = ctypes.c_int(n_components_chosen)
        c_pca(np_X, n_samples, n_features, self.n_components, variance_percentage, np_principal_components, np_principal_axes, np_variance, np_reconstructed_X, ctypes.byref(c_n_components_chosen))
        n_components_chosen = c_n_components_chosen.value

        # Extract true return value using n_components_chosen
        np_principal_components = np_principal_components[:n_samples * n_components_chosen].reshape(n_samples, n_components_chosen)
        np_principal_axes = np_principal_axes[:n_components_chosen * n_features].reshape(n_components_chosen, n_features)
        np_variance = np_variance[:n_components_chosen]

        self.principal_components = np_principal_components.tolist()
        self.principal_axes = np_principal_axes.tolist()
        self.variance = np_variance.tolist()
        self.reconstructed_X = np_reconstructed_X.tolist()
    
    def predict(self, X, **kwargs):
        '''
        PCA is an unsupervised learning algorithm, therefore, no predict method will be implemented
        '''
        raise NotImplementedError
