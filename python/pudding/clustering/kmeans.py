'''
KMeans clustering related functions
'''

from scipy.sparse.construct import rand
from ..lib import _LIB
import ctypes
import numpy as np

CONTIGUOUS_FLAG = 'C_CONTIGUOUS'

def kmeans(X, initial_centers=None, n_clusters=8, max_iter=300, tol=1e-4, cuda_enabled=False, rand_seed=0) -> tuple:
    '''
    Perform kmeans clustering on the given data

    Inputs:
        - X: the input data, of shape (n_samples, n_features)
        - initial_centers: optional, if None, the initial centers will be selected as several randomly sampled data points
        - n_clusters: the number of clusters to form
        - max_iter: maximum number of iterations
        - tol: tolerance for determine coverage and end the algorithm
        - cuda_enabled: whether to use the GPU version
        - rand_seed: the random state

    Return: a tuple of (centers, membership, n_iter)
        - centers: a 2D array, of shape (n_clusters, n_features)
        - membership: a 1D array, of shape (n_samples,)
        - n_iter: a int, the number of iterations actually performs
    '''

    # Pre-condition check
    if initial_centers is not None and len(X) != 0:
        assert len(initial_centers) == n_clusters and len(initial_centers[0]) == len(X[0])

    # Set the random seed
    np.random.seed(rand_seed)

    # Prepare the data
    np_X = np.array(X).astype(np.float32)
    n_samples, n_features = np_X.shape

    # Prepare the initial centers
    if initial_centers is None:
        np_initial_centers = np.array(np_X[np.random.choice(n_samples, n_clusters, replace=False)]).astype(np.float32)
    else:
        np_initial_centers = np.array(initial_centers).astype(np.float32)
    
    # Prepare for the return value
    np_membership = np.empty(n_samples,).astype(np.int32)
    np_centers = np.empty((n_clusters, n_features)).astype(np.float32)
    n_iter = 0

    # Prepare the function being called
    c_kmeans = _LIB.kmeans
    c_kmeans.restype = None
    c_kmeans.argtypes = [
        np.ctypeslib.ndpointer(ctypes.c_float, flags=CONTIGUOUS_FLAG),
        np.ctypeslib.ndpointer(ctypes.c_float, flags=CONTIGUOUS_FLAG),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_float,
        ctypes.c_bool,
        np.ctypeslib.ndpointer(ctypes.c_float, flags=CONTIGUOUS_FLAG),
        np.ctypeslib.ndpointer(ctypes.c_int, flags=CONTIGUOUS_FLAG),
        ctypes.POINTER(ctypes.c_int),
    ]

    # Call the function
    c_n_iter = ctypes.c_int(n_iter)
    c_kmeans(np_X, np_initial_centers, n_samples, n_features, n_clusters, max_iter, tol, cuda_enabled, np_centers, np_membership, ctypes.byref(c_n_iter))
    n_iter = c_n_iter.value

    return (np_centers.tolist(), np_membership.tolist(), n_iter)