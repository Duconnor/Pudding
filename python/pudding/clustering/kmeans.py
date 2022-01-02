'''
KMeans clustering related functions
'''

from ..lib import _LIB, CONTIGUOUS_FLAG
from ..base import _BaseModel
import ctypes
import numpy as np

class KMeans(_BaseModel):
    '''
    KMeans Clustering.

    Parameters
    ----------
    n_clusters: the number of clusters to form
    
    max_iter: maximum number of iterations
    
    tol: tolerance for determine coverage and end the algorithm
    
    cuda_enabled: whether to use the GPU version
    
    rand_seed: the random state

    Attributes
    ----------
    n_clusters: a int, the number of clusters

    centers: a 2D array, of shape (n_clusters, n_features)
    
    membership: a 1D array, of shape (n_samples,)

    n_iter: a int, the number of iterations actually performs

    Methods
    -------
    fit(X, y=None, **kwargs): perform the kmeans clustering on the given data X

    '''
    def __init__(self, n_clusters=8, max_iter=300, tol=1e-4, cuda_enabled=False, rand_seed=None):
        super().__init__()
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.cuda_enabled = cuda_enabled
        self.rand_seed = rand_seed
    
    def fit(self, X, y=None, **kwargs):
        '''
        Perform kmeans clustering on the given data

        Inputs:
            - X: numpy array, the input data, of shape (n_samples, n_features)
            - y: ignored, KMeans clustering is an unsupervised learning algorithm
            - initial_centers: optional, if None, the initial centers will be selected as several randomly sampled data points

        Return: None, the result cluster centroid and membership assignment can be obtained through accessing the corresponding attributes
        '''

        initial_centers = kwargs.pop('initial_centers', None)

        # Pre-condition check
        if initial_centers is not None and len(X) != 0:
            assert len(initial_centers) == self.n_clusters and len(initial_centers[0]) == len(X[0])

        # Set the random seed
        if self.rand_seed is not None:
            np.random.seed(self.rand_seed)

        # Prepare the data
        np_X = X.astype(np.float32)
        n_samples, n_features = np_X.shape

        # Prepare the initial centers
        if initial_centers is None:
            np_initial_centers = np.array(np_X[np.random.choice(n_samples, self.n_clusters, replace=False)]).astype(np.float32)
        else:
            np_initial_centers = np.array(initial_centers).astype(np.float32)
        
        # Prepare for the return value
        np_membership = np.empty(n_samples,).astype(np.int32)
        np_centers = np.empty((self.n_clusters, n_features)).astype(np.float32)
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
        c_kmeans(np_X, np_initial_centers, n_samples, n_features, self.n_clusters, self.max_iter, self.tol, self.cuda_enabled, np_centers, np_membership, ctypes.byref(c_n_iter))
        n_iter = c_n_iter.value

        self.centers = np_centers
        self.membership = np_membership
        self.n_iter = n_iter

    def predict(self, X, **kwargs):
        '''
        KMeans is an unsupervised learning algorithm, therefore the predict method will not be implemented
        '''
        raise NotImplementedError
