import time
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pudding

def benchmarking_kmeans():
    '''
    Benchmarking kmeans CPU and kmeans GPU
    '''

    # Generate random data
    seed = 0
    np.random.seed(seed)
    n_examples = 100000
    n_clusters = 128
    X, _ = make_blobs(n_samples=n_examples, centers=n_clusters, cluster_std=1)    

    
    # Our implementation CPU
    t_cpu_start = time.time()
    pudding.clustering.kmeans(X, n_clusters=n_clusters, cuda_enabled=False, rand_seed=seed)
    t_cpu_end = time.time()
    cpu_ellapse = t_cpu_end - t_cpu_start

    # Our implementation GPU
    t_gpu_start = time.time()
    _, membership, _ = pudding.clustering.kmeans(X, n_clusters=n_clusters, cuda_enabled=True, rand_seed=seed)
    t_gpu_end = time.time()
    gpu_ellapse = t_gpu_end - t_gpu_start

    # Sci-kit learn's implementation
    sklearn_kmeans = KMeans(n_clusters=n_clusters, init='random', n_init=1)
    t_sklearn_start = time.time()
    sklearn_kmeans.fit(X)
    t_sklearn_end = time.time()
    sklearn_ellapse = t_sklearn_end - t_sklearn_start

    # Summary
    print('=' * 50)
    print('Benchmarking summary of KMeans')
    print('=' * 50)
    print('Num of clusters: %d' % n_clusters)
    print('Num of data points: %d' % n_examples)
    print('=' * 50)
    print('Sklearn time: %fs' % (sklearn_ellapse))
    print('CPU time: %fs' % (cpu_ellapse))
    print('GPU time: %fs' % (gpu_ellapse))
    print('Speed up: %f' %(cpu_ellapse / gpu_ellapse))

    plt.scatter(X[:, 0], X[:, 1], c=membership)
    plt.savefig('cluster.png')

if __name__ == '__main__':
    benchmarking_kmeans()
