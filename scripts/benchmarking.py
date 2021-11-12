import time
import numpy as np
from sklearn.datasets import make_blobs
import pudding

def benchmarking_kmeans():
    '''
    Benchmarking kmeans CPU and kmeans GPU
    '''

    # Generate random data
    seed = 0
    np.random.seed(seed)
    n_examples = 1000000
    truth_centers = [[1, 1], [-1, -1], [1, -1]]
    X, _ = make_blobs(n_samples=n_examples, centers=truth_centers, cluster_std=0.7)

    
    # Our implementation CPU
    t_cpu_start = time.time()
    pudding.clustering.kmeans(X, n_clusters=3, cuda_enabled=False, rand_seed=seed)
    t_cpu_end = time.time()
    cpu_ellapse = t_cpu_end - t_cpu_start

    # Our implementation GPU
    t_gpu_start = time.time()
    pudding.clustering.kmeans(X, n_clusters=3, cuda_enabled=True, rand_seed=seed)
    t_gpu_end = time.time()
    gpu_ellapse = t_gpu_end - t_gpu_start

    # Summary
    print('=' * 50)
    print('Benchmarking summary of KMeans')
    print('=' * 50)
    print('Num of clusters: %d' % len(truth_centers))
    print('Num of data points: %d' % n_examples)
    print('=' * 50)
    print('CPU time: %fs' % (cpu_ellapse))
    print('GPU time: %fs' % (gpu_ellapse))
    print('Relative speed up: %f' %((cpu_ellapse - gpu_ellapse) / cpu_ellapse))

if __name__ == '__main__':
    benchmarking_kmeans()
