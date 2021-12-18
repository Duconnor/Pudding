import time
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from pudding.clustering import kmeans

def benchmarking_kmeans():
    '''
    Benchmarking kmeans CPU and kmeans GPU
    '''

    # Generate random data
    seed = 0
    np.random.seed(seed)
    n_clusters = 128

    times = {'Pudding': [], 'Scikit-learn': []}
    n_examples_list = [200, 1000, 2500, 5000, 7500, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000]

    for n_examples in n_examples_list:
        X, _ = make_blobs(n_samples=int(n_examples), n_features=3, centers=n_clusters, cluster_std=1)
        init_center = np.array(X[np.random.choice(int(n_examples), n_clusters, replace=False)]) 

        # Our implementation GPU
        t_gpu_start = time.time()
        _, membership, _ = kmeans(X, n_clusters=n_clusters, initial_centers=init_center, cuda_enabled=True, rand_seed=seed)
        t_gpu_end = time.time()
        gpu_ellapse = t_gpu_end - t_gpu_start
        times['Pudding'].append(gpu_ellapse)

        # Sci-kit learn's implementation
        sklearn_kmeans = KMeans(n_clusters=n_clusters, init=init_center, n_init=1)
        t_sklearn_start = time.time()
        sklearn_kmeans.fit(X)
        t_sklearn_end = time.time()
        sklearn_ellapse = t_sklearn_end - t_sklearn_start
        times['Scikit-learn'].append(sklearn_ellapse)

        # Summary
        print('=' * 50)
        print('Benchmarking summary of KMeans')
        print('=' * 50)
        print('Num of clusters: %d' % n_clusters)
        print('Num of data points: %d' % n_examples)
        print('=' * 50)
        print('Sklearn time: %fs' % (sklearn_ellapse))
        print('GPU time: %fs' % (gpu_ellapse))
        print('Speed up: %f' %(sklearn_ellapse / gpu_ellapse))
    
    plt.plot(n_examples_list[1:], times['Pudding'][1:], label='Pudding')
    plt.plot(n_examples_list[1:], times['Scikit-learn'][1:], label='Scikit-learn')
    plt.legend()
    plt.xlabel('Number of data points')
    plt.ylabel('Times (seconds)')
    plt.title('Execution time for KMeans on %d clusters' % n_clusters)
    plt.savefig('kmeans_benchmark_res.jpg')

if __name__ == '__main__':
    benchmarking_kmeans()
