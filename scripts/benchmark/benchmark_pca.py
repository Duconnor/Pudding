'''
This benchmark script was inspired by https://github.com/nmerrill67/GPU_GSPCA/blob/master/demo.py
'''

import numpy as np
import matplotlib.pyplot as plt
from time import time

from sklearn.decomposition import PCA
import pudding

def benchmark_random():
    '''
    This function benchmark our implementation with Scikit-learn's fully optimized PCA using a randomly generated data matrix
    '''

    n_samples = [500, 1000, 5000, 10000, 30000]

    times_sklearn_num_samples = []
    times_pudding_num_samples = []

    # Benchmar as the number samples grows
    for n_sample in n_samples:
        n_feature = 500
        n_components = 500

        X = np.random.rand(n_sample, n_feature)

        t_sklearn_start = time()
        # PCA in pudding performs both fit, transform and inverse transform all at the same time so in order to have a fair comparision, we perform these operations here too
        sklearn_pca = PCA(n_components=n_components)
        new_X = sklearn_pca.fit_transform(X)
        sklearn_pca.inverse_transform(new_X)
        times_sklearn_num_samples.append(time() - t_sklearn_start)

        t_pudding_start = time()
        pudding_pca = pudding.dimension_reduction.PCA(n_components=n_components)
        pudding_pca.fit(X)
        times_pudding_num_samples.append(time() - t_pudding_start)
    
    plt.figure()
    plt.plot(n_samples[1:], times_pudding_num_samples[1:], label='Pudding')
    plt.plot(n_samples[1:], times_sklearn_num_samples[1:], label='Scikit-learn')
    plt.legend()
    plt.xlabel('Number of data points')
    plt.ylabel('Times (seconds)')
    plt.title('Execution time for PCA on %d dimensional data' % n_feature)
    plt.savefig('pca_benchmark_num_samples.jpg')
    

if __name__ == '__main__':
    benchmark_random()
