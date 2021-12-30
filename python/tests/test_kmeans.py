from pudding.clustering import kmeans
import pytest
import numpy as np
from sklearn.datasets import make_blobs
import pudding

def testKmeansToyData():
    '''
    Test KMeans uisng a toy dataset
    '''
    X = [[0.0, 0.0], [0.5, 0.0], [0.5, 1.0], [1.0, 1.0]]
    initial_centers = [[0.0, 0.0], [1.0, 1.0]]

    expected_membership = [0, 0, 1, 1]
    expected_centers = [[0.25, 0.0], [0.75, 1.0]]
    expected_iterations = 2

    pudding_kmeans = pudding.clustering.KMeans(n_clusters=len(initial_centers), cuda_enabled=False)
    pudding_kmeans.fit(X, initial_centers=initial_centers)

    assert pudding_kmeans.membership == expected_membership
    for center, expected_center in zip(pudding_kmeans.centers, expected_centers):
        assert center == pytest.approx(expected_center)
    assert expected_iterations == pudding_kmeans.n_iter

    pudding_kmeans = pudding.clustering.KMeans(n_clusters=len(initial_centers), cuda_enabled=True)
    pudding_kmeans.fit(X, initial_centers=initial_centers)

    assert pudding_kmeans.membership == expected_membership
    for center, expected_center in zip(pudding_kmeans.centers, expected_centers):
        assert center == pytest.approx(expected_center)
    assert expected_iterations == pudding_kmeans.n_iter

def testKmeansCPUGPU():
    '''
    Test our implementation with some randomly generated data between the CPU and GPU version
    '''

    # Generate random data
    seed = 0
    np.random.seed(seed)
    n_examples = 3000
    truth_centers = [[1, 1], [-1, -1], [1, -1]]
    X, _ = make_blobs(n_samples=n_examples, centers=truth_centers, cluster_std=0.7)

    
    # Our implementation CPU
    cpu_kmeans = pudding.clustering.KMeans(n_clusters=3, cuda_enabled=False, rand_seed=seed)
    cpu_kmeans.fit(X)
    our_cpu_centers, our_cpu_membership, our_cpu_n_iter = cpu_kmeans.centers, cpu_kmeans.membership, cpu_kmeans.n_iter

    # Our implementation GPU
    gpu_kmeans = pudding.clustering.KMeans(n_clusters=3, cuda_enabled=True, rand_seed=seed)
    gpu_kmeans.fit(X)
    our_gpu_centers, our_gpu_membership, our_gpu_n_iter = gpu_kmeans.centers, gpu_kmeans.membership, gpu_kmeans.n_iter

    # Assertions
    assert our_cpu_membership == our_gpu_membership

    for our_cpu_center, our_gpu_center in zip(our_cpu_centers, our_gpu_centers):
        assert our_cpu_center == pytest.approx(our_gpu_center)

    assert our_cpu_n_iter == our_gpu_n_iter

def testKmeansCPUGPULarge():
    '''
    Test our implementation with some randomly generated data between the CPU and GPU version using a larger number of data pooints
    '''

    # Generate random data
    seed = 0
    np.random.seed(seed)
    n_examples = 1000000
    truth_centers = [[1, 1], [-1, -1], [1, -1]]
    X, _ = make_blobs(n_samples=n_examples, centers=truth_centers, cluster_std=0.7)

    
    # Our implementation CPU
    cpu_kmeans = pudding.clustering.KMeans(n_clusters=3, cuda_enabled=False, rand_seed=seed)
    cpu_kmeans.fit(X)
    our_cpu_centers = cpu_kmeans.centers

    # Our implementation GPU
    gpu_kmeans = pudding.clustering.KMeans(n_clusters=3, cuda_enabled=True, rand_seed=seed)
    gpu_kmeans.fit(X)
    our_gpu_centers = gpu_kmeans.centers

    # Assertions
    for our_cpu_center, our_gpu_center in zip(our_cpu_centers, our_gpu_centers):
        assert our_cpu_center == pytest.approx(our_gpu_center, rel=1e-1)

def testKmeansEmptyCluster():
    '''
    Test KMeans when there is empty cluster
    '''
    X = [[0.0, 0.0], [0.5, 0.0], [0.5, 1.0], [1.0, 1.0]]
    initial_centers = [[0.0, 0.0], [10.0, 10.0]]

    expected_membership = [0, 0, 0, 0]
    expected_centers = [[0.5, 0.5], [10.0, 10.0]]
    expected_iterations = 2

    cpu_kmeans = pudding.clustering.KMeans(n_clusters=len(initial_centers), cuda_enabled=False)
    cpu_kmeans.fit(X, initial_centers=initial_centers)
    centers, membership, n_iterations = cpu_kmeans.centers, cpu_kmeans.membership, cpu_kmeans.n_iter

    assert membership == expected_membership
    for center, expected_center in zip(centers, expected_centers):
        assert center == pytest.approx(expected_center)
    assert expected_iterations == n_iterations

    gpu_kmeans = pudding.clustering.KMeans(n_clusters=len(initial_centers), cuda_enabled=True)
    gpu_kmeans.fit(X, initial_centers=initial_centers)
    centers, membership, n_iterations = gpu_kmeans.centers, gpu_kmeans.membership, gpu_kmeans.n_iter

    assert membership == expected_membership
    for center, expected_center in zip(centers, expected_centers):
        assert center == pytest.approx(expected_center)
    assert expected_iterations == n_iterations