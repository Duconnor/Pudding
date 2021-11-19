# KMeans Clustering

KMeans is a widely used clustering algorithm. Generally speaking, it tries to cluster data by minimizing the within-cluster sum-of-squares. More information about this algorithm one can refer to this [site](https://scikit-learn.org/stable/modules/clustering.html#k-means).

# Benchmark Result

The GPU version of KMeans is benchmarked with respect to a naive CPU version implemented by me and a fully optimized CPU version implemented in scikit learn.

# Example Usage: Image Quantization

The example usage can be found in ```examples/clustering/kmeans```.

# Notes on Usage

There are also a few things that need careful consideration when using KMeans on your own dataset.

1. The choice of the number of clusters. Since KMeans requires the number of clusters to be specified, it is important to choose an appropriate number.
2. The choice of the initial cluster centers. This is also of vital importance to the KMeans algorithm. Pudding currently only supports random cluster center initialization, so if you find the outcome of the algorithm not satisfactory, you can either try to mannualy set the initial center using some prior knowledge about the specific dataset you are using or re-run the algorithm with the ```rand_seed``` parameter set to a different value. I may in the near future add more features like the support of the ```kmeans++``` algorithm.