---
layout: default
title: KMeans Clustering
parent: Unsupervised Learning Algorithms
grand_parent: Current Supported Algorithms
nav_order: 1
---
# KMeans Clustering

KMeans is a simple and widely used clustering algorithm. Generally speaking, it tries to cluster data by minimizing the within-cluster sum-of-squares error. For more information about this algorithm, one can refer to this [site](https://scikit-learn.org/stable/modules/clustering.html#k-means).

# Benchmark Result

The GPU version of KMeans is benchmarked with respect to the fully optimized CPU version implemented in Scikit learn. You can perform the benchmark simply by running the Python script ```scripts/benchmark/benchmark_kmeans.py```.

Specifically, I randonly sample points in a 3D space using the ```make_blobs``` function provided in the ```sklearn.datasets``` package. The number of samples ranges from 1000 to 100000 while the number of clusters formed is fixed to 128. The results are summarized in the following table:

|Number of Samples|Sci-kit Run Time (sec)|GPU Run Time (sec)|Speedup|
|:-:|:-:|:-:|:-:|
|1000|0.224|0.050|4.482x|
|10000|2.859|0.174|16.420x|
|30000|2.762|0.230|11.998x|
|50000|12.199|0.870|14.021x|
|100000|10.059|0.750|13.404x|

As we can see, compared to a fully optimized CPU version like the implementation in scikit-learn, the GPU version is about 14 times faster in average as the dataset gets larger.

<div align=center>
<img src="../../../assets/kmeans_benchmark_res.jpg">
<center style="font-size:14px;color:#C0C0C0;">Benchmark result.</center> 
</div>

# One Application: Image Quantization

One interesting application of the KMeans algorithm is image quantization. The task is simple, given an image that has many different colors in it, we perform KMeans clustering on the pixels in the image and discover a small set of clusters (e.g. 64). Then for every pixel, we use its corresponding cluster center to replace its original value. By dong so, we are able to use a much smaller number of colors to represent an image.

The code corresponds to this example can be found in ```examples/clustering/kmeans/image_quantization.py```. Specifically, we perform KMeans clustering on all pixles in a 1080p high resolution image. For the exactly same task, **KMeans in scikit-learn uses 47 seconds while our implementation only uses 17 seconds**. Check out the code and run the example yourself!

```python
'''
This script is based on scikit-learn's official example of image quantization.
https://scikit-learn.org/stable/auto_examples/cluster/plot_color_quantization.html#sphx-glr-auto-examples-cluster-plot-color-quantization-py
'''

import numpy as np
from PIL import Image
from time import time
from sklearn.cluster import KMeans

import pudding

n_colors = 64

# Load the sample photo
image = Image.open('image.jpg')

# Dividing by 255 so that the value is in the range [0-1]
image = np.array(image, dtype=np.float64) / 255

# Transform the image to a 2D numpy array.
w, h, d = original_shape = tuple(image.shape)
assert d == 3
image_array = np.reshape(image, (w * h, d))

print('Perform the KMeans clustering using scit-kit learn...')
t0 = time()
scikit_kmeans = KMeans(n_clusters=n_colors, n_init=1)
scikit_kmeans.fit(image_array)
print(f'Done in {time() - t0:0.3f}s.')

print("Perform the KMeans clustering using Pudding on GPU...")
t1 = time()
pudding_kmeans = pudding.clustering.KMeans(n_clusters=n_colors, cuda_enabled=True)
pudding_kmeans.fit(image_array)
centers, membership = pudding_kmeans.centers, pudding_kmeans.membership
assert not np.isnan(centers).any()
print(f"Done in {time() - t1:0.3f}s.")

def recreate_image(codebook, labels, w, h):
    """Recreate the (compressed) image from the code book & labels"""
    return codebook[labels].reshape(w, h, -1)

# Save the quantized image
np_image = recreate_image(np.array(centers), np.array(membership), w, h)
formatted_image = (np_image * 255).astype(np.uint8)
quantized_image = Image.fromarray(formatted_image)
quantized_image.save('quantized_pudding.jpg')

np_image = recreate_image(scikit_kmeans.cluster_centers_, scikit_kmeans.labels_, w, h)
formatted_image = (np_image * 255).astype(np.uint8)
quantized_image = Image.fromarray(formatted_image)
quantized_image.save('quantized_sklearn.jpg')
```

<div align=center>
<img src="../../../assets/image.jpg">
<center style="font-size:14px;color:#C0C0C0;">The original image.</center> 
</div>

<div align=center>
<img src="../../../assets/quantized_sklearn.jpg">
<center style="font-size:14px;color:#C0C0C0;">Scikit-learn's result.</center>
</div> 

<div align=center>
<img src="../../../assets/quantized_pudding.jpg">
<center style="font-size:14px;color:#C0C0C0;">Pudding's result.</center>
</div> 


# Notes on Usage

There are a few things that need careful consideration when using KMeans on your own dataset.

1. The choice of the number of clusters. Since KMeans requires the number of clusters to be specified, it is important to choose an appropriate number.
2. The choice of the initial cluster centers. This is also of vital importance to the KMeans algorithm. Pudding currently only supports random cluster center initialization, so if you find the outcome of the algorithm not satisfactory, you can either try to mannualy set the initial center using some prior knowledge about the specific dataset you are using or re-run the algorithm with the ```rand_seed``` parameter set to a different value. I may in the near future add more features like the support of the ```kmeans++``` algorithm.
3. The constraint on the dimension. "Curse of dimension" is a well-known problem for KMeans clustering since it uses Euclidean distance, which often fails when dimension becomes high. If the dimension of your data is rather high (e.g. >= 10), you probably need to use some dimensional reduction techniques like PCA to first project the data in a lower dimensional space before you perform KMeans on them.
