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