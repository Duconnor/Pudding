'''
This example script is based on Shankar Muthuswamy's blog about PCA
https://shankarmsy.github.io/posts/pca-sklearn.html
'''

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_olivetti_faces

from pudding.dimension_reduction import PCA

# First, load the dataset and visualize the faces
oliv = fetch_olivetti_faces(data_home='.')
fig = plt.figure(figsize=(6, 6))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
for i in range(64):
    ax = fig.add_subplot(8, 8, i + 1, xticks=[], yticks=[])
    ax.imshow(oliv.images[i], cmap=plt.cm.bone, interpolation='nearest') 

plt.savefig('face_preview.jpg')

# Project the original face dataset into a lower dimensional feature space
# Let's say, from 4096 -> 64
pca = PCA(n_components=64)
pca.fit(oliv.data)
principal_components, principal_axes, reconstructed_X = pca.principal_components, pca.principal_axes, pca.reconstructed_X

# Visualize the top 9 principal axes
fig = plt.figure(figsize=(6, 6))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
for i in range(9):
    ax = fig.add_subplot(3, 3, i + 1, xticks=[], yticks=[])
    ax.imshow(np.reshape(np.array(principal_axes)[i, :], (64, 64)), cmap=plt.cm.bone, interpolation='nearest') 

fig.savefig('principal_axes.jpg')

# Display the recontruction of the original face image from the 64-dim lower representation
reconstructed_img = np.reshape(reconstructed_X,(400,64,64))
fig = plt.figure(figsize=(6, 6))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
for i in range(64):
    ax = fig.add_subplot(8, 8, i + 1, xticks=[], yticks=[])
    ax.imshow(reconstructed_img[i], cmap=plt.cm.bone, interpolation='nearest') 

fig.savefig('reconstructed_images.jpg')
