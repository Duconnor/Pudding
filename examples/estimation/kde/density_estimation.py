'''
This example demonstrates how to use kernel density estimation to estimate the underlying density of a set of i.i.d 1D samples.

The code is adoped and modified from https://scikit-learn.org/stable/auto_examples/neighbors/plot_kde_1d.html#sphx-glr-auto-examples-neighbors-plot-kde-1d-py.
'''

import numpy as np
import matplotlib.pyplot as plt

from sklearn.neighbors import KernelDensity
from scipy.stats import norm

from pudding.estimation import KDE

N = 100
np.random.seed(1)
X = np.concatenate(
    (np.random.normal(0, 1, int(0.3 * N)), np.random.normal(5, 1, int(0.7 * N)))
)[:, np.newaxis]

X_plot = np.linspace(-5, 10, 1000)[:, np.newaxis]
true_dens = 0.3 * norm(0, 1).pdf(X_plot[:, 0]) + 0.7 * norm(5, 1).pdf(X_plot[:, 0])

sklearn_kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(X)
sklearn_log_density = sklearn_kde.score_samples(X_plot)

pudding_kde = KDE(kernel='gaussian', bandwidth=0.5)
pudding_kde.fit(X)
pudding_density = pudding_kde.predict(X_plot)

fig, ax = plt.subplots(1, 2, sharey=True, figsize=(9, 4))
fig.subplots_adjust(wspace=0)

for i, (density, method_name) in enumerate(zip([pudding_density, np.exp(sklearn_log_density)], ['Pudding', 'Scikit-learn'])):
    ax[i].fill(X_plot[:, 0], true_dens, fc="black", alpha=0.2)
    ax[i].plot(X_plot[:, 0], density, color='navy', lw=2, linestyle="-")
    ax[i].set_title(method_name)

plt.savefig('density_estimation.jpg')
