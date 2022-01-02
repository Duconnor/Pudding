'''
This benchmark scripts is adopted from https://jakevdp.github.io/blog/2013/12/01/kernel-density-estimation/
'''

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from time import time

from sklearn.neighbors import KernelDensity
from pudding.estimation import KDE

def kde_sklearn(x, x_grid, bandwidth=0.2, **kwargs):
    '''Kernel Density Estimation with Scikit-learn'''
    kde_skl = KernelDensity(bandwidth=bandwidth, **kwargs)
    kde_skl.fit(x[:, np.newaxis])
    # score_samples() returns the log-likelihood of the samples
    log_pdf = kde_skl.score_samples(x_grid[:, np.newaxis])
    return np.exp(log_pdf)

def kde_pudding(x, x_grid, bandwidth=0.2, **kwargs):
    '''Kernel Density Estimation with Pudding'''
    if len(x.shape) == 1:
        x = np.reshape(x, (-1, 1))
    if len(x_grid.shape) == 1:
        x_grid = np.reshape(x, (-1, 1))
    kde_pudding = KDE(kernel='gaussian', bandwidth=bandwidth)
    kde_pudding.fit(x)
    return kde_pudding.predict(x_grid)

kde_funcnames = ['Scikit-learn', 'Pudding']
kde_funcs = [kde_sklearn, kde_pudding]

functions = dict(zip(kde_funcnames, kde_funcs))


def plot_scaling(N=1000, bandwidth=0.1, rtol=0.0,
                 Nreps=3, kwds=None, xgrid=None, fig_name=None):
    """
    Plot the time scaling of KDE algorithms.
    Either N, bandwidth, or rtol should be a 1D array.
    """
    if xgrid is None:
        xgrid = np.linspace(-10, 10, 50000)
    if kwds is None:
        kwds=dict()
    for name in functions:
        if name not in kwds:
            kwds[name] = {}
    times = defaultdict(list)
    
    B = np.broadcast(N, bandwidth, rtol)
    assert len(B.shape) == 1
    
    for N_i, bw_i, rtol_i in B:
        x = np.random.normal(size=int(N_i))
        kwds['Scikit-learn']['rtol'] = rtol_i
        for name, func in functions.items():
            t = 0.0
            for i in range(Nreps):
                t0 = time()
                func(x, xgrid, bw_i, **kwds[name])
                t1 = time()
                t += (t1 - t0)
            times[name].append(t / Nreps)

    for name in kde_funcnames:
        times[name] = times[name][1:]
    N = N[1:]
            
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.grid(color='white', linestyle='-', linewidth=2)
    
    if np.size(N) > 1:
        for name in kde_funcnames:
            ax.loglog(N, times[name], label=name)
        ax.set_xlabel('Number of points')
    elif np.size(bandwidth) > 1:
        for name in kde_funcnames:
            ax.loglog(bandwidth, times[name], label=name)
        ax.set_xlabel('Bandwidth')
    elif np.size(rtol) > 1:
        for name in kde_funcnames:
            ax.loglog(rtol, times[name], label=name)
        ax.set_xlabel('Relative Tolerance')
        
    ax.legend(loc=0)
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Execution time for KDE '
                 '({0} evaluations)'.format(len(xgrid)))

    if fig_name is not None:
        plt.savefig(fig_name)
    
    return times

if __name__ == '__main__':
    plot_scaling(N=np.logspace(0, 4, 11), fig_name='kde_benchmark_num_points.jpg')
