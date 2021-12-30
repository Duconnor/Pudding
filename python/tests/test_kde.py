import numpy as np
import pytest

from sklearn.neighbors import KernelDensity

from pudding.estimation import KDE

@pytest.mark.parametrize('n_samples', [10, 1000, 4000, 5000, 100000])
def test_gaussian_kde_random_data(n_samples):
    '''
    This test case tests Pudding's kde using sklearn's kde as reference using some randomly generated data
    '''
    # Prepare for the test data
    rng = np.random.RandomState(42)
    X = rng.random_sample((n_samples, 3))

    kernel = 'gaussian'
    bandwidth = 0.5
    samples = X[:100]

    # Scikit-learn's result
    sklearn_kde = KernelDensity(kernel=kernel, bandwidth=bandwidth).fit(X)
    sklearn_log_density = sklearn_kde.score_samples(samples)
    sklearn_density = np.exp(sklearn_log_density)
    
    # Pudding's result
    pudding_kde = KDE(kernel=kernel, bandwidth=bandwidth)
    pudding_kde.fit(X)
    pudding_density = pudding_kde.predict(samples)
    pudding_density = np.array(pudding_density)

    # Check
    assert np.allclose(sklearn_density, pudding_density)
