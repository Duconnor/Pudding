import pudding
import pytest
import numpy as np

from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

import pudding

iris = load_iris()

def test_pca_reduction_toy_data():
    '''
    This test case test Pudding's PCA using some toy data
    '''
    # Prepare the test data
    X = [[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]]
    n_components = 2

    expected_principal_components = np.array([[1.38340578, 0.2935787], [2.22189802, -0.25133484], [3.6053038, 0.04224385], [-1.38340578, -0.2935787], [-2.22189802, 0.25133484], [-3.6053038, -0.04224385]])
    expected_principal_axes = np.array([[-0.83849224, -0.54491354], [0.54491354, -0.83849224]])
    expected_variances = np.array([7.93954312, 0.06045688])
    expected_reconstructed_X = np.array(X)
    
    # Launch
    pca = pudding.dimension_reduction.PCA(n_components=n_components)
    pca.fit(X)
    principal_components, principal_axes, variances, reconstructed_X = pca.principal_components, pca.principal_axes, pca.variance, pca.reconstructed_X

    # Convert to numpy array
    principal_components = np.array(principal_components)
    principal_axes = np.array(principal_axes)
    variances = np.array(variances)

    # Check
    assert principal_components.shape == (len(X), n_components)
    assert principal_axes.shape == (n_components, len(X[0]))
    assert len(variances) == n_components

    for val_this_component, principal_axis, expected_val_this_component, expected_principal_axis in zip(principal_components.T, principal_axes, expected_principal_components.T, expected_principal_axes):
        assert np.allclose(principal_axis, expected_principal_axis) or np.allclose(-principal_axis, expected_principal_axis)
        if np.allclose(principal_axis, expected_principal_axis):
            assert np.allclose(val_this_component, expected_val_this_component, atol=1e-5, rtol=1e-5)
        else:
            assert np.allclose(-val_this_component, expected_val_this_component, atol=1e-5, rtol=1e-5)
    assert np.allclose(variances, expected_variances)
    assert np.allclose(reconstructed_X, expected_reconstructed_X)


@pytest.mark.parametrize("n_components", list(range(1, iris.data.shape[1])) + [0.3, 0.6, 0.9] + [None])
def test_pca_reduction_with_sklearn(n_components):
    '''
    This test case test Pudding's PCA reduction with sklearn's PCA using the iris dataset
    '''
    # Prepare the iris dataset
    X = iris.data

    # sklearn's result
    sklearn_pca = PCA(n_components=n_components, svd_solver='full')
    sklearn_pca.fit(X)
    X_r = sklearn_pca.transform(X)
    sklearn_recontructed_X = sklearn_pca.inverse_transform(X_r)

    # Pudding's result
    pudding_pca = pudding.dimension_reduction.PCA(n_components=n_components)
    pudding_pca.fit(X)
    principal_components, principal_axes, variances, reconstructed_X = pudding_pca.principal_components, pudding_pca.principal_axes, pudding_pca.variance, pudding_pca.reconstructed_X

    # Convert to numpy res
    principal_components = np.array(principal_components)
    principal_axes = np.array(principal_axes)
    variances = np.array(variances)

    # Size check
    assert principal_components.shape == X_r.shape
    assert len(principal_axes) == len(sklearn_pca.components_)
    assert len(variances) == len(sklearn_pca.explained_variance_)

    # Contents check
    # Comparing the PCA result is a bit tricky here. For principal axes, x and -x should be equivalent
    # Therefore, we cannot simply ask all results to be exactly the same
    assert np.allclose(variances, sklearn_pca.explained_variance_)
    for val_this_component, principal_axis, sklearn_val_this_component, sklearn_principal_axis in zip(principal_components.T, principal_axes, X_r.T, sklearn_pca.components_):
        assert np.allclose(principal_axis, sklearn_principal_axis) or np.allclose(-principal_axis, sklearn_principal_axis)
        if np.allclose(principal_axis, sklearn_principal_axis):
            assert np.allclose(val_this_component, sklearn_val_this_component, atol=1e-5, rtol=1e-5)
        else:
            assert np.allclose(-val_this_component, sklearn_val_this_component, atol=1e-5, rtol=1e-5)
    assert np.allclose(reconstructed_X, sklearn_recontructed_X, atol=1e-5, rtol=1e-5)
