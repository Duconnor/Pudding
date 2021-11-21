import pytest
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

iris = load_iris()

# @pytest.mark.parametrize("n_components", range(1, iris.data.shape[1]))
# def test_pca(svd_solver, n_components):
#     X = iris.data
#     pca = PCA(n_components=n_components, svd_solver=svd_solver)

#     # check the shape of fit.transform
#     X_r = pca.fit(X).transform(X)
#     assert X_r.shape[1] == n_components

#     # check the equivalence of fit.transform and fit_transform
#     X_r2 = pca.fit_transform(X)
#     assert_allclose(X_r, X_r2)
#     X_r = pca.transform(X)
#     assert_allclose(X_r, X_r2)

#     # Test get_covariance and get_precision
#     cov = pca.get_covariance()
#     precision = pca.get_precision()
#     assert_allclose(np.dot(cov, precision), np.eye(X.shape[1]), atol=1e-12)