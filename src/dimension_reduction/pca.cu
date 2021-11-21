#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <pudding/dimension_reduction.h>

void _pcaGPU(const float* X, const int numSamples, const int numFeatures, const int targetDimension, float* projectedX) {
    /*
     * The GPU version of PCA
     * PCA can be implemented using the cuBLAS and cuSolver library.
     * 
     * PCA has __ major steps:
     * 1. Compute the mean of X.
     * 2. Perform SVD on the centered data X - mean.
     * 3. Project the original data into the lower dimensional space.
     */
}