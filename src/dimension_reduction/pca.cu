#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

#include <pudding/dimension_reduction.h>
#include <helper/helper_CUDA.h>
#include <helper/helper.cuh>

void _pcaGPU(const float* X, const int numSamples, const int numFeatures, const int numComponets, const float variancePercentage, float* principalComponets, float* principalAxes, float* variances) {
    /*
     * The GPU version of PCA
     * PCA can be implemented using the cuBLAS and cuSolver library.
     * 
     * PCA has four major steps:
     * 1. Compute the mean of X.
     * 2. Perform SVD on the centered data X - mean.
     * 3. Select the number of components using either numComponents of variancePercentage.
     * 4. Obtain the principal components, set the return results
     */

    // Perform simple pre-condition check
    if (numComponets == -1) {
        assert (variancePercentage > 0 && variancePercentage < 1);
    } else {
        assert (numComponets < min(numSamples, numFeatures));
    }

    // Malloc space on GPU
    float* deviceX;
    float* deviceAllOneVec;
    float* deviceMeanVec; // The mean vector of X.
    float* deviceCenteredX; // The centered X.
    float* deviceWorkBuffer; // The work buffer for performing SVD using cusolver.
    float* deviceS; // The sorted non-zero singular values.
    float* deviceU; // The left singular matrix U.
    float* deviceV; // The right singular matrix V^T.
    float* deviceVariances; // The actual variances along principal directions.
    // Temporary array for transpose
    float* tempDeviceX;
    
    CUDA_CALL( cudaMalloc(&deviceX, sizeof(float) * numSamples * numFeatures) );
    CUDA_CALL( cudaMalloc(&deviceAllOneVec, sizeof(float) * numSamples) );
    CUDA_CALL( cudaMalloc(&deviceMeanVec, sizeof(float) * numFeatures) );
    CUDA_CALL( cudaMalloc(&deviceCenteredX, sizeof(float) * numSamples * numFeatures) );
    CUDA_CALL( cudaMalloc(&deviceS, sizeof(float) * min(numSamples, numFeatures)) );
    CUDA_CALL( cudaMalloc(&deviceU, sizeof(float) * numSamples * numSamples) );
    CUDA_CALL( cudaMalloc(&deviceV, sizeof(float) * numFeatures * numFeatures) );
    CUDA_CALL( cudaMalloc(&tempDeviceX, sizeof(float) * numSamples * numFeatures) );

    CUDA_CALL( cudaMemset(deviceAllOneVec, 1, sizeof(float) * numSamples) );

    // Determine the block width
    const int BLOCKWIDTH = 1024;

    // Prepare the handle for cublas
    cublasHandle_t cublasHandle = NULL;
    CUBLAS_CALL( cublasCreate(&cublasHandle) );

    // Prepare useful constant for cublas calls
    const float one = 1.0, zero = 0.0, negOne = -1.0;

    // Transpose deviceX to enable coalesced memory access in the kernel
    CUDA_CALL( cudaMemcpy(tempDeviceX, X, sizeof(float) * numSamples * numFeatures, cudaMemcpyHostToDevice) );
    CUBLAS_CALL( cublasSgeam(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, numSamples, numFeatures, &one, tempDeviceX, numFeatures, &zero, tempDeviceX, numSamples, deviceX, numSamples) );

    // PCA begins here:
    // 1. Compute the mean of X. X is of shape (numSample, numFeature), the mean vector is of shape (numFeature,)
    // We use the cublas call to perform a matrix-vector matrix multiplcation in order to perform the row reduction.
    float oneOverNumSamples = 1.0 / numSamples;
    CUBLAS_CALL( cublasSgemv(cublasHandle, CUBLAS_OP_T, numSamples, numFeatures, &oneOverNumSamples, deviceX, numSamples, deviceAllOneVec, one, &zero, deviceMeanVec, one) );

    // 2. Perform SVD on the centered data.
    // 2.1. Center the data
    wrapperMatrixVectorSubtraction(deviceX, numFeatures, numSamples, deviceMeanVec, deviceCenteredX);
    // 2.2. Perform SVD on deviceCenteredX
    // TODO: Maybe the default configuration is better?
    // Configuration of gesvdj
    const float tol = 1e-7;
    const int maxSweeps = 15;
    const cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR;
    const int econ = 0;

    // Numerical results of gesvdj
    float residual = 0;
    int executedSweeps = 0;

    // Create the cusolver handle and bind a stream
    cusolverDnHandle_t cusolverHandle = NULL;
    cudaStream_t stream = NULL;
    CUSOLVER_CALL( cusolverDnCreate(&cusolverHandle) );
    CUDA_CALL( cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) );
    CUSOLVER_CALL( cusolverDnSetStream(cusolverHandle, stream) );

    // Set the configuration of gesvdj
    gesvdjInfo_t gesvdParams = NULL;
    CUSOLVER_CALL( cusolverDnCreateGesvdjInfo(&gesvdParams) );
    CUSOLVER_CALL( cusolverDnXgesvdjSetTolerance(gesvdParams, tol) );
    CUSOLVER_CALL( cusolverDnXgesvdjSetMaxSweeps(gesvdParams, maxSweeps) );

    // Prepare the work buffer
    float workBufferSize = 0.0;
    CUSOLVER_CALL( cusolverDnSgesvdj_bufferSize(cusolverHandle, jobz, econ, numSamples, numFeatures, deviceCenteredX, numSamples, deviceS, deviceU, numSamples, deviceV, numFeatures, &workBufferSize, gesvdParams) );
    CUDA_CALL( cudaMalloc(&deviceWorkBuffer, sizeof(float) * workBufferSize) );

    // Perform the actual SVD computation
    int info = 0;
    CUSOLVER_CALL( cusolverDnSgesvdj(cusolverHandle, jobz, econ, numSamples, numFeatures, deviceCenteredX, numSamples, deviceS, deviceU, numSamples, deviceV, numFeatures, deviceWorkBuffer, workBufferSize, &info, gesvdParams) );

    // TODO: Do we need to synchronize device here?

    // 3. Select the number of components
    if (numComponets == -1) {
        // 3.1. In this case, we need to select the number of components such that the ratio of the accumulated variance goes above the required variancePercentage
        assert (false); // TODO: For simplicity, add this later.
    }

    // 4. Obtain the principal components and set the result
    // 4.1. Compute the variances using the singular values
    // TODO: write our own kernel that performs element-wise multiplication

    // Free GPU spaces
    CUDA_CALL( cudaFree(deviceX) );
    CUDA_CALL( cudaFree(deviceAllOneVec) );
    CUDA_CALL( cudaFree(deviceMeanVec) );
    CUDA_CALL( cudaFree(deviceCenteredX) );
    CUDA_CALL( cudaFree(deviceWorkBuffer) );
    CUDA_CALL( cudaFree(deviceS) );
    CUDA_CALL( cudaFree(deviceU) );
    CUDA_CALL( cudaFree(deviceV) );
    CUDA_CALL( cudaFree(tempDeviceX) );

    CUBLAS_CALL( cublasDestroy(cublasHandle) );
    CUDA_CALL( cudaStreamDestroy(stream) );
    CUSOLVER_CALL( cusolverDnDestroy(cusolverHandle) );
    CUSOLVER_CALL( cusolverDnDestroyGesvdjInfo(gesvdParams) );
}