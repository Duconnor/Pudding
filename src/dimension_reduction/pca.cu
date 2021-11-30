#include <assert.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

#include <pudding/dimension_reduction.h>
#include <helper/helper_CUDA.h>
#include <helper/helper.cuh>

void _pcaGPU(const float* X, const int numSamples, const int numFeatures, const int numComponents, const float variancePercentage, float* principalComponets, float* principalAxes, float* variances, float* reconstructedX, int* numComponentsChosen) {
    /*
     * The GPU version of PCA
     * PCA can be implemented using the cuBLAS and cuSolver library.
     * 
     * PCA has four major steps:
     * 1. Compute the mean of X.
     * 2. Perform SVD on the centered data X - mean.
     * 3. Select the number of components using either numComponents of variancePercentage.
     * 4. Obtain the principal components, set the return results.
     * 5. Reconstruct the original X (this is actually not part of PCA but it can be useful so I put it here).
     */

    // Perform simple pre-condition check
    if (numComponents == -1) {
        assert (variancePercentage > 0 && variancePercentage < 1);
    } else {
        assert (numComponents <= min(numSamples, numFeatures));
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
    float* devicePrincipalComponets; // The principal components (i.e. the lower dimensional representation of the original data).
    float* deviceReconstructedX; // The reconstructed X.
    float* devicePrincipalAxes; // The principal axes.
    int* deviceInfo; // This is used for solving SVD using cusolver.
    
    CUDA_CALL( cudaMalloc(&deviceX, sizeof(float) * numSamples * numFeatures) );
    CUDA_CALL( cudaMalloc(&deviceAllOneVec, sizeof(float) * numSamples) );
    CUDA_CALL( cudaMalloc(&deviceMeanVec, sizeof(float) * numFeatures) );
    CUDA_CALL( cudaMalloc(&deviceCenteredX, sizeof(float) * numSamples * numFeatures) );
    CUDA_CALL( cudaMalloc(&deviceS, sizeof(float) * min(numSamples, numFeatures)) );
    CUDA_CALL( cudaMalloc(&deviceU, sizeof(float) * numSamples * numSamples) );
    CUDA_CALL( cudaMalloc(&deviceV, sizeof(float) * numFeatures * numFeatures) );
    CUDA_CALL( cudaMalloc(&deviceVariances, sizeof(float) * min(numFeatures, numSamples)) );
    CUDA_CALL( cudaMalloc(&deviceReconstructedX, sizeof(float) * numSamples * numFeatures) );
    CUDA_CALL( cudaMalloc(&deviceInfo, sizeof(int)) );

    wrapperInitializeAllElementsToXKernel(deviceAllOneVec, 1.0, numSamples);
    CUDA_CALL( cudaMemcpy(deviceX, X, sizeof(float) * numSamples * numFeatures, cudaMemcpyHostToDevice) );

    // Prepare the handle for cublas
    cublasHandle_t cublasHandle = NULL;
    CUBLAS_CALL( cublasCreate(&cublasHandle) );

    // Prepare useful constant for cublas calls
    const float one = 1.0, zero = 0.0, negOne = -1.0;

    // Transpose deviceX to enable coalesced memory access in the kernel
    transposeMatrix(deviceX, numSamples, numFeatures);

    // PCA begins here:
    // 1. Compute the mean of X. X is of shape (numSample, numFeature), the mean vector is of shape (numFeature,)
    // We use the cublas call to perform a matrix-vector matrix multiplcation in order to perform the row reduction.
    float oneOverNumSamples = 1.0 / numSamples;
    CUBLAS_CALL( cublasSgemv(cublasHandle, CUBLAS_OP_T, numSamples, numFeatures, &oneOverNumSamples, deviceX, numSamples, deviceAllOneVec, one, &zero, deviceMeanVec, one) );

    // 2. Perform SVD on the centered data.
    // 2.1. Center the data
    wrapperMatrixVectorAddition(deviceX, numFeatures, numSamples, deviceMeanVec, negOne, deviceCenteredX);
    // 2.2. Perform SVD on deviceCenteredX
    // Configuration of gesvdj
    const cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR;
    const int econ = 0;

    // Create the cusolver handle and bind a stream
    cusolverDnHandle_t cusolverHandle = NULL;
    cudaStream_t stream = NULL;
    CUSOLVER_CALL( cusolverDnCreate(&cusolverHandle) );
    CUDA_CALL( cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) );
    CUSOLVER_CALL( cusolverDnSetStream(cusolverHandle, stream) );

    // Set the configuration of gesvdj
    gesvdjInfo_t gesvdParams = NULL;
    CUSOLVER_CALL( cusolverDnCreateGesvdjInfo(&gesvdParams) );

    // Prepare the work buffer
    int workBufferSize = 0;
    CUSOLVER_CALL( cusolverDnSgesvdj_bufferSize(cusolverHandle, jobz, econ, numSamples, numFeatures, deviceCenteredX, numSamples, deviceS, deviceU, numSamples, deviceV, numFeatures, &workBufferSize, gesvdParams) );
    CUDA_CALL( cudaMalloc(&deviceWorkBuffer, sizeof(float) * workBufferSize) );

    // Perform the actual SVD computation
    CUSOLVER_CALL( cusolverDnSgesvdj(cusolverHandle, jobz, econ, numSamples, numFeatures, deviceCenteredX, numSamples, deviceS, deviceU, numSamples, deviceV, numFeatures, deviceWorkBuffer, workBufferSize, deviceInfo, gesvdParams) );

    // We need to synchronize device here cause we use stream before
    CUDA_CALL( cudaDeviceSynchronize() );

    // 3. Select the number of components
    // The selection is based on the varince. So we first need to compute the variances using the singular values
    wrapperVectorVectorElementWiseMultiplication(deviceS, deviceS, min(numFeatures, numSamples), 1.0 / (numSamples - 1), deviceVariances);
    // Then we copy the variances from device to host
    CUDA_CALL( cudaMemcpy(variances, deviceVariances, sizeof(float) * min(numFeatures, numSamples), cudaMemcpyDeviceToHost) );
    if (numComponents == -1) {
        // In this case, we need to select the number of components such that the ratio of the accumulated variance goes above the required variancePercentage
        // In order to select, we need the summation of all variances
        float* variancesSum = (float*)malloc(sizeof(float));
        // Actually, cublasSasum compute the sum of the **absolute value** of elements in a vector. However, variance is guaranteed to be positive here, so we can just use this function to compute the summation.
        CUBLAS_CALL( cublasSasum(cublasHandle, min(numFeatures, numSamples), deviceVariances, one, variancesSum) );
        // Based on the sum, we select the number of components needed
        *numComponentsChosen = 0;
        float currentSum = 0.0;
        while (*numComponentsChosen < min(numFeatures, numSamples)) {
            currentSum += variances[*numComponentsChosen];
            *numComponentsChosen = *numComponentsChosen + 1;
            if (currentSum > variancePercentage * (*variancesSum)) {
                break;
            }
        }
        if (variancesSum) {
            free(variancesSum);
        }
    } else {
        *numComponentsChosen = numComponents;
    }

    // 4. Obtain the principal components and set the result
    // 4.1. Copy the first numComponents columns from V to principalAxes
    // Since cublas uses column-major storage, the column of V is stored continuously
    // Therefore, we can simply copy the first numComponents * numFeatures * sizeof(float) here :)
    CUDA_CALL( cudaMemcpy(principalAxes, deviceV, sizeof(float) * (*numComponentsChosen) * numFeatures, cudaMemcpyDeviceToHost) );
    // 4.2. Obtain the principal components by performing U * S
    CUDA_CALL( cudaMalloc(&devicePrincipalComponets, sizeof(float) * numSamples * (*numComponentsChosen)) );
    CUBLAS_CALL( cublasSdgmm(cublasHandle, CUBLAS_SIDE_RIGHT, numSamples, *numComponentsChosen, deviceU, numSamples, deviceS, one, devicePrincipalComponets, numSamples) );
    transposeMatrix(devicePrincipalComponets, *numComponentsChosen, numSamples);
    CUDA_CALL( cudaMemcpy(principalComponets, devicePrincipalComponets, sizeof(float) * numSamples * *numComponentsChosen, cudaMemcpyDeviceToHost) );

    // 5. Reconstruct the original data X
    CUDA_CALL( cudaMalloc(&devicePrincipalAxes, sizeof(float) * (*numComponentsChosen) * numFeatures) );
    CUDA_CALL( cudaMemcpy(devicePrincipalAxes, deviceV, sizeof(float) * (*numComponentsChosen) * numFeatures, cudaMemcpyDeviceToDevice) );
    CUBLAS_CALL( cublasSgemm(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_T, numSamples, numFeatures, *numComponentsChosen, &one, devicePrincipalComponets, *numComponentsChosen, devicePrincipalAxes, numFeatures, &zero, deviceReconstructedX, numSamples) );
    wrapperMatrixVectorAddition(deviceReconstructedX, numFeatures, numSamples, deviceMeanVec, one, deviceReconstructedX);
    transposeMatrix(deviceReconstructedX, numFeatures, numSamples);
    CUDA_CALL( cudaMemcpy(reconstructedX, deviceReconstructedX, sizeof(float) * numSamples * numFeatures, cudaMemcpyDeviceToHost) );

    // Free GPU spaces
    CUDA_CALL( cudaFree(deviceX) );
    CUDA_CALL( cudaFree(deviceAllOneVec) );
    CUDA_CALL( cudaFree(deviceMeanVec) );
    CUDA_CALL( cudaFree(deviceCenteredX) );
    CUDA_CALL( cudaFree(deviceWorkBuffer) );
    CUDA_CALL( cudaFree(deviceS) );
    CUDA_CALL( cudaFree(deviceU) );
    CUDA_CALL( cudaFree(deviceV) );
    CUDA_CALL( cudaFree(deviceVariances) );
    CUDA_CALL( cudaFree(devicePrincipalComponets) );
    CUDA_CALL( cudaFree(deviceReconstructedX) );
    CUDA_CALL( cudaFree(devicePrincipalAxes) );
    CUDA_CALL( cudaFree(deviceInfo) );

    CUBLAS_CALL( cublasDestroy(cublasHandle) );
    CUDA_CALL( cudaStreamDestroy(stream) );
    CUSOLVER_CALL( cusolverDnDestroy(cusolverHandle) );
    CUSOLVER_CALL( cusolverDnDestroyGesvdjInfo(gesvdParams) );
}
