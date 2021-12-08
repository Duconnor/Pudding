#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <assert.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <helper/helper_CUDA.h>
#include <helper/helper.cuh>

typedef float (*KERNEL_FUNC_P)(float);

enum KERNEL_FUNC_NAME {GAUSSIAN};

__device__
float gaussian(float x) {
    return exp(-((x * x) / 2));
}

__global__
void applyKernelFunctionKernel(float* X, int numElements, KERNEL_FUNC_NAME kernelFuncName, const float bandwidth) {
    // Each thread is responsible for one element
    int eleIdx = threadIdx.x + blockIdx.x * blockDim.x;

    // Prepare the kernel function
    KERNEL_FUNC_P kernelFunc = NULL;

    switch (kernelFuncName) {
        case GAUSSIAN: kernelFunc = gaussian; break;
        default: break; // Won't reach here
    }

    while (eleIdx < numElements) {
        X[eleIdx] = kernelFunc(sqrt(X[eleIdx]) / bandwidth);

        eleIdx += blockDim.x * gridDim.x;
    }
}

void _kdeScoreGPU(const float* X, const int numSamples, const int numFeatures, const char* kernel, const float bandwidth, const float* samplesX, const int numTestSamples, float* scores) {
    /*
     * Kernel density estimation is a three steps process
     * 1. Compute the pair-wise distance between test samples and training samples.
     * 2. Apply the corresponding kernel function on the distance matrix.
     * 3. Perform reduction (mean) for each test sample.
     */

    float one = 1.0, zero = 0.0;

    // Parse the kernel function and prepare for the normalization coefficient
    KERNEL_FUNC_NAME kernelFuncName;
    float normalizationCoeff = 0.0;
    if (strcmp(kernel, "gaussian") == 0) {
        kernelFuncName = GAUSSIAN;
        normalizationCoeff = (one / pow(2 * M_PI, 0.5 * numFeatures)) * (one / pow(bandwidth, numFeatures));
    } else {
        assert(false);
    }

    // Malloc space on GPU
    float* deviceX;
    float* deviceSamplesX;
    float* devicePairwiseDistance;
    float* deviceScores;

    CUDA_CALL( cudaMalloc(&deviceX, sizeof(float) * numSamples * numFeatures) );
    CUDA_CALL( cudaMalloc(&deviceSamplesX, sizeof(float) * numTestSamples * numFeatures) );
    CUDA_CALL( cudaMalloc(&devicePairwiseDistance, sizeof(float) * numSamples * numTestSamples) );
    CUDA_CALL( cudaMalloc(&deviceScores, sizeof(float) * numTestSamples) );

    CUDA_CALL( cudaMemcpy(deviceX, X, sizeof(float) * numSamples * numFeatures, cudaMemcpyHostToDevice) );
    CUDA_CALL( cudaMemcpy(deviceSamplesX, samplesX, sizeof(float) * numTestSamples * numFeatures, cudaMemcpyHostToDevice) );

    // Prepare cublas handle
    cublasHandle_t cublasHandle;
    CUBLAS_CALL( cublasCreate(&cublasHandle) );

    // Compute the pair wise distance
    // First, we need to transpose deviceX and deviceSamplesX
    transposeMatrix(deviceX, numSamples, numFeatures);
    transposeMatrix(deviceSamplesX, numTestSamples, numFeatures);
    // Perform the pair wise distance calculation using helper function
    wrapperComputePairwiseEuclideanDistanceKerenl(deviceX, deviceSamplesX, numSamples, numTestSamples, numFeatures, devicePairwiseDistance);

    // Apply the kernel function on each element of the distance matrix
    const int BLOCKWIDTH = 1024;
    const int GRIDWIDTH = min(65535, ((numSamples * numTestSamples) + BLOCKWIDTH - 1) / BLOCKWIDTH);
    applyKernelFunctionKernel<<<GRIDWIDTH, BLOCKWIDTH>>>(devicePairwiseDistance, numSamples * numTestSamples, kernelFuncName, bandwidth);

    // Perform the reduction
    // We use cublas function cublasSgemv() to perform the reduction, therefore, we need a all 1 vector
    float* deviceAllOne;
    CUDA_CALL( cudaMalloc(&deviceAllOne, sizeof(float) * numSamples) );
    wrapperInitializeAllElementsToXKernel(deviceAllOne, one, numSamples);
    float alpha = (one / numSamples) * normalizationCoeff;
    CUBLAS_CALL( cublasSgemv(cublasHandle, CUBLAS_OP_N, numTestSamples, numSamples, &alpha, devicePairwiseDistance, numTestSamples, deviceAllOne, one, &zero, deviceScores, one) );

    // Copy the result back to host
    CUDA_CALL( cudaMemcpy(scores, deviceScores, sizeof(float) * numTestSamples, cudaMemcpyDeviceToHost) );

    // Free resources
    CUDA_CALL( cudaFree(deviceX) );
    CUDA_CALL( cudaFree(deviceSamplesX) );
    CUDA_CALL( cudaFree(devicePairwiseDistance) );
    CUDA_CALL( cudaFree(deviceScores) );
    CUBLAS_CALL( cublasDestroy(cublasHandle) );

    CUDA_CALL( cudaFree(deviceAllOne) );

    return;
}