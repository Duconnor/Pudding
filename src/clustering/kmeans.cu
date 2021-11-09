#include <assert.h>
#include <math.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "../helper/helperCUDA.h"

__global__
void computeDistanceKernel(const float* X, const float* centers, float* distances, const int numSamples, const int numFeatures, const int numCenters) {
    int threadId = threadIdx.x + blockIdx.x * blockDim.x;
    int idxSample = threadId / numCenters;
    int idxCenter = threadId % numCenters;

    while(idxSample < numSamples) {
        for (int i = 0; i < numFeatures; i++) {
            distances[idxSample * numCenters + idxCenter] += pow(X[idxSample * numFeatures + i] - centers[idxCenter * numFeatures + i], 2);
        }
        threadId += gridDim.x * blockDim.x;
        idxSample = threadId / numCenters;
        idxCenter = threadId % numCenters;
    }
}

__global__
void determineMembershipKernel(const float* X, const float* distances, int* membership, const int numSamples, const int numFeatures, const int numCenters) {
    int idxSample = threadIdx.x + blockIdx.x * blockDim.x;

    while (idxSample < numSamples) {
        int minIdx = 0;
        for (int i = 1; i < numCenters; i++) {
            if (distances[idxSample * numCenters + i] < distances[idxSample * numCenters + minIdx]) {
                minIdx = i;
            }
        }
        membership[idxSample] = minIdx;
        idxSample += gridDim.x * blockDim.x;
    }
}

__global__
void updateCentersKernel(const float* X, const int* membership, float* centers, int* numSamplesThisCenter, const int numSamples, const int numFeatures, const int numCenters) {
    /*
     * Pre-condition: centers and numSamplesThisCenter are initialized to all zeros
     */
    int idxCenter = threadIdx.x + blockIdx.x * blockDim.x;
    
    while (idxCenter < numCenters) {

        for (int i = 0; i < numSamples; i++) {
            if (membership[i] == idxCenter) {
                for (int j = 0; j < numFeatures; j++) {
                    centers[idxCenter * numFeatures + j] += X[i * numFeatures + j];
                }
                numSamplesThisCenter[idxCenter]++;
            }
        }

        for (int j = 0; j < numFeatures; j++) {
            centers[idxCenter * numFeatures + j] /= numSamplesThisCenter[idxCenter];
        }

        idxCenter += gridDim.x * blockDim.x;
    }
}

/* GPU version of KMeans */
void _kmeansGPU(const float* X, const float* initCenters, const int numSamples, const int numFeatures, const int numCenters, const int maxNumIteration, const float tolerance, float* centers, int* membership, int* numIterations) {
    /*
     * Use GPU to accelerate the KMeans algorithm
     * The whole process will be done using three separate kernels:
     *  1. The first kernel compute the distance of points to clusters
     *  2  The second kernel determine the membership of points to clusters
     *  3. The third kernel update the center of each cluster
     */

    assert(maxNumIteration >= 0);

    memcpy(centers, initCenters, sizeof(float) * numCenters * numFeatures);
    bool endFlag = maxNumIteration == 0;
    int iterationCount = 0;

    // Malloc space on GPU
    float* deviceX;
    float* deviceCenters;
    float* deviceDistance;
    int* deviceMembership;
    int* deviceNumSamplesThisCenter;
    float* deviceOldCenters;

    CUDA_CALL( cudaMalloc(&deviceX, sizeof(float) * numSamples * numFeatures) );
    CUDA_CALL( cudaMalloc(&deviceCenters, sizeof(float) * numCenters * numFeatures) );
    CUDA_CALL( cudaMalloc(&deviceDistance, sizeof(float) * numSamples * numCenters) );
    CUDA_CALL( cudaMalloc(&deviceMembership, sizeof(int) * numSamples) );
    CUDA_CALL( cudaMalloc(&deviceNumSamplesThisCenter, sizeof(int) * numSamples) );
    CUDA_CALL( cudaMalloc(&deviceOldCenters, sizeof(float) * numCenters * numFeatures) );

    CUDA_CALL( cudaMemcpy(deviceX, X, sizeof(float) * numSamples * numFeatures, cudaMemcpyHostToDevice) );
    CUDA_CALL( cudaMemcpy(deviceCenters, centers, sizeof(float) * numCenters * numFeatures, cudaMemcpyHostToDevice) );
    CUDA_CALL( cudaMemset(deviceMembership, 0, sizeof(int) * numSamples) );

    // Determine the block width
    const int BLOCKWIDTH = 1024;

    // Initialize the cublas handle
    cublasHandle_t cublasHandle;
    cublasCreate(&cublasHandle);

    while (!endFlag) {
        // Compute the distance between samples and clusters
        int numBlock = min(65535, ((numSamples * numCenters) + BLOCKWIDTH - 1) / BLOCKWIDTH);
        CUDA_CALL( cudaMemset(deviceDistance, 0, sizeof(float) * numSamples * numCenters) );
        computeDistanceKernel<<<numBlock, BLOCKWIDTH>>>(deviceX, deviceCenters, deviceDistance, numSamples, numFeatures, numCenters);
        // Determine the membership of each sample
        numBlock = min(65535, ((numSamples) + BLOCKWIDTH - 1) / BLOCKWIDTH);
        determineMembershipKernel<<<numBlock, BLOCKWIDTH>>>(deviceX, deviceDistance, deviceMembership, numSamples, numFeatures, numCenters);

        // Save the result of old centers
        CUDA_CALL( cudaMemcpy(deviceOldCenters, deviceCenters, sizeof(float) * numCenters * numFeatures, cudaMemcpyDeviceToDevice) );
        CUDA_CALL( cudaMemset(deviceCenters, 0, sizeof(float) * numCenters * numFeatures));

        // Update the center estimation
        CUDA_CALL( cudaMemset(deviceNumSamplesThisCenter, 0, sizeof(int) * numSamples) );
        numBlock = min(65535, ((numCenters) + BLOCKWIDTH - 1) / BLOCKWIDTH);
        updateCentersKernel<<<numBlock, BLOCKWIDTH>>>(deviceX, deviceMembership, deviceCenters, deviceNumSamplesThisCenter, numSamples, numFeatures, numCenters);

        // Test for coverage
        iterationCount++;
        if (iterationCount >= maxNumIteration) {
            endFlag = true;
        } else {
            float one = 1, negOne = -1;
            // Careful here, cuBlas assumes column major storage
            // Perform element-wise subtraction
            CUBLAS_CALL( cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, numFeatures, numSamples, &one, deviceOldCenters, numFeatures, &negOne, deviceCenters, numFeatures, deviceOldCenters, numFeatures) );
            float diff = 0.0;
            // Compute the F-norm
            CUBLAS_CALL( cublasSdot(cublasHandle, numCenters * numFeatures, deviceOldCenters, 1, deviceOldCenters, 1, &diff) );
            endFlag = sqrt(diff) < tolerance;
        }
    }

    // Copy the result back to host
    CUDA_CALL( cudaMemcpy(centers, deviceCenters, sizeof(float) * numCenters * numFeatures, cudaMemcpyDeviceToHost) );
    CUDA_CALL( cudaMemcpy(membership, deviceMembership, sizeof(int) * numSamples, cudaMemcpyDeviceToHost) );
    *numIterations = iterationCount;

    // Free all resources on GPU
    CUDA_CALL( cudaFree(deviceX) );
    CUDA_CALL( cudaFree(deviceCenters) );
    CUDA_CALL( cudaFree(deviceDistance) );
    CUDA_CALL( cudaFree(deviceMembership) );
    CUDA_CALL( cudaFree(deviceNumSamplesThisCenter) );
    CUDA_CALL( cudaFree(deviceOldCenters) );

    CUDA_CALL( cublasDestroy(cublasHandle) );
    return;
}