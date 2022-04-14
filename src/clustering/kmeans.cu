#include <assert.h>
#include <math.h>
#include <float.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <helper/helper.cuh>
#include <helper/helper_CUDA.h>

__global__
void determineMembershipKernel(const float* X, const float* centers, int* membership, const int numSamples, const int numFeatures, const int numCenters) {
    // Each thread is responsible for each data point
    int idxSample = threadIdx.x + blockIdx.x * blockDim.x;

    // Use shared memory to accelerate computation
    // We copy X and centers to shared memory to avoid unnecessary global memory access
    extern __shared__ float sharedMem[];

    float* sharedX = sharedMem;
    // Each block has blockDim.x threads, and each threads is responsible for one data point
    // Therefore, in each block, the shared memory allocated for X is blockDim.x * numFeatures in total
    const int numSamplesThisBlock = blockDim.x;
    float* sharedCenters = sharedMem + (numSamplesThisBlock * numFeatures);
    
    const int idxSampleSharedMem = threadIdx.x;

    // The first 'numCenters' threads in this block are responsible for loading the data of centers
    // TODO: We can't just load all centers into the shared memory, there wouldn't be enough space in some extreme case.
    int sharedCentersIdx = threadIdx.x;
    while (sharedCentersIdx < numCenters) {
        for (int idxFeature = 0; idxFeature < numFeatures; idxFeature++) {
            sharedCenters[idxFeature * numCenters + sharedCentersIdx] = centers[idxFeature * numCenters + sharedCentersIdx];
        }
        sharedCentersIdx += blockDim.x;
    }
    __syncthreads();

    // We need to make sure all threads in a block loop for the same number of times or else the behavior of __syncthreads() is undefined. See more in https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#synchronization-functions.
    const int loopCount = numSamples % (gridDim.x * blockDim.x) == 0 ? numSamples / (gridDim.x * blockDim.x) : numSamples / (gridDim.x * blockDim.x) + 1;
    int loopIdx = 0;
    while (loopIdx < loopCount) {
        // Load this block's data into the shared memory
        for (int idxFeature = 0; idxFeature < numFeatures; idxFeature++) {
            if (idxSample < numSamples) {
                sharedX[idxFeature * numSamplesThisBlock + idxSampleSharedMem] = X[idxFeature * numSamples + idxSample];
            }
        }
        __syncthreads();

        float minDist = FLT_MAX;
        int minDistIdx = -1;
        for (int idxCenter = 0; idxCenter < numCenters; idxCenter++) {
            float dist = 0;
            for (int idxFeature = 0; idxFeature < numFeatures; idxFeature++) {
                dist += pow(sharedX[idxFeature * numSamplesThisBlock + idxSampleSharedMem] - sharedCenters[idxFeature * numCenters + idxCenter], 2);
            }
            if (minDistIdx == -1 || minDist > dist) {
                minDist = dist;
                minDistIdx = idxCenter;
            }
        }
        if (idxSample < numSamples) {
            membership[idxSample] = minDistIdx;
        }
        idxSample += gridDim.x * blockDim.x;
        loopIdx++;
    }
}

__global__
void updateCentersKernel(const float* X, const int* membership, float* centers, int* deviceSamplesCount, const int idxCenter, const int numSamples, const int numFeatures, const int numCenters) {
    // For updating every center, this kernel will be invoked once
    // In this kernel, we perform a reduction sum
    int idxSample = threadIdx.x + blockIdx.x * blockDim.x;

    extern __shared__ float sharedMem[];

    float* sharedX = sharedMem;
    const int numSamplesSharedMem = blockDim.x;
    int* sharedSampleCount = (int*)(sharedMem + (numSamplesSharedMem * numFeatures));

    const int idxSampleSharedMem = threadIdx.x;

    for (int idxFeature = 0; idxFeature < numFeatures; idxFeature++) {
        sharedX[idxFeature * numSamplesSharedMem + idxSampleSharedMem] = 0;
    }
    sharedSampleCount[idxSampleSharedMem] = 0;
    __syncthreads();

    // Same reason as above, need to make sure all threads in a block loop for the same number of times.
    const int loopCount = numSamples % (gridDim.x * blockDim.x) == 0 ? numSamples / (gridDim.x * blockDim.x) : numSamples / (gridDim.x * blockDim.x) + 1;
    int loopIdx = 0;
    while (loopIdx < loopCount) {
        // Initialize the shared memory
        int member = idxSample < numSamples ? membership[idxSample] : -1;
        sharedSampleCount[idxSampleSharedMem] = member == idxCenter;
        for (int idxFeature = 0; idxFeature < numFeatures; idxFeature++) {
            sharedX[idxFeature * numSamplesSharedMem + idxSampleSharedMem] = idxSample < numSamples ? X[idxFeature * numSamples + idxSample] * (member == idxCenter) : 0;
        }
        __syncthreads();

        // Reduction begin here
        int range = blockDim.x;
        for (int i = 0; i < (int)log2((float)blockDim.x); i++) {
            range /= 2;
            if (idxSampleSharedMem < range) {
                sharedSampleCount[idxSampleSharedMem] += sharedSampleCount[idxSampleSharedMem + range];
                sharedSampleCount[idxSampleSharedMem + range] = 0;
                for (int idxFeature = 0; idxFeature < numFeatures; idxFeature++) {
                    sharedX[idxFeature * numSamplesSharedMem + idxSampleSharedMem] += sharedX[idxFeature * numSamplesSharedMem + (idxSampleSharedMem + range)];
                    sharedX[idxFeature * numSamplesSharedMem + (idxSampleSharedMem + range)] = 0;
                }
            }
            __syncthreads();
        }

        // Use atomic operation to accumulate the result in global memory
        if (threadIdx.x == 0) {
            atomicAdd(deviceSamplesCount, sharedSampleCount[0]);
            sharedSampleCount[0] = 0;
            for (int idxFeature = 0; idxFeature < numFeatures; idxFeature++) {
                atomicAdd(centers + (idxFeature * numCenters + idxCenter), sharedX[idxFeature * numSamplesSharedMem]);
                sharedX[idxFeature * numSamplesSharedMem] = 0;
            }
        }

        idxSample += gridDim.x * blockDim.x;
        loopIdx++;
    }
}

/* GPU version of KMeans */
void _kmeansGPU(const float* X, const float* initCenters, const int numSamples, const int numFeatures, const int numCenters, const int maxNumIteration, const float tolerance, float* centers, int* membership, int* numIterations) {
    /*
     * Use GPU to accelerate the KMeans algorithm
     * The whole process will be done using two separate kernels:
     *  1. The first kernel determine the membership of points to clusters
     *  2. The second kernel update the center of each cluster
     */

    assert(maxNumIteration >= 0);

    memcpy(centers, initCenters, sizeof(float) * numCenters * numFeatures);
    bool endFlag = maxNumIteration == 0;
    int iterationCount = 0;

    // Malloc space on GPU
    float* deviceX;
    float* deviceCenters;
    float* deviceDist;             // Pair-wise distance of X and centers, of shape (numSamples, numCenters)
    float* deviceDummyArgMaxRow;   // A dummy vector storing the return value of wrapperMatrixArgMaxRowKernel() when determining the membership
    int* deviceMembership;
    float* deviceMaskVec;          // Mask vector, used for updating centers
    float* deviceOldCenters;
    int* deviceSamplesCount;

    CUDA_CALL( cudaMalloc(&deviceX, sizeof(float) * numSamples * numFeatures) );
    CUDA_CALL( cudaMalloc(&deviceCenters, sizeof(float) * numCenters * numFeatures) );
    CUDA_CALL( cudaMalloc(&deviceDist, sizeof(float) * numSamples * numCenters) );
    CUDA_CALL( cudaMalloc(&deviceDummyArgMaxRow, sizeof(float) * numSamples) );
    CUDA_CALL( cudaMalloc(&deviceMembership, sizeof(int) * numSamples) );
    CUDA_CALL( cudaMalloc(&deviceMaskVec, sizeof(int) * numSamples) );
    CUDA_CALL( cudaMalloc(&deviceOldCenters, sizeof(float) * numCenters * numFeatures) );
    CUDA_CALL( cudaMalloc(&deviceSamplesCount, sizeof(int)) );
    CUDA_CALL( cudaMemcpy(deviceX, X, sizeof(float) * numSamples * numFeatures, cudaMemcpyHostToDevice) );
    CUDA_CALL( cudaMemcpy(deviceCenters, centers, sizeof(float) * numCenters * numFeatures, cudaMemcpyHostToDevice) );

    // Malloc space on CPU
    int* samplesCount = (int*)malloc(sizeof(int));

    // Initialize the cublas handle
    cublasHandle_t cublasHandle;
    CUBLAS_CALL( cublasCreate(&cublasHandle) );
    // These are useful when calling cublas functions
    float one = 1.0, negOne = -1.0;
    float zero{float{0.0}};

    // Transpose deviceX, deviceCenters here to enable coalesced memory access in the kernel
    transposeMatrix(deviceX, numSamples, numFeatures);
    transposeMatrix(deviceCenters, numCenters, numFeatures);

    while (!endFlag) {        
        // Determine the membership of each sample
        wrapperComputePairwiseEuclideanDistanceKerenl(deviceX, deviceCenters, numSamples, numCenters, numFeatures, deviceDist);
        // Determine the membership of each sample requires us to select the argmin along each row
        // Since we already have a kernel that selects the argmax along each row, we try to reuse that instead of implementing a new one
        // Therefore, we multiply each element by -1 to make it negative
        CUBLAS_CALL( cublasSscal(cublasHandle, numSamples * numCenters, &negOne, deviceDist, one) );
        wrapperMatrixArgMaxRowKernel(deviceDist, numSamples, numCenters, deviceDummyArgMaxRow, deviceMembership);

        // Save the result of old centers
        CUDA_CALL( cudaMemcpy(deviceOldCenters, deviceCenters, sizeof(float) * numCenters * numFeatures, cudaMemcpyDeviceToDevice) );

        // Update the center estimation
        for (int idxCenter = 0; idxCenter < numCenters; idxCenter++) {
            // We use cublas functions + a mask vector to perform update
            wrapperGenerateMaskVectorKernel(deviceMembership, idxCenter, numSamples, deviceMaskVec);
            float numSamplesThisClass{0.0};
            CUBLAS_CALL( cublasSasum(cublasHandle, numSamples, deviceMaskVec, one, &numSamplesThisClass) );
            if (numSamplesThisClass == 0.0) {
                // Empty cluster, keep it unchanged
                // Do nothing
            } else {
                float denominator{float{1.0} / numSamplesThisClass};
                CUBLAS_CALL( cublasSgemv(cublasHandle, CUBLAS_OP_T, numSamples, numFeatures, &denominator, deviceX, numSamples, deviceMaskVec, one, &zero, deviceCenters + idxCenter, numCenters) );
            }
        }

        // Test for coverage
        iterationCount++;
        if (iterationCount >= maxNumIteration) {
            endFlag = true;
        } else {
            // Careful here, cuBlas assumes column major storage
            // Perform element-wise subtraction
            CUBLAS_CALL( cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, numCenters, numFeatures, &one, deviceOldCenters, numCenters, &negOne, deviceCenters, numCenters, deviceOldCenters, numCenters) );
            float diff = 0.0;
            // Compute the F-norm
            CUBLAS_CALL( cublasSdot(cublasHandle, numCenters * numFeatures, deviceOldCenters, 1, deviceOldCenters, 1, &diff) );
            endFlag = diff < tolerance;
        }
    }

    // Copy the result back to host
    // Tranpose the deviceCenters back
    transposeMatrix(deviceCenters, numFeatures, numCenters);

    CUDA_CALL( cudaMemcpy(centers, deviceCenters, sizeof(float) * numCenters * numFeatures, cudaMemcpyDeviceToHost) );
    CUDA_CALL( cudaMemcpy(membership, deviceMembership, sizeof(int) * numSamples, cudaMemcpyDeviceToHost) );
    *numIterations = iterationCount;

    // Free all resources on GPU
    CUDA_CALL( cudaFree(deviceX) );
    CUDA_CALL( cudaFree(deviceCenters) );
    CUDA_CALL( cudaFree(deviceMembership) );
    CUDA_CALL( cudaFree(deviceOldCenters) );
    CUDA_CALL( cudaFree(deviceSamplesCount) );

    // Free resources on CPU
    if (samplesCount) {
        free(samplesCount);
    }

    CUDA_CALL( cublasDestroy(cublasHandle) );
    return;
}
