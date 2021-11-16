#include <assert.h>
#include <math.h>
#include <float.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <helper/helper.cuh>
#include <helper/helperCUDA.h>

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
    if (threadIdx.x < numCenters) {
        for (int idxFeature = 0; idxFeature < numFeatures; idxFeature++) {
            sharedCenters[idxFeature * numCenters + threadIdx.x] = centers[idxFeature * numCenters + threadIdx.x];
        }
    }
    __syncthreads();

    while (idxSample < numSamples) {
        // Load this block's data into the shared memory
        for (int idxFeature = 0; idxFeature < numFeatures; idxFeature++) {
            sharedX[idxFeature * numSamplesThisBlock + idxSampleSharedMem] = X[idxFeature * numSamples + idxSample];
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
        membership[idxSample] = minDistIdx;
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
                    centers[j * numCenters + idxCenter] += X[j * numSamples + i];
                }
                numSamplesThisCenter[idxCenter]++;
            }
        }

        for (int j = 0; j < numFeatures; j++) {
            centers[j * numCenters + idxCenter] /= numSamplesThisCenter[idxCenter];
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

    // Determine the block width
    const int BLOCKWIDTH = 1024;

    // Initialize the cublas handle
    cublasHandle_t cublasHandle;
    cublasCreate(&cublasHandle);

    // Transpose deviceX, deviceCenters here to enable coalesced memory access in the kernel
    float one = 1.0, zero = 0.0;

    // Allocate temporary array here for transpose
    float* tempDeviceX;
    float* tempDeviceCenters;

    CUDA_CALL( cudaMalloc(&tempDeviceX, sizeof(float) * numSamples * numFeatures) );
    CUDA_CALL( cudaMalloc(&tempDeviceCenters, sizeof(float) * numCenters * numFeatures) );

    CUDA_CALL( cudaMemcpy(tempDeviceX, X, sizeof(float) * numSamples * numFeatures, cudaMemcpyHostToDevice) );
    CUDA_CALL( cudaMemcpy(tempDeviceCenters, centers, sizeof(float) * numCenters * numFeatures, cudaMemcpyHostToDevice) );

    CUBLAS_CALL( cublasSgeam(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, numSamples, numFeatures, &one, tempDeviceX, numFeatures, &zero, tempDeviceX, numSamples, deviceX, numSamples) );
    CUBLAS_CALL( cublasSgeam(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, numCenters, numFeatures, &one, tempDeviceCenters, numFeatures, &zero, tempDeviceCenters, numCenters, deviceCenters, numCenters) );

    while (!endFlag) {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        float ellapsed = 0.0;
        
        // Determine the membership of each sample
        int numBlock = min(65535, ((numSamples) + BLOCKWIDTH - 1) / BLOCKWIDTH);
        int numBytesSharedMemory = BLOCKWIDTH * sizeof(float) * numFeatures + sizeof(float) * numCenters * numFeatures;
        cudaEventRecord(start, 0);
        
        determineMembershipKernel<<<numBlock, BLOCKWIDTH, numBytesSharedMemory>>>(deviceX, deviceCenters, deviceMembership, numSamples, numFeatures, numCenters);

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        ellapsed = 0.0;
        cudaEventElapsedTime(&ellapsed, start, stop);
        std::cout << "Compute membership: " << ellapsed << std::endl;  

        // Save the result of old centers
        CUDA_CALL( cudaMemcpy(deviceOldCenters, deviceCenters, sizeof(float) * numCenters * numFeatures, cudaMemcpyDeviceToDevice) );
        CUDA_CALL( cudaMemset(deviceCenters, 0, sizeof(float) * numCenters * numFeatures));

        // Update the center estimation
        CUDA_CALL( cudaMemset(deviceNumSamplesThisCenter, 0, sizeof(int) * numSamples) );
        numBlock = min(65535, ((numCenters) + BLOCKWIDTH - 1) / BLOCKWIDTH);

        cudaEventRecord(start, 0);

        updateCentersKernel<<<numBlock, BLOCKWIDTH>>>(deviceX, deviceMembership, deviceCenters, deviceNumSamplesThisCenter, numSamples, numFeatures, numCenters);

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        ellapsed = 0.0;
        cudaEventElapsedTime(&ellapsed, start, stop);
        std::cout << "Update center: " << ellapsed << std::endl;

        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        // Test for coverage
        iterationCount++;
        if (iterationCount >= maxNumIteration) {
            endFlag = true;
        } else {
            float negOne = -1.0;
            // Careful here, cuBlas assumes column major storage
            // Perform element-wise subtraction
            CUBLAS_CALL( cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, numCenters, numFeatures, &one, deviceOldCenters, numCenters, &negOne, deviceCenters, numCenters, deviceOldCenters, numCenters) );
            float diff = 0.0;
            // Compute the F-norm
            CUBLAS_CALL( cublasSdot(cublasHandle, numCenters * numFeatures, deviceOldCenters, 1, deviceOldCenters, 1, &diff) );
            endFlag = sqrt(diff) < tolerance;
        }
    }

    // Copy the result back to host
    // Tranpose the deviceCenters back
    CUBLAS_CALL( cublasSgeam(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, numFeatures, numCenters, &one, deviceCenters, numCenters, &zero, deviceCenters, numFeatures, tempDeviceCenters, numFeatures) );
    float* temp = tempDeviceCenters;
    tempDeviceCenters = deviceCenters;
    deviceCenters = temp;

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