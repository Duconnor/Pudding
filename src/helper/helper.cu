#include <helper/helper.cuh>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <assert.h>

#include <helper/helper_CUDA.h>

void copyToHostAndDisplayFloat(const float* devicePtr, int row, int col) {
    float* debug = (float*)malloc(sizeof(float) * row * col);
    CUDA_CALL( cudaMemcpy(debug, devicePtr, sizeof(float) * row * col, cudaMemcpyDeviceToHost) );
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            std::cout << debug[i * col + j] << " ";
        }
        std::cout << std::endl;
    }
    if (debug) {
        free(debug);
    }
}

__global__
void matrixVectorAdditionKernel(const float* matrix, const int numRow, const int numCol, const float* vector, float scale, float* res) {
    // Each thread is responsible for each column
    int colIdx = threadIdx.x + blockIdx.x * blockDim.x;

    // Use shared memory to accelerate the computation
    extern __shared__ float sharedMem[];
    int sharedIdx = threadIdx.x;

    // Copy vector into the shared memory
    while (sharedIdx < numRow) {
        sharedMem[sharedIdx] = vector[sharedIdx];
        sharedIdx += blockDim.x;
    }
    __syncthreads();

    // Perform the subtraction
    while (colIdx < numCol) {
        for (int i = 0; i < numRow; i++) {
            res[i * numCol + colIdx] = matrix[i * numCol + colIdx] + scale * sharedMem[i];
        }

        colIdx += blockDim.x * gridDim.x;
    }
}

__global__
void vectorVectorElementWiseMultiplicationKernel(const float* vecOne, const float* vecTwo, const int numElements, const float scale, float* res) {
    // Each thread is responsible for each element
    int eleIdx = threadIdx.x + blockIdx.x * blockDim.x;

    while (eleIdx < numElements) {
        res[eleIdx] = (vecOne[eleIdx] * vecTwo[eleIdx]) * scale;

        eleIdx += blockDim.x * gridDim.x;
    }
}

__global__
void initializeAllElementsToXKernel(float* vec, const float X, const int numElements) {
    // Each thread is responsible for each element
    int eleIdx = threadIdx.x + blockIdx.x * blockDim.x;

    while (eleIdx < numElements) {
        vec[eleIdx] = X;

        eleIdx += blockDim.x * gridDim.x;
    }
    
}

/*
 * The implementation of this kernel borrows the idea from https://github.com/vincentfpgarcia/kNN-CUDA/blob/master/code/knncuda.cu.
 */
__global__
void computePairwiseEuclideanDistanceKerenl(const float* refX, const float* queryX, const int numExamplesRef, const int numExamplesQuery, const int numFeatures, float* dist) {
    // Block must be square, else won't work, reason see blow.
    assert(blockDim.x == blockDim.y);

    // This kernel uses a 2D view of threads
    // Each thread (x, y) is responsible for the result in the index of (y, x) in the dist matrix
    int distIdxX = threadIdx.y + blockDim.y * blockIdx.y;
    int distIdxY = threadIdx.x + blockDim.x * blockIdx.x;

    // Use shared memory
    extern __shared__ float sharedMem[];
    float* sharedRefX = sharedMem;
    float* sharedQueryX = sharedMem + (blockDim.x * blockDim.y);

    const int sharedMemIdxX = threadIdx.y;
    const int sharedMemIdxY = threadIdx.x;

    int numThreadsGrid = (gridDim.x * gridDim.y) * (blockDim.x * blockDim.y);
    const int distLoopRound = (numExamplesQuery * numExamplesRef) % numThreadsGrid == 0 ? (numExamplesQuery * numExamplesRef) / numThreadsGrid : (numExamplesQuery * numExamplesRef) / numThreadsGrid + 1;
    for (int i = 0; i < distLoopRound; i++) {

        /* It's kind of confusing here so I think it would be better to clarify the design here.
         *
         * When it comes to write to the dist matrix (i.e. the output matrix), (threadIdx.x, threadIdx.y) is responsible for element at index (threadIdx.y + blockDim.y * blockIdx.y, threadIdx.x + blockDim.x * blockIdx.x).
         * 
         * When it comes to load the shared memory, (theradIdx.x, threadIdx.y) is responsible for loading the element at index (threadIdx.y, threadIdx.x + blockDim.y * blockDim.y) from refX **and** the element at index (threadIdx.y, threaIdx.x + blockDim.x * blockDim.x) from queryX.
         * 
         * Therefore, the block must be a square, or else the number of queries we load won't match the number of queries we need when computing the distance. And clearly such design choice aims to enable coalesced memory access whenever we read/write to the global memory.
         */

        const int idxYRefX = threadIdx.x + blockDim.y * blockIdx.y;
        const int idxYQueryX = threadIdx.x + blockDim.x * blockIdx.x;
        int idxXRefX = threadIdx.y;
        int idxXQueryX = threadIdx.y;
        const int singleDistLoopRound = numFeatures % blockDim.x == 0 ? numFeatures / blockDim.x : numFeatures / blockDim.x + 1;

        float distAccumlator = 0.0;
        for (int j = 0; j < singleDistLoopRound; j++) {
            // Load the data from queryX and refX respectively
            sharedRefX[sharedMemIdxX * blockDim.y + sharedMemIdxY] = idxXRefX < numFeatures && idxYRefX < numExamplesRef ? refX[idxXRefX * numExamplesRef + idxYRefX] : 0;
            sharedQueryX[sharedMemIdxX * blockDim.y + sharedMemIdxY] = idxXQueryX < numFeatures && idxYQueryX < numExamplesQuery ? queryX[idxXQueryX * numExamplesQuery + idxYQueryX] : 0;

            // Wait until all threads finish loading
            __syncthreads();

            // Compute the distance (partial results)
            for (int j = 0; j < blockDim.x; j++) {
                float temp = sharedRefX[i * blockDim.y + sharedMemIdxX] - sharedQueryX[i * blockDim.y + sharedMemIdxY];
                distAccumlator += temp * temp;
            }

            // Wait for all threads to finish computation before we go to the second loop and change the value in shared memory
            __syncthreads();

            idxXRefX += blockDim.x;
            idxXQueryX += blockDim.x;
        }
        if (distIdxX < numExamplesRef && distIdxY < numExamplesQuery) {
            dist[distIdxX * numExamplesQuery + distIdxY] = distAccumlator;
        }

        distIdxX += gridDim.x * blockDim.x;
        distIdxY += gridDim.y * blockDim.y;
    }
}

void wrapperMatrixVectorAddition(const float* matrix, const int numRow, const int numCol, const float* vector, float scale, float* res) {
    // Kernel configuration
    const int BLOCKWIDTH = 1024;
    const int BLOCKSIZE = min(65535, ((numCol) + BLOCKWIDTH - 1) / BLOCKWIDTH);
    const int SHAREDMEMSIZE = sizeof(float) * numRow;
    // Launch the kernel
    matrixVectorAdditionKernel<<<BLOCKSIZE, BLOCKWIDTH, SHAREDMEMSIZE>>>(matrix, numRow, numCol, vector, scale, res);
}

void wrapperVectorVectorElementWiseMultiplication(const float* vecOne, const float* vecTwo, const int numElements, const float scale, float* res) {
    assert (scale != 0);

    // Kernel configuration
    const int BLOCKWIDTH = 1024;
    const int BLOCKSIZE = min(65535, ((numElements) + BLOCKWIDTH - 1) / BLOCKWIDTH);
    // Launch the kernel
    vectorVectorElementWiseMultiplicationKernel<<<BLOCKSIZE, BLOCKWIDTH>>>(vecOne, vecTwo, numElements, scale, res);
}

void transposeMatrix(float* matrix, const int numRow, const int numCol) {
    // cublas handle
    cublasHandle_t cublasHandle = NULL;
    CUBLAS_CALL( cublasCreate(&cublasHandle) );
    float one = 1.0, zero = 0.0;

    float* tempMatrix = NULL;
    CUDA_CALL( cudaMalloc(&tempMatrix, sizeof(float) * numRow *numCol) );
    CUDA_CALL( cudaMemcpy(tempMatrix, matrix, sizeof(float) * numRow * numCol, cudaMemcpyHostToDevice) );
    CUBLAS_CALL( cublasSgeam(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, numRow, numCol, &one, tempMatrix, numCol, &zero, tempMatrix, numRow, matrix, numRow) );

    // Free resources
    CUDA_CALL( cudaFree(tempMatrix) );
    CUBLAS_CALL( cublasDestroy(cublasHandle) );
}

void wrapperInitializeAllElementsToXKernel(float* vec, const float X, const int numElements) {
    // Kernel configuration
    const int BLOCKWIDTH = 1024;
    const int BLOCKSIZE = min(65535, ((numElements) + BLOCKWIDTH - 1) / BLOCKWIDTH);
    // Launch the kernel
    initializeAllElementsToXKernel<<<BLOCKSIZE, BLOCKWIDTH>>>(vec, X, numElements); 
}

void wrapperComputePairwiseEuclideanDistanceKerenl(const float* refX, const float* queryX, const int numExamplesRef, const int numExamplesQuery, const int numFeatures, float* dist) {
    // Kernel configuration
    const int BLOCKWIDTH = 16;
    dim3 BLOCK(BLOCKWIDTH, BLOCKWIDTH, 1);
    dim3 GRID(min(255, numExamplesQuery / BLOCKWIDTH + 1), min(255, numExamplesRef / BLOCKWIDTH + 1), 1);
    const int SHAREDMEMSIZE = sizeof(float) * BLOCKWIDTH * BLOCKWIDTH * 2;
    // Launch the kernel
    computePairwiseEuclideanDistanceKerenl<<<BLOCK, GRID, SHAREDMEMSIZE>>>(refX, queryX, numExamplesRef, numExamplesQuery, numFeatures, dist);
}
