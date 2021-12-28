#include <helper/helper.cuh>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <assert.h>
#include <float.h>

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
    // Use shared memory
    extern __shared__ float sharedMem[];
    float* sharedRefX = sharedMem;
    float* sharedQueryX = sharedMem + (blockDim.x * blockDim.y);

    const int sharedMemIdxX = threadIdx.y;
    const int sharedMemIdxY = threadIdx.x;

    const int refLoopRound = numExamplesRef % (gridDim.y * blockDim.y) == 0 ? numExamplesRef / (gridDim.y * blockDim.y) : numExamplesRef / (gridDim.y * blockDim.y) + 1;
    const int queryLoopRound = numExamplesQuery % (gridDim.x * blockDim.x) == 0 ? numExamplesQuery / (gridDim.x * blockDim.x) : numExamplesQuery / (gridDim.x * blockDim.x) + 1;

    int distIdxX = threadIdx.y + blockDim.y * blockIdx.y;
    int idxYRefX = threadIdx.x + blockDim.y * blockIdx.y;
    for (int i = 0; i < refLoopRound; i++) {

        int distIdxY = threadIdx.x + blockDim.x * blockIdx.x;
        int idxYQueryX = threadIdx.x + blockDim.x * blockIdx.x;
        for (int j = 0; j < queryLoopRound; j++) {

            /* It's kind of confusing here so I think it would be better to clarify the design here.
            *
            * When it comes to write to the dist matrix (i.e. the output matrix), (threadIdx.x, threadIdx.y) is responsible for element at index (threadIdx.y + blockDim.y * blockIdx.y, threadIdx.x + blockDim.x * blockIdx.x).
            * 
            * When it comes to load the shared memory, (theradIdx.x, threadIdx.y) is responsible for loading the element at index (threadIdx.y, threadIdx.x + blockDim.y * blockDim.y) from refX **and** the element at index (threadIdx.y, threaIdx.x + blockDim.x * blockDim.x) from queryX.
            * 
            * Therefore, the block must be a square, or else the number of queries we load won't match the number of queries we need when computing the distance. And clearly such design choice aims to enable coalesced memory access whenever we read/write to the global memory.
            */

            int idxXRefX = threadIdx.y;
            int idxXQueryX = threadIdx.y;
            const int singleDistLoopRound = numFeatures % blockDim.x == 0 ? numFeatures / blockDim.x : numFeatures / blockDim.x + 1;

            float distAccumlator = 0.0;
            for (int k = 0; k < singleDistLoopRound; k++) {
                // Load the data from queryX and refX respectively
                sharedRefX[sharedMemIdxX * blockDim.y + sharedMemIdxY] = idxXRefX < numFeatures && idxYRefX < numExamplesRef ? refX[idxXRefX * numExamplesRef + idxYRefX] : 0;
                sharedQueryX[sharedMemIdxX * blockDim.y + sharedMemIdxY] = idxXQueryX < numFeatures && idxYQueryX < numExamplesQuery ? queryX[idxXQueryX * numExamplesQuery + idxYQueryX] : 0;

                // Wait until all threads finish loading
                __syncthreads();

                // Compute the distance (partial results)
                for (int k = 0; k < blockDim.x; k++) {
                    float temp = sharedRefX[k * blockDim.y + sharedMemIdxX] - sharedQueryX[k * blockDim.y + sharedMemIdxY];
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

            distIdxY += gridDim.x * blockDim.x;
            idxYQueryX += gridDim.x * blockDim.x;
        }

        distIdxX += gridDim.y * blockDim.y;
        idxYRefX += gridDim.y * blockDim.y;
    }
}

__global__
void generateMaskVectorKernel(const int* labelVec, const int targetLabel, const int numElements, float* maskVec) {
    int idxSample = threadIdx.x + blockIdx.x * blockDim.x;

    while (idxSample < numElements) {
        maskVec[idxSample] = labelVec[idxSample] == targetLabel;

        idxSample += gridDim.x * blockDim.x;
    }
}

__global__
void applyUnaryFunctionKernel(float* vec, const int numElements, UNARY_FUNC_NAME unaryFuncName) {
    // Prepare for the function
    UNARY_FUNC_P unaryFunc;
    switch (unaryFuncName) {
        case LOG: unaryFunc = log; break;
        default: assert(false && "Unsupported Unary Function");
    }

    int idxSample = threadIdx.x + blockIdx.x * blockDim.x;

    while (idxSample < numElements) {
        vec[idxSample] = unaryFunc(vec[idxSample]);

        idxSample += gridDim.x * blockDim.x;
    }
}

__global__
void matrixArgMaxRowKernel(const float* matrix, const int numRow, const int numCol, float* maxVal, int* maxIdx) {
    /*
     * In this implementation, one block is responsible for finding the argmax of a single row.
     * Also, to support cases when the number of rows exceeds the number of blocks, we also need to iterate until all rows have been processed.
     */

    /* For each block, the shared memory is split into three parts. Specifically, for the elements covered by this block, we store:
     * 1. The index of these elements.          -> numThreadsInBlock * sizeof(int)
     * 2. The actual value of these elements.   -> numThreadsInBlock * sizeof(float)
     * 3. The current max value and its index.  -> 1 * sizeof(float) + 1 * sizeof(int)
     */
    extern __shared__ float sharedMem[];
    int* sharedIndex = (int*)sharedMem;
    float* sharedData = sharedMem + blockDim.x;
    float* sharedCurrentMaxVal = sharedData + blockDim.x;
    int* sharedCurrentMaxValIndex = (int*)sharedCurrentMaxVal + 1;

    int rowIdx = blockIdx.x; // The block idx is the row idx
    const int sharedMemIdx = threadIdx.x;

    while (rowIdx < numRow) {
        // Each loop corresponds to the reduction of a single row
        // Initialize the current max value and max index
        if (threadIdx.x == 0) {
            *sharedCurrentMaxVal = -FLT_MAX;
            *sharedCurrentMaxValIndex = -1;
        }

        const int colLoopCount = numCol % blockDim.x == 0 ? numCol / blockDim.x : numCol / blockDim.x + 1;
        int colIdx = threadIdx.x;
        for (int idxColLoop = 0; idxColLoop < colLoopCount; idxColLoop++) {
            // This corresponds to a fragment of a row reduction
            // First, load the index and the data into the shared memory
            sharedIndex[sharedMemIdx] = colIdx;
            sharedData[sharedMemIdx] = colIdx < numCol ? matrix[rowIdx * numCol + colIdx] : -FLT_MAX;
            __syncthreads();

            // Second, use a for loop to perform reduction
            int range = blockDim.x;
            for (int i = 0; i < (int)log2((float)blockDim.x); i++) {
                range /= 2;
                if (sharedMemIdx < range) {
                    sharedIndex[sharedMemIdx] = sharedData[sharedMemIdx] < sharedData[sharedMemIdx + range] ? sharedIndex[sharedMemIdx + range] : sharedIndex[sharedMemIdx];
                    sharedData[sharedMemIdx] = sharedData[sharedMemIdx] < sharedData[sharedMemIdx + range] ? sharedData[sharedMemIdx + range] : sharedData[sharedMemIdx];
                }
                __syncthreads();
            }

            // The first thread in this block is responsible for updating the temporary max value and index
            if (threadIdx.x == 0) {
                *sharedCurrentMaxValIndex = *sharedCurrentMaxVal < sharedData[sharedMemIdx] ? sharedIndex[sharedMemIdx] : *sharedCurrentMaxValIndex;
                *sharedCurrentMaxVal = *sharedCurrentMaxVal < sharedData[sharedMemIdx] ? sharedData[sharedMemIdx] : *sharedCurrentMaxVal;
            }
            // No need to syncthreads here because the critical area is only accessible for the first thread

            colIdx += blockDim.x;
        }
        // The first thread in this block is responsible for writting the final max val and index to the global memory
        if (threadIdx.x == 0) {
            maxIdx[rowIdx] = *sharedCurrentMaxValIndex;
            maxVal[rowIdx] = *sharedCurrentMaxVal;
        }
        // No need to syncthreads here because sharedCurrentMaxValIndex is only accessible by the first thread, which is now writing to the global memory

        rowIdx += gridDim.x;
    }

}

void wrapperMatrixVectorAddition(const float* matrix, const int numRow, const int numCol, const float* vector, float scale, float* res) {
    // Kernel configuration
    const int BLOCKWIDTH = 1024;
    const int BLOCKSIZE = min(65535, ((numCol) + BLOCKWIDTH - 1) / BLOCKWIDTH);
    const int SHAREDMEMSIZE = sizeof(float) * numRow;
    // Check whether we have enough shared memory
    if (SHAREDMEMSIZE > MAXSHAREDMEMBYTES) {
        assert(false && "No enough shared memory");
    }
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
    CUDA_CALL( cudaMemcpy(tempMatrix, matrix, sizeof(float) * numRow * numCol, cudaMemcpyDeviceToDevice) );
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
    // Check whether we have enough shared memory
    if (SHAREDMEMSIZE > MAXSHAREDMEMBYTES) {
        assert(false && "No enough shared memory");
    }
    // Launch the kernel
    computePairwiseEuclideanDistanceKerenl<<<GRID, BLOCK, SHAREDMEMSIZE>>>(refX, queryX, numExamplesRef, numExamplesQuery, numFeatures, dist);
}

void wrapperGenerateMaskVectorKernel(const int* labelVec, const int targetLabel, const int numElements, float* maskVec) {
    // Kernel configuration
    const int BLOCKWIDTH = 1024;
    const int NUMBLOCK = min(65535, ((numElements) + BLOCKWIDTH - 1) / BLOCKWIDTH);
    // Launch the kernel
    generateMaskVectorKernel<<<NUMBLOCK, BLOCKWIDTH>>>(labelVec, targetLabel, numElements, maskVec);
}

void wrapperApplyUnaryFunctionKernel(float* vec, const int numElements, UNARY_FUNC_NAME unaryFuncName) {
    // Kernel configuration
    const int BLOCKWIDTH = 1024;
    const int NUMBLOCK = min(65535, ((numElements) + BLOCKWIDTH - 1) / BLOCKWIDTH);
    // Launch the kernel
    applyUnaryFunctionKernel<<<NUMBLOCK, BLOCKWIDTH>>>(vec, numElements, unaryFuncName);
}

void wrapperMatrixArgMaxRowKernel(const float* matrix, const int numRow, const int numCol, float* maxVal, int* maxIdx) {
    // Kernel configuration
    const int BLOCKWIDTH = 1024;
    const int NUMBLOCK = min(65535, numRow);
    const int SHAREDMEMSIZE = sizeof(int) * BLOCKWIDTH + sizeof(float) * BLOCKWIDTH + sizeof(float) + sizeof(int);
    // Check whether we have enough shared memory
    if (SHAREDMEMSIZE > MAXSHAREDMEMBYTES) {
        assert(false &* "No enough shared memory");
    }
    // Launch the kernel
    matrixArgMaxRowKernel<<<NUMBLOCK, BLOCKWIDTH, SHAREDMEMSIZE>>>(matrix, numRow, numCol, maxVal, maxIdx);
}
