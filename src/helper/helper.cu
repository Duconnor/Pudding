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
void matrixVectorSubtractionKernel(const float* matrix, const int numRow, const int numCol, const float* vector, float* res) {
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
            res[i * numCol + colIdx] = matrix[i * numCol + colIdx] - sharedMem[i];
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

void wrapperMatrixVectorSubtraction(const float* matrix, const int numRow, const int numCol, const float* vector, float* res) {
    // Kernel configuration
    const int BLOCKWIDTH = 1024;
    const int BLOCKSIZE = min(65535, ((numCol) + BLOCKWIDTH - 1) / BLOCKWIDTH);
    const int SHAREDMEMSIZE = sizeof(float) * numRow;
    // Launch the kernel
    matrixVectorSubtractionKernel<<<BLOCKSIZE, BLOCKWIDTH, SHAREDMEMSIZE>>>(matrix, numRow, numCol, vector, res);
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
