#include <helper/helper_CUDA.h>
#include <helper/helper.cuh>

#include <cuda_runtime.h>

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

void wrapperMatrixVectorSubtraction(const float* matrix, const int numRow, const int numCol, const float* vector, float* res) {
    // Kernel configuration
    int BLOCKWIDTH = 1024;
    int BLOCKSIZE = min(65535, ((numCol) + BLOCKWIDTH - 1) / BLOCKWIDTH);
    int SHAREDMEMSIZE = sizeof(float) * numRow;
    // Launch the kernel
    matrixVectorSubtractionKernel<<<BLOCKSIZE, BLOCKWIDTH, SHAREDMEMSIZE>>>(matrix, numRow, numCol, vector, res);
}
