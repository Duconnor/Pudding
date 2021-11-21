#include <helper/helper_CUDA.h>

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