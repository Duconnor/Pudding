#define CATCH_CONFIG_MAIN
#include <cstdlib>
#include <vector>

#include <catch2/catch.hpp>
#include <cuda_runtime.h>

#include <helper/helper.cuh>
#include <helper/helper_CUDA.h>

TEST_CASE ("Test matrix vector subtraction kernel", "[matrix-vector-subtraction]") {
    // Prepare the test data
    const int numRow = 3;
    const int numCol = 4;
    std::vector<float> matrix = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0};
    std::vector<float> vector = {5.0, 5.0, 5.0};

    std::vector<float> expectedRes = {-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0};

    // Copy data to device
    float* deviceMatrix = NULL;
    float* deviceVector = NULL;
    float* deviceRes = NULL;

    CUDA_CALL( cudaMalloc(&deviceMatrix, sizeof(float) * numRow * numCol) );
    CUDA_CALL( cudaMalloc(&deviceVector, sizeof(float) * numRow) );
    CUDA_CALL( cudaMalloc(&deviceRes, sizeof(float) * numRow * numCol) );

    CUDA_CALL( cudaMemcpy(deviceMatrix, matrix.data(), sizeof(float) * numRow * numCol, cudaMemcpyHostToDevice) );
    CUDA_CALL( cudaMemcpy(deviceVector, vector.data(), sizeof(float) * numRow, cudaMemcpyHostToDevice) );

    // Launche the kernel
    wrapperMatrixVectorSubtraction(deviceMatrix, numRow, numCol, deviceVector, deviceRes);

    // Copy data back to host
    float* res = (float*)malloc(sizeof(float) * numRow * numCol);
    CUDA_CALL( cudaMemcpy(res, deviceRes, sizeof(float) * numRow * numCol, cudaMemcpyDeviceToHost) );

    // Assertions
    std::vector<float> vecRes(res, res + (numRow * numCol));
    REQUIRE_THAT(vecRes, Catch::Approx(expectedRes));

    if (res) {
        free(res);
    }
    // Free all resources
    CUDA_CALL( cudaFree(deviceMatrix) );
    CUDA_CALL( cudaFree(deviceVector) );
    CUDA_CALL( cudaFree(deviceRes) );
}