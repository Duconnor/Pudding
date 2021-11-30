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

TEST_CASE ("Test vector vector element wise multiplication", "[vector-vector-element-wise-multiplication]") {
    // Prepare the test data
    const int numElements = 6;
    std::vector<float> vecOne = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    std::vector<float> vecTwo = {7.0, 8.0, 9.0, 10.0, 11.0, 12.0};
    float scale = 2.0;

    std::vector<float> expectedRes = {14.0, 32.0, 54.0, 80.0, 110.0, 144.0};

    // Copy data to device
    float* deviceVecOne = NULL;
    float* deviceVecTwo = NULL;
    float* deviceRes = NULL;

    CUDA_CALL( cudaMalloc(&deviceVecOne, sizeof(float) * numElements) );
    CUDA_CALL( cudaMalloc(&deviceVecTwo, sizeof(float) * numElements) );
    CUDA_CALL( cudaMalloc(&deviceRes, sizeof(float) * numElements) );

    CUDA_CALL( cudaMemcpy(deviceVecOne, vecOne.data(), sizeof(float) * numElements, cudaMemcpyHostToDevice) );
    CUDA_CALL( cudaMemcpy(deviceVecTwo, vecTwo.data(), sizeof(float) * numElements, cudaMemcpyHostToDevice) );

    // Launche the kernel
    wrapperVectorVectorElementWiseMultiplication(deviceVecOne, deviceVecTwo, numElements, scale, deviceRes);

    // Copy data back to host
    float* res = (float*)malloc(sizeof(float) * numElements);
    CUDA_CALL( cudaMemcpy(res, deviceRes, sizeof(float) * numElements, cudaMemcpyDeviceToHost) );

    // Assertions
    std::vector<float> vecRes(res, res + numElements);
    REQUIRE_THAT(vecRes, Catch::Approx(expectedRes));

    if (res) {
        free(res);
    }
    // Free all resources
    CUDA_CALL( cudaFree(deviceVecOne) );
    CUDA_CALL( cudaFree(deviceVecTwo) );
    CUDA_CALL( cudaFree(deviceRes) );
}

TEST_CASE ("Test matrix transpose", "[matrix-transpose]") {
    // Prepare the test data
    const int numElements = 12;
    std::vector<float> matrix = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0};

    std::vector<float> expectedRes = {1.0, 4.0, 7.0, 10.0, 2.0, 5.0, 8.0, 11.0, 3.0, 6.0, 9.0, 12.0};

    // Copy data to device
    float* deviceMatrix = NULL;

    CUDA_CALL( cudaMalloc(&deviceMatrix, sizeof(float) * numElements) );

    CUDA_CALL( cudaMemcpy(deviceMatrix, matrix.data(), sizeof(float) * numElements, cudaMemcpyHostToDevice) );

    // Launche the kernel
    transposeMatrix(deviceMatrix, 4, 3);

    // Copy data back to host
    float* res = (float*)malloc(sizeof(float) * numElements);
    CUDA_CALL( cudaMemcpy(res, deviceMatrix, sizeof(float) * numElements, cudaMemcpyDeviceToHost) );

    // Assertions
    std::vector<float> vecRes(res, res + numElements);
    REQUIRE_THAT(vecRes, Catch::Approx(expectedRes));

    if (res) {
        free(res);
    }
    // Free all resources
    CUDA_CALL( cudaFree(deviceMatrix) );
}

TEST_CASE ("Test array initialization", "[array-initialization]") {
    // Prepare the test data
    const int numElements = 12;
    const float targetValue = 233;
    std::vector<float> vec(numElements, 0);

    std::vector<float> expectedRes(numElements, targetValue);

    // Copy data to device
    float* deviceVec = NULL;

    CUDA_CALL( cudaMalloc(&deviceVec, sizeof(float) * numElements) );

    CUDA_CALL( cudaMemcpy(deviceVec, vec.data(), sizeof(float) * numElements, cudaMemcpyHostToDevice) );

    // Launche the kernel
    wrapperInitializeAllElementsToXKernel(deviceVec, targetValue, numElements);

    // Copy data back to host
    float* res = (float*)malloc(sizeof(float) * numElements);
    CUDA_CALL( cudaMemcpy(res, deviceVec, sizeof(float) * numElements, cudaMemcpyDeviceToHost) );

    // Assertions
    std::vector<float> vecRes(res, res + numElements);
    REQUIRE_THAT(vecRes, Catch::Approx(expectedRes));

    if (res) {
        free(res);
    }
    // Free all resources
    CUDA_CALL( cudaFree(deviceVec) );
}
