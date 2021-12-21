#define CATCH_CONFIG_MAIN
#include <cstdlib>
#include <vector>

#include <catch2/catch.hpp>
#include <cuda_runtime.h>

#include <helper/helper.cuh>
#include <helper/helper_CUDA.h>

#include <helper.h>

TEST_CASE ("Test matrix vector addition kernel", "[matrix-vector-addition]") {
    // Prepare the test data
    const int numRow = 3;
    const int numCol = 4;
    std::vector<float> matrix = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0};
    std::vector<float> vector = {5.0, 5.0, 5.0};
    float scale = -1.0;

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
    wrapperMatrixVectorAddition(deviceMatrix, numRow, numCol, deviceVector, scale, deviceRes);

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

TEST_CASE ("Test pair wise euclidean distance computation", "[pair-wise-euclidean-distance]") {
    // Prepare the test data
    const int numExamplesRef = 3;
    const int numExamplesQuery = 2;
    const int numFeatures = 2;

    std::vector<std::vector<float>> refX = {{0, 2, -1}, {1, 1, -2}};
    std::vector<std::vector<float>> queryX = {{0, 2}, {1, -1}};

    std::vector<std::vector<float>> expectedDist = {{0, 8}, {4, 4}, {10, 10}};

    // Copy data to device
    float* deviceRefX;
    float* deviceQueryX;

    CUDA_CALL( cudaMalloc(&deviceRefX, sizeof(float) * numFeatures * numExamplesRef) );
    CUDA_CALL( cudaMalloc(&deviceQueryX, sizeof(float) * numFeatures * numExamplesQuery) );

    CUDA_CALL( cudaMemcpy(deviceRefX, flatten(refX).data(), sizeof(float) * numFeatures * numExamplesRef, cudaMemcpyHostToDevice) );
    CUDA_CALL( cudaMemcpy(deviceQueryX, flatten(queryX).data(), sizeof(float) * numFeatures * numExamplesQuery, cudaMemcpyHostToDevice) );

    // Prepare for output
    float* deviceDist;

    CUDA_CALL( cudaMalloc(&deviceDist, sizeof(float) * numExamplesRef * numExamplesQuery) );

    // Call the function
    wrapperComputePairwiseEuclideanDistanceKerenl(deviceRefX, deviceQueryX, numExamplesRef, numExamplesQuery, numFeatures, deviceDist);

    // Copy the output back to host
    float* dist = (float*)malloc(sizeof(float) * numExamplesRef * numExamplesQuery);

    CUDA_CALL( cudaMemcpy(dist, deviceDist, sizeof(float) * numExamplesRef * numExamplesQuery, cudaMemcpyDeviceToHost) );

    std::vector<float> vecDist(dist, dist + (numExamplesRef * numExamplesQuery));

    // Check
    REQUIRE_THAT(vecDist, Catch::Approx(flatten(expectedDist)));

    // Free resources
    CUDA_CALL( cudaFree(deviceRefX) );
    CUDA_CALL( cudaFree(deviceQueryX) );
    if (dist) {
        free(dist);
    }
}

TEST_CASE ("Test mask vector generation", "[mask-generation]") {
    // Prepare the test data
    const int numElements = 8;
    const int targetLabel = 2;
    std::vector<int> labelVec = {0, 1, 2, 1, 2, 0, 2, 10};

    std::vector<float> expectedMaskVec = {0, 0, 1, 0, 1, 0, 1, 0};

    // Copy data to device and also prepare space for result
    int* deviceLabelVec = NULL;
    float* deviceMaskVec = NULL;

    CUDA_CALL( cudaMalloc(&deviceLabelVec, sizeof(int) * numElements) );
    CUDA_CALL( cudaMalloc(&deviceMaskVec, sizeof(float) * numElements) );

    CUDA_CALL( cudaMemcpy(deviceLabelVec, labelVec.data(), sizeof(int) * numElements, cudaMemcpyHostToDevice) );

    // Launche the kernel
    wrapperGenerateMaskVectorKernel(deviceLabelVec, targetLabel, numElements, deviceMaskVec);

    // Copy data back to host
    float* maskVec = (float*)malloc(sizeof(float) * numElements);
    CUDA_CALL( cudaMemcpy(maskVec, deviceMaskVec, sizeof(float) * numElements, cudaMemcpyDeviceToHost) );

    // Assertions
    std::vector<float> vecMaskVec(maskVec, maskVec + numElements);
    REQUIRE_THAT(vecMaskVec, Catch::Approx(expectedMaskVec));

    if (maskVec) {
        free(maskVec);
    }
    // Free all resources
    CUDA_CALL( cudaFree(deviceLabelVec) );
    CUDA_CALL( cudaFree(deviceMaskVec) );
}
