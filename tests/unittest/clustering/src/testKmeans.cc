#define CATCH_CONFIG_MAIN
#include <vector>
#include <cstdlib>
#include <iostream>
#include <ctime>

#include <catch2/catch.hpp>
#include <pudding/clustering.h>

#include <helper.h>


/*
 * This test case checks kmeans using a toy dataset
 */
TEST_CASE ("Test CPU kmeans", "[kmeans-cpu]") {
    int numSamples = 4;
    int numFeatures = 2;
    int numCenters = 2;
    int maxNumIteration = 10;
    int numIterations = 0;
    float tolerance = 1e-4;
    bool cudaEnabled = false;
    std::vector<std::vector<float>> X = {{0.0, 0.0}, {0.5, 0.0}, {0.5, 1.0}, {1.0, 1.0}};
    std::vector<std::vector<float>> initCenters = {{0.0, 0.0}, {1.0, 1.0}};

    std::vector<int> expectedMemberships = {0, 0, 1, 1};
    std::vector<std::vector<float>> expectedCenters = {{0.25, 0.0}, {0.75, 1.0}};
    int expectedNumIterations = 2;

    float* centers = (float*)malloc(sizeof(float) * numCenters * numFeatures);
    int* membership = (int*)malloc(sizeof(int) * numCenters);

    kmeans(flatten(X).data(), flatten(initCenters).data(), numSamples, numFeatures, numCenters, maxNumIteration, tolerance, cudaEnabled, centers, membership, &numIterations);

    std::vector<float> vecCenters(centers, centers + (numCenters * numFeatures));
    std::vector<int> vecMembership(membership, membership + numSamples);

    REQUIRE_THAT(vecCenters, Catch::Approx(flatten(expectedCenters)));
    REQUIRE(vecMembership == expectedMemberships);
    REQUIRE(numIterations == expectedNumIterations);

    if (centers) {
        free(centers);
    }
    if (membership) {
        free(membership);
    }
}

TEST_CASE ("Test GPU kmeans", "[kmeans-gpu]") {
    int numSamples = 4;
    int numFeatures = 2;
    int numCenters = 2;
    int maxNumIteration = 10;
    int numIterations = 0;
    float tolerance = 1e-4;
    bool cudaEnabled = true;
    std::vector<std::vector<float>> X = {{0.0, 0.0}, {0.5, 0.0}, {0.5, 1.0}, {1.0, 1.0}};
    std::vector<std::vector<float>> initCenters = {{0.0, 0.0}, {1.0, 1.0}};

    std::vector<int> expectedMemberships = {0, 0, 1, 1};
    std::vector<std::vector<float>> expectedCenters = {{0.25, 0.0}, {0.75, 1.0}};
    int expectedNumIterations = 2;

    float* centers = (float*)malloc(sizeof(float) * numCenters * numFeatures);
    int* membership = (int*)malloc(sizeof(int) * numCenters);

    kmeans(flatten(X).data(), flatten(initCenters).data(), numSamples, numFeatures, numCenters, maxNumIteration, tolerance, cudaEnabled, centers, membership, &numIterations);

    std::vector<float> vecCenters(centers, centers + (numCenters * numFeatures));
    std::vector<int> vecMembership(membership, membership + numSamples);

    REQUIRE_THAT(vecCenters, Catch::Approx(flatten(expectedCenters)));
    REQUIRE(vecMembership == expectedMemberships);
    REQUIRE(numIterations == expectedNumIterations);

    if (centers) {
        free(centers);
    }
    if (membership) {
        free(membership);
    }
}

TEST_CASE ("Test GPU kmeans using the CPU version", "[kmeans-cpu-gpu]") {
    std::srand(std::time(0));

    int numSamples = 100;
    int numFeatures = 64;
    int numCenters = 10;
    int maxNumIteration = 10;
    float tolerance = 1e-4;
    std::vector<std::vector<float>> X(numSamples, std::vector<float>(numFeatures, 0.0));
    for (int i = 0; i < numSamples; i++) {
        for (int j = 0; j < numFeatures; j++) {
            X[i][j] = (float)(rand() % 100);
        }
    }
    std::vector<std::vector<float>> initCenters(numCenters, std::vector<float>(numFeatures, 0.0));
    for (int i = 0; i < numCenters; i++) {
        int idxSample = rand() % numSamples;
        for (int j = 0; j < numFeatures; j++) {
            initCenters[i][j] = X[idxSample][j];
        }
    }

    // CPU Version
    float* centersCPU = (float*)malloc(sizeof(float) * numCenters * numFeatures);
    int* membershipCPU = (int*)malloc(sizeof(int) * numCenters);
    int numIterationsCPU = 0;

    kmeans(flatten(X).data(), flatten(initCenters).data(), numSamples, numFeatures, numCenters, maxNumIteration, tolerance, false, centersCPU, membershipCPU, &numIterationsCPU);

    // std::vector<float> vecCentersCPU(centersCPU, centersCPU + (numCenters * numFeatures));
    // std::vector<int> vecMembershipCPU(membershipCPU, membershipCPU + numSamples);

    // // GPU Version
    // float* centersGPU = (float*)malloc(sizeof(float) * numCenters * numFeatures);
    // int* membershipGPU = (int*)malloc(sizeof(int) * numCenters);
    // int numIterationsGPU = 0;

    // kmeans(flatten(X).data(), flatten(initCenters).data(), numSamples, numFeatures, numCenters, maxNumIteration, tolerance, true, centersGPU, membershipGPU, &numIterationsGPU);

    // std::vector<float> vecCentersGPU(centersGPU, centersGPU + (numCenters * numFeatures));
    // std::vector<int> vecMembershipGPU(membershipGPU, membershipGPU + numSamples);

    // REQUIRE_THAT(vecCentersCPU, Catch::Approx(vecCentersGPU));
    // REQUIRE(vecMembershipCPU == vecMembershipGPU);
    // REQUIRE(numIterationsCPU == numIterationsGPU);

    // if (centersCPU) {
    //     free(centersCPU);
    // }
    // if (membershipCPU) {
    //     free(membershipCPU);
    // }
    // if (centersGPU) {
    //     free(centersGPU);
    // }
    // if (membershipGPU) {
    //     free(membershipGPU);
    // }
}