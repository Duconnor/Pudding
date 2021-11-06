#define CATCH_CONFIG_MAIN
#include <vector>
#include <cstdlib>
#include <iostream>

#include <catch2/catch.hpp>
#include <pudding/clustering.h>

#include <helper.h>


/*
 * This test case checks kmeans using a toy dataset
 */
TEST_CASE ("Test CPU kmeans", "[kmeans]") {
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

TEST_CASE ("Test GPU kmeans", "[kmeans]") {
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