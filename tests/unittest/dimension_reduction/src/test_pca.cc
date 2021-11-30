#define CATCH_CONFIG_MAIN
#include <vector>
#include <cstdlib>
#include <iostream>

#include <catch2/catch.hpp>
#include <pudding/dimension_reduction.h>

#include <helper.h>

#define min(a, b) a > b ? b : a

void checkPCAResult(std::vector<float>& variances, std::vector<float>& principalComponents, std::vector<float>& principalAxes, std::vector<float>& expectedVariances, std::vector<float>& expectedPrincipalComponents, std::vector<float>& expectedPrincipalAxes) {
    
    const int numComponents = variances.size();
    const int numFeatures = principalAxes.size() / numComponents;
    const int numSamples = principalComponents.size() / numComponents;

    for (int componentIdx = 0; componentIdx < numComponents; componentIdx++) {
        // We need to perform check for individual axis
        // First, we extract the corresponding data out
        std::vector<float> axisThisComponent = std::vector<float>(principalAxes.begin() + (componentIdx * numFeatures), principalAxes.begin() + (componentIdx * numFeatures + numFeatures));
        std::vector<float> expectedAxisThisComponent = std::vector<float>(expectedPrincipalAxes.begin() + (componentIdx * numFeatures), expectedPrincipalAxes.begin() + (componentIdx * numFeatures + numFeatures));
        std::vector<float> expectedAxisThisComponentNeg = neg(expectedAxisThisComponent);
        std::vector<float> valThisComponent(numSamples, 0);
        std::vector<float> expectedValThisComponent(numSamples, 0);
        for (int i = 0; i < numSamples; i++) {
            valThisComponent[i] = principalComponents[i * numComponents + componentIdx];
            expectedValThisComponent[i] = expectedPrincipalComponents[i * numComponents + componentIdx];
        }
        std::vector<float> expectedValThisComponentNeg = neg(expectedValThisComponent);

        // Perform check
        REQUIRE_THAT(axisThisComponent, Catch::Approx(expectedAxisThisComponent) || Catch::Approx(expectedAxisThisComponentNeg));
        if (abs(axisThisComponent[0] - expectedAxisThisComponent[0]) < 1e-4) {
            REQUIRE_THAT(valThisComponent, Catch::Approx(expectedValThisComponent));
        } else {
            REQUIRE_THAT(valThisComponent, Catch::Approx(expectedValThisComponentNeg));
        }
    }
    
    REQUIRE_THAT(variances, Catch::Approx(expectedVariances));
}


/*
 * This test case test PCA using a simple toy data
 * In this test case, no dimension reduction is needed
 */
TEST_CASE ("Test PCA on toy data one", "[pca-toy-one]") {
    const int numSamples = 6;
    const int numFeatures = 2;
    const int numComponents = 2;
    const float variancePercentage = 0;

    std::vector<float> X = {-1, -1, -2, -1, -3, -2, 1, 1, 2, 1, 3, 2};

    std::vector<float> expectedPrincipalComponents = {1.38340578, 0.2935787, 2.22189802, -0.25133484, 3.6053038, 0.04224385, -1.38340578, -0.2935787, -2.22189802, 0.25133484, -3.6053038, -0.04224385};
    std::vector<float> expectedPrincipalComponentsNeg = neg(expectedPrincipalComponents); // Element-wise negation of the expectedPrincipalComponents
    std::vector<float> expectedPrincipalAxes = {-0.83849224, -0.54491354, 0.54491354, -0.83849224};
    std::vector<float> expectedPrincipalAxesNeg = neg(expectedPrincipalAxes);
    std::vector<float> expectedVariances = {7.93954312, 0.06045688};
    std::vector<float> expectedReconstructedX = {-1, -1, -2, -1, -3, -2, 1, 1, 2, 1, 3, 2};
    int expectedNumComponentsChosen = 2;

    // The problem here is that we may not know in advance how many components will be chosen. Therefore, here we allocate space for the largest possible case.
    float* principalComponents = (float*)malloc(sizeof(float) * numSamples * numFeatures);
    float* principalAxes = (float*)malloc(sizeof(float) * min(numFeatures, numSamples) * numFeatures);
    float* variances = (float*)malloc(sizeof(float) * min(numFeatures, numSamples));
    float* reconstructedX = (float*)malloc(sizeof(float) * numSamples * numFeatures);
    int numComponentsChosen = 0;

    pca(X.data(), numSamples, numFeatures, numComponents, variancePercentage, principalComponents, principalAxes, variances, reconstructedX, &numComponentsChosen);

    // This must be correct cause we will be recovering the actual principal components, principal axes and variances based on this value
    REQUIRE(numComponentsChosen == expectedNumComponentsChosen);

    std::vector<float> vecPrincipalComponents(principalComponents, principalComponents + (numSamples * numComponentsChosen));
    std::vector<float> vecPrincipalAxes(principalAxes, principalAxes + (numFeatures * numComponentsChosen));
    std::vector<float> vecVariances(variances, variances + numComponentsChosen);
    std::vector<float> vecReconstructedX(reconstructedX, reconstructedX + (numSamples * numFeatures));

    checkPCAResult(vecVariances, vecPrincipalComponents, vecPrincipalAxes, expectedVariances, expectedPrincipalComponents, expectedPrincipalAxes);
    REQUIRE_THAT(vecReconstructedX, Catch::Approx(expectedReconstructedX));
}

/*
 * This test case test PCA using a simple toy data
 * In this test case, dimension reduction happens
 */
TEST_CASE ("Test PCA on toy data two", "[pca-toy-two]") {
    const int numSamples = 6;
    const int numFeatures = 2;
    const int numComponents = 1;
    const float variancePercentage = 0;

    std::vector<float> X = {-1, -1, -2, -1, -3, -2, 1, 1, 2, 1, 3, 2};

    std::vector<float> expectedPrincipalComponents = {1.38340578, 2.22189802, 3.6053038, -1.38340578, -2.22189802, -3.6053038};
    std::vector<float> expectedPrincipalComponentsNeg = neg(expectedPrincipalComponents);
    std::vector<float> expectedPrincipalAxes = {-0.83849224, -0.54491354};
    std::vector<float> expectedPrincipalAxesNeg = neg(expectedPrincipalAxes);
    std::vector<float> expectedVariances = {7.93954312};
    std::vector<float> expectedReconstructedX = {-1.15997501, -0.75383654, -1.86304424, -1.21074232, -3.02301925, -1.96457886, 1.15997501, 0.75383654, 1.86304424, 1.21074232, 3.02301925, 1.96457886};
    int expectedNumComponentsChosen = 1;

    // The problem here is that we may not know in advance how many components will be chosen. Therefore, here we allocate space for the largest possible case.
    float* principalComponents = (float*)malloc(sizeof(float) * numSamples * numFeatures);
    float* principalAxes = (float*)malloc(sizeof(float) * min(numFeatures, numSamples) * numFeatures);
    float* variances = (float*)malloc(sizeof(float) * min(numFeatures, numSamples));
    float* reconstructedX = (float*)malloc(sizeof(float) * numSamples * numFeatures);
    int numComponentsChosen = 0;

    pca(X.data(), numSamples, numFeatures, numComponents, variancePercentage, principalComponents, principalAxes, variances, reconstructedX, &numComponentsChosen);

    // This must be correct cause we will be recovering the actual principal components, principal axes and variances based on this value
    REQUIRE(numComponentsChosen == expectedNumComponentsChosen);

    std::vector<float> vecPrincipalComponents(principalComponents, principalComponents + (numSamples * numComponentsChosen));
    std::vector<float> vecPrincipalAxes(principalAxes, principalAxes + (numFeatures * numComponentsChosen));
    std::vector<float> vecVariances(variances, variances + numComponentsChosen);
    std::vector<float> vecReconstructedX(reconstructedX, reconstructedX + (numSamples * numFeatures));

    checkPCAResult(vecVariances, vecPrincipalComponents, vecPrincipalAxes, expectedVariances, expectedPrincipalComponents, expectedPrincipalAxes);
    REQUIRE_THAT(vecReconstructedX, Catch::Approx(expectedReconstructedX));
}

/*
 * This test case test PCA using a simple toy data
 * In this test case, the number of components is chosen based on the variance percentage required
 */
TEST_CASE ("Test PCA on toy data auto selection of components", "[pca-toy-auto-selections]") {
    const int numSamples = 6;
    const int numFeatures = 2;
    const int numComponents = -1;

    std::vector<float> X = {-1, -1, -2, -1, -3, -2, 1, 1, 2, 1, 3, 2};

    // The problem here is that we may not know in advance how many components will be chosen. Therefore, here we allocate space for the largest possible case.
    float* principalComponents = (float*)malloc(sizeof(float) * numSamples * numFeatures);
    float* principalAxes = (float*)malloc(sizeof(float) * min(numFeatures, numSamples) * numFeatures);
    float* variances = (float*)malloc(sizeof(float) * min(numFeatures, numSamples));
    float* reconstructedX = (float*)malloc(sizeof(float) * numSamples * numFeatures);
    int numComponentsChosen = 0;

    // First case, only one principal component (scores) is enough
    float variancePercentage = 0.9;

    std::vector<float> expectedPrincipalComponents = {1.38340578, 2.22189802, 3.6053038, -1.38340578, -2.22189802, -3.6053038};
    std::vector<float> expectedPrincipalComponentsNeg = neg(expectedPrincipalComponents);
    std::vector<float> expectedPrincipalAxes = {-0.83849224, -0.54491354};
    std::vector<float> expectedPrincipalAxesNeg = neg(expectedPrincipalAxes);
    std::vector<float> expectedVariances = {7.93954312};
    std::vector<float> expectedReconstructedX = {-1.15997501, -0.75383654, -1.86304424, -1.21074232, -3.02301925, -1.96457886, 1.15997501, 0.75383654, 1.86304424, 1.21074232, 3.02301925, 1.96457886};
    int expectedNumComponentsChosen = 1;

    pca(X.data(), numSamples, numFeatures, numComponents, variancePercentage, principalComponents, principalAxes, variances, reconstructedX, &numComponentsChosen);

    // This must be correct cause we will be recovering the actual principal components, principal axes and variances based on this value
    REQUIRE(numComponentsChosen == expectedNumComponentsChosen);

    std::vector<float> vecPrincipalComponents(principalComponents, principalComponents + (numSamples * numComponentsChosen));
    std::vector<float> vecPrincipalAxes(principalAxes, principalAxes + (numFeatures * numComponentsChosen));
    std::vector<float> vecVariances(variances, variances + numComponentsChosen);
    std::vector<float> vecReconstructedX(reconstructedX, reconstructedX + (numSamples * numFeatures));

    checkPCAResult(vecVariances, vecPrincipalComponents, vecPrincipalAxes, expectedVariances, expectedPrincipalComponents, expectedPrincipalAxes);
    REQUIRE_THAT(vecReconstructedX, Catch::Approx(expectedReconstructedX));

    // Second case, two principal components are all needed
    variancePercentage = 0.999;

    expectedPrincipalComponents = {1.38340578, 0.2935787, 2.22189802, -0.25133484, 3.6053038, 0.04224385, -1.38340578, -0.2935787, -2.22189802, 0.25133484, -3.6053038, -0.04224385};
    expectedPrincipalComponentsNeg = neg(expectedPrincipalComponents); // Element-wise negation of the expectedPrincipalComponents
    expectedPrincipalAxes = {-0.83849224, -0.54491354, 0.54491354, -0.83849224};
    expectedPrincipalAxesNeg = neg(expectedPrincipalAxes);
    expectedVariances = {7.93954312, 0.06045688};
    expectedReconstructedX = {-1, -1, -2, -1, -3, -2, 1, 1, 2, 1, 3, 2};
    expectedNumComponentsChosen = 2;

    pca(X.data(), numSamples, numFeatures, numComponents, variancePercentage, principalComponents, principalAxes, variances, reconstructedX, &numComponentsChosen);

    // This must be correct cause we will be recovering the actual principal components, principal axes and variances based on this value
    REQUIRE(numComponentsChosen == expectedNumComponentsChosen);

    vecPrincipalComponents = std::vector<float>(principalComponents, principalComponents + (numSamples * numComponentsChosen));
    vecPrincipalAxes = std::vector<float>(principalAxes, principalAxes + (numFeatures * numComponentsChosen));
    vecVariances = std::vector<float>(variances, variances + numComponentsChosen);
    vecReconstructedX = std::vector<float>(reconstructedX, reconstructedX + (numSamples * numFeatures));

    checkPCAResult(vecVariances, vecPrincipalComponents, vecPrincipalAxes, expectedVariances, expectedPrincipalComponents, expectedPrincipalAxes);
    REQUIRE_THAT(vecReconstructedX, Catch::Approx(expectedReconstructedX));
}