#define CATCH_CONFIG_MAIN
#include <vector>
#include <cstdlib>
#include <iostream>

#include <catch2/catch.hpp>
#include <pudding/estimation.h>

#include <helper.h>

TEST_CASE ("Test gaussian KDE score toy data", "[gaussian-kde-score-toy-data]") {
    // Prepare for the data
    const int numSamples = 10;
    const int numFeatures = 3;
    const std::string kernel = "gaussian";
    const float bandwidth = 0.5;
    const int numTestSamples = 3;

    std::vector<std::vector<float>> X = {{0.37454012, 0.95071431, 0.73199394}, {0.59865848, 0.15601864, 0.15599452}, {0.05808361, 0.86617615, 0.60111501}, {0.70807258, 0.02058449, 0.96990985}, {0.83244264, 0.21233911, 0.18182497}, {0.18340451, 0.30424224, 0.52475643}, {0.43194502, 0.29122914, 0.61185289}, {0.13949386, 0.29214465, 0.36636184}, {0.45606998, 0.78517596, 0.19967378}, {0.51423444, 0.59241457, 0.04645041}};
    std::vector<std::vector<float>> samplesX = {{0.37454012, 0.95071431, 0.73199394}, {0.59865848, 0.15601864, 0.15599452}, {0.05808361, 0.86617615, 0.60111501}};

    std::vector<float> expectedScores = {0.20528528, 0.26393574, 0.21687565};

    // Prepare for the return value
    float* scores = (float*)malloc(sizeof(float) * numTestSamples);

    // Call the function
    kdeScore(flatten(X).data(), numSamples, numFeatures, kernel.c_str(), bandwidth, flatten(samplesX).data(), numTestSamples, scores);
    std::vector<float> vecScores(scores, scores + numTestSamples);

    // Check for the result
    REQUIRE_THAT(vecScores, Catch::Approx(expectedScores));

    // Free resources
    if (scores) {
        free(scores);
    }
}
