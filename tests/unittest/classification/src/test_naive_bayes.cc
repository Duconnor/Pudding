#define CATCH_CONFIG_MAIN
#include <vector>
#include <cstdlib>
#include <iostream>

#include <catch2/catch.hpp>
#include <pudding/classification.h>

#include <helper.h>

TEST_CASE ("Test fit Multinomial Naive Bayes classifier using random toy data", "[multinomial-naive-bayes-fit-toy-data]") {
    // Prepare for the data
    const int numSamples = 6;
    const int vocabularySize = 10;
    const int numClasses = 6;
    const float alpha = 1.0;

    std::vector<std::vector<float>> X = {{3, 4, 0, 1, 3, 0, 0, 1, 4, 4}, {1, 2, 4, 2, 4, 3, 4, 2, 4, 2}, {4, 1, 1, 0, 1, 1, 1, 1, 0, 4}, {1, 0, 0, 3, 2, 1, 0, 3, 1, 1}, {3, 4, 0, 1, 3, 4, 2, 4, 0, 3}, {1, 2, 0, 4, 1, 2, 2, 1, 0, 1}};
    std::vector<int> y = {1, 2, 3, 4, 5, 6};
    
    std::vector<float> expectedClassProbability = {0.16666667, 0.16666667, 0.16666667, 0.16666667, 0.16666667, 0.16666667};
    std::vector<float> expectedWordProbability = {0.13333333, 0.16666667, 0.03333333, 0.06666667, 0.13333333, 0.03333333, 0.03333333, 0.06666667, 0.16666667, 0.16666667, 0.05263158, 0.07894737, 0.13157895, 0.07894737, 0.13157895, 0.10526316, 0.13157895, 0.07894737, 0.13157895, 0.07894737, 0.20833333, 0.08333333, 0.08333333, 0.04166667, 0.08333333, 0.08333333, 0.08333333, 0.08333333, 0.04166667, 0.20833333, 0.09090909, 0.04545455, 0.04545455, 0.18181818, 0.13636364, 0.09090909, 0.04545455, 0.18181818, 0.09090909, 0.09090909, 0.11764706, 0.14705882, 0.02941176, 0.05882353, 0.11764706, 0.14705882, 0.08823529, 0.14705882, 0.02941176, 0.11764706, 0.08333333, 0.125, 0.04166667, 0.20833333, 0.08333333, 0.125, 0.125, 0.08333333, 0.04166667, 0.08333333};

    // Prepare for the output
    float* classProbability = (float*)malloc(sizeof(float) * numClasses);
    float* wordProbability = (float*)malloc(sizeof(float) * numClasses * vocabularySize);
    
    // Call the function
    naiveBayesMultinomialFit(flatten(X).data(), y.data(), numSamples, vocabularySize, numClasses, alpha, classProbability, wordProbability);

    std::vector<float> vecClassProbability(classProbability, classProbability + numClasses);
    std::vector<float> vecWordProbability(wordProbability, wordProbability + (numClasses * vocabularySize));

    // Check the results
    REQUIRE_THAT(vecClassProbability, Catch::Approx(expectedClassProbability));
    REQUIRE_THAT(vecWordProbability, Catch::Approx(expectedWordProbability));
}
