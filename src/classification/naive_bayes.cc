#include <pudding/classification.h>

extern void _naiveBayesMultinomialFitGPU(const float* X, const int* y, const int numSamples, const int vocabularySize, const int numClasses, const float alpha, float* classProbability, float* wordProbability);

void naiveBayesMultinomialFit(const float* X, const int* y, const int numSamples, const int vocabularySize, const int numClasses, const float alpha, float* classProbability, float* wordProbability) {
    return _naiveBayesMultinomialFitGPU(X, y, numSamples, vocabularySize, numClasses, alpha, classProbability, wordProbability);
}
