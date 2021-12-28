#include <pudding/classification.h>

extern void _naiveBayesMultinomialFitGPU(const float* X, const int* y, const int numSamples, const int vocabularySize, const int numClasses, const float alpha, float* classProbability, float* wordProbability);

extern void _naiveBayesMultinomialPredictGPU(const float* X, const float* classProbability, const float* wordProbability, const int numSamples, const int vocabularySize, const int numClasses, int* predictions);

void naiveBayesMultinomialFit(const float* X, const int* y, const int numSamples, const int vocabularySize, const int numClasses, const float alpha, float* classProbability, float* wordProbability) {
    return _naiveBayesMultinomialFitGPU(X, y, numSamples, vocabularySize, numClasses, alpha, classProbability, wordProbability);
}

void naiveBayesMultinomialPredict(const float* X, const float* classProbability, const float* wordProbability, const int numSamples, const int vocabularySize, const int numClasses, int* predictions) {
    return _naiveBayesMultinomialPredictGPU(X, classProbability, wordProbability, numSamples, vocabularySize, numClasses, predictions);
}
