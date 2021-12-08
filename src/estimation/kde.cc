#include <pudding/estimation.h>

extern void _kdeScoreGPU(const float* X, const int numSamples, const int numFeatures, const char* kernel, const float bandwidth, const float* samplesX, const int numTestSamples, float* scores);

void kdeScore(const float* X, const int numSamples, const int numFeatures, const char* kernel, const float bandwidth, const float* samplesX, const int numTestSamples, float* scores) {
    _kdeScoreGPU(X, numSamples, numFeatures, kernel, bandwidth, samplesX, numTestSamples, scores);
}
