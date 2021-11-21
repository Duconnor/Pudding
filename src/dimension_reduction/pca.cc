#include <pudding/dimension_reduction.h>

extern void _pcaGPU(const float* X, const int numSamples, const int numFeatures, const int targetDimension, float* projectedX);

void pca(const float* X, const int numSamples, const int numFeatures, const int targetDimension, float* projectedX) {
    return _pcaGPU(X, numSamples, numFeatures, targetDimension, projectedX);
}