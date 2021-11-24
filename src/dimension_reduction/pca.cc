#include <pudding/dimension_reduction.h>

extern void _pcaGPU(const float* X, const int numSamples, const int numFeatures, const int numComponets, const float variancePercentage, float* principalComponets, float* principalAxes, float* variances);

void pca(const float* X, const int numSamples, const int numFeatures, const int numComponets, const float variancePercentage, float* principalComponets, float* principalAxes, float* variances) {
    return _pcaGPU(X, numSamples, numFeatures, numComponets, variancePercentage, principalComponets, principalAxes, variances);
}