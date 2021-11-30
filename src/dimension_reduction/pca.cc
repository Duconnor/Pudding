#include <pudding/dimension_reduction.h>

extern void _pcaGPU(const float* X, const int numSamples, const int numFeatures, const int numComponents, const float variancePercentage, float* principalComponets, float* principalAxes, float* variances, float* reconstructedX, int* numComponentsChosen);

void pca(const float* X, const int numSamples, const int numFeatures, const int numComponents, const float variancePercentage, float* principalComponets, float* principalAxes, float* variances, float* reconstructedX, int* numComponentsChosen) {
    return _pcaGPU(X, numSamples, numFeatures, numComponents, variancePercentage, principalComponets, principalAxes, variances, reconstructedX, numComponentsChosen);
}
