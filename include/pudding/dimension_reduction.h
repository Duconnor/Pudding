#ifndef PUDDING_DIMENSION_REDUCTION_H
#define PUDDING_DIMENSION_REDUCTION_H

/**
 * Perform Principal Component Analysis (PCA) on the given data. 
 * 
 * NOTE: NO CPU VERSION IS AVAILABLE FOR THIS FUNCTION!!
 * 
 * @param X The input data, of shape (numSamples, numFeatures)
 * @param numSamples The first dimension of X, the number of samples
 * @param numFeatures The second dimension of X, the original dimension of X
 * @param numComponents The number of components to keep, can be set to -1 if you want to choose the number of components based on the percentage of variance
 * @param variancePercentage If the above parameter is -1, this parameter is used to select the number of components such that the amount of variance is greater than the percentage specified here. If valid (not -1), must be 0 < varaincePercentage < 1
 * @param principalComponets The principal components, of shape (numSamples, numComponents)
 * @param principalAxes The principal directions, of shape (numComponents, numFeatures)
 * @param variances The variance of each principal directions
 * @param reconstructedX The reconstruction of the original data X which has the lowest possible reconstruction error, of shape (numSample, numFeatures)
 * @param numComponentsChosen The actual number of components chosen, if numComponents is not -1, this equals to numComponents, otherwise, it is the number of components chosen to meet the variancePercentage
 */
extern "C" void pca(const float* X, const int numSamples, const int numFeatures, const int numComponents, const float variancePercentage, float* principalComponets, float* principalAxes, float* variances, float* reconstructedX, int* numComponentsChosen);

#endif