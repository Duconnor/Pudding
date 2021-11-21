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
 * @param targetDimension The target dimension we want to project our data into, must be smaller than numFeatures
 * @param projectedX The transformed result of X
 */
extern "C" void pca(const float* X, const int numSamples, const int numFeatures, const int targetDimension, float* projectedX);

#endif