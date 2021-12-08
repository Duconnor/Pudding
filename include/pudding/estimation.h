#ifndef PUDDING_ESTIMATION_H
#define PUDDING_ESTIMATION_H

/**
 * Given the training data X and sample data samplesX, this function computes the log-likelihood of each sample using the kernel density estimation.  
 * 
 * NOTE: NO CPU VERSION !!
 * 
 * @param X The training data, of shape (numSamples, numFeatures).
 * @param numSamples The number of samples in the training data.
 * @param numFeatures The dimension of features.
 * @param kernel A string, represents the kernel to use, valid kernels are "gaussian", "epanechnikov".
 * @param bandwidth The bandwidth hyperparameter used in the kernel function.
 * @param samplesX The test data, of shape (numTestSamples, numFeatures).
 * @param numTestSamples The number of samples in the test data.
 * @param scores The return value, of shape (numTestSamples), the log-likelihood of each test sample.
 */
extern "C" void kdeScore(const float* X, const int numSamples, const int numFeatures, const char* kernel, const float bandwidth, const float* samplesX, const int numTestSamples, float* scores);

#endif