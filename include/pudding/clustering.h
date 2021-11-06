#ifndef PUDDING_CLUSTERING_H
#define PUDDING_CLUSTERING_H

/**
 * Perform kmeans clustering on the given data
 * 
 * @param X The input data, of shape (numSamples, numFeatures)
 * @param initCenters The initialized center
 * @param numSamples The first dimension of the input data X
 * @param numFeatures The second dimension of the input data X
 * @param numCenters The number of centers formed
 * @param maxNumIteration The maximum number of iteration
 * @param tolerance Relative tolerance with regards to Frobenius norm of the difference in the cluster centers of two consecutive iterations to declare convergence
 * @param cudaEnabled Whether to use cuda to accelerate it
 * @param centers The final found centers, of shape (numCenters, numFeatures)
 * @param membership The assignment of each data to the corresponding center, of shape (numSamples)
 * @param numIterations The number of iterations performed
 */
void kmeans(const float* X, const float* initCenters, const int numSamples, const int numFeatures, const int numCenters, const int maxNumIteration, const float tolerance, const bool cudaEnabled, float* centers, int* membership, int* numIterations);

#endif