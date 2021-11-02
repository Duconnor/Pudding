#ifndef PUDDING_CLUSTERING_H
#define PUDDING_CLUSTERING_H

namespace pudding {
    /*
     * Perform kmeans clustering on the given data
     * 
     * @param X: the input data, of shape (numSamples, numFeatures)
     * @param initCenters: the initialized center
     * @param numSamples: the first dimension of the input data X
     * @param numFeatures: the second dimension of the input data X
     * @param numCenters: the number of centers formed
     * @param maxNumIteration: the maximum number of iteration
     * @param tolerance: relative tolerance with regards to Frobenius norm of the difference in the cluster centers of two consecutive iterations to declare convergence
     * @param cudaEnabled: whether to use cuda to accelerate it
     * @param centers: the final found centers, of shape (numCenters, numFeatures)
     * @param membership: the assignment of each data to the corresponding center, of shape (numSamples)
     */
    void kmeans(float* X, float* initCenters, int numSamples, int numFeatures, int numCenters, int maxNumIteration, float tolerance, bool cudaEnabled, float* centers, int* membership);
}

#endif