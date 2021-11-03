#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <string.h>
#include <assert.h>

#include <iostream>

#include <pudding/clustering.h>
#include "../helper/helper.h"

/* CPU-version of KMeans */
void _kmeansCPU(float* X, float* initCenters, int numSamples, int numFeatures, int numCenters, int maxNumIteration, float tolerance, float* centers, int* membership, int* numIterations) {
    /*
    * KMeans algorithm:
    * 1. Initialize the centroids -> this is already done
    * 2. Loop until coverage:
    *  2.1. Assign each sample to one of the clusters.
    *  2.2. Re-compute the centroid as the mean of samples belonging to this cluster.
    */

    // Pre-condition check
    assert(maxNumIteration >= 0);

    memcpy(centers, initCenters, sizeof(float) * numCenters * numFeatures);
    bool endFlag = maxNumIteration == 0;
    int iterationCount = 0;
    while (!endFlag) {
        // Step 2.1
        for (int i = 0; i < numSamples; i++) {
            // For every sample, we compute the distance between it and every centroid
            int minDistanceIdx = 0;
            float minDistance = FLT_MAX;
            for (int j = 0; j < numCenters; j++) {
                float distance = 0;
                for (int k = 0; k < numFeatures; k++) {
                    distance += pow((X[twoDimIndexToOneDim(i, k, numSamples, numFeatures)] - centers[twoDimIndexToOneDim(j, k, numCenters, numFeatures)]), 2);
                }
                if (distance < minDistance) {
                    minDistance = distance;
                    minDistanceIdx = j;
                }
            }
            membership[i] = minDistanceIdx;
        }

        // Step 2.2
        float* oldCenters = (float*)malloc(sizeof(float) * numCenters * numFeatures);
        memcpy(oldCenters, centers, sizeof(float) * numCenters * numFeatures);
        for (int i = 0; i < numCenters; i++) {
            // For every cluster, we compute the mean of samples belonging to this cluster
            
            // Initialize the center to zero
            for (int j = 0; j < numFeatures; j++) {
                centers[twoDimIndexToOneDim(i, j, numCenters, numFeatures)] = 0.0;
            }

            int numSamplesCount = 0;
            for (int j = 0; j < numSamples; j++) {
                if (membership[j] == i) {
                    numSamplesCount++;
                    for (int k = 0; k < numFeatures; k++) {
                        centers[twoDimIndexToOneDim(i, k, numCenters, numFeatures)] += X[twoDimIndexToOneDim(j, k, numSamples, numFeatures)];
                    }
                }
            }

            for (int j = 0; j < numFeatures; j++) {
                centers[twoDimIndexToOneDim(i, j, numCenters, numFeatures)] /= numSamplesCount;
            }
        }

        // Test for coverage
        iterationCount++;
        if (iterationCount >= maxNumIteration) {
            endFlag = true;
        } else {
            // Use the tolerance to check the coverage
            float diff = 0.0;
            for (int i = 0; i < numCenters * numFeatures; i++) {
                diff += pow((centers[i] - oldCenters[i]), 2);
            }
            endFlag = sqrt(diff) < tolerance;
        }
        if (oldCenters) {
            free(oldCenters);
        }
    }
    *numIterations = iterationCount;
}

void kmeans(float* X, float* initCenters, int numSamples, int numFeatures, int numCenters, int maxNumIteration, float tolerance, bool cudaEnabled, float* centers, int* membership, int* numIterations) {
    if (cudaEnabled) {
        assert(false);
    } else {
        _kmeansCPU(X, initCenters, numSamples, numFeatures, numCenters, maxNumIteration, tolerance, centers, membership, numIterations);
    }
    return;
}