#ifndef HELPER_HELPER_CUH
#define HELPER_HELPER_CUH

#define MAXSHAREDMEMBYTES 48 * 1024 // The maximum shared memory one block can use is 48KB

void copyToHostAndDisplayFloat(const float* devicePtr, int row, int col);

/*
 * This is a wrapper function for a kernel call that performs matrix vector addition (i.e. broadcasting).
 * Specifically, this function perfroms X + scale * a
 * Pre-condition
 *  1. The vector must be in the shape of (numRow,).
 *  2. Need to specify the size of the shared memory, which is sizeof(float) * numRow.
 *  3. All pointers must be pointers on the device side.
 */
void wrapperMatrixVectorAddition(const float* matrix, const int numRow, const int numCol, const float* vector, float scale, float* res);

/*
 * This is a wrapper function for a kernel call that performs element-wise multiplication between two vectors, followed by a element-wise scale.
 * In other words, this function performs (x1 .* x2) / scale.
 */
void wrapperVectorVectorElementWiseMultiplication(const float* vecOne, const float* vecTwo, const int numElements, const float scale, float* res);

/*
 * This function performs matrix transpose using cublas functions.
 */
void transposeMatrix(float* matrix, const int numRow, const int numCol);

/*
 * This function set all elements in an array to a specific value
 * Note: cudaMemset cannot be used to set value other than 0 because it actually sets for byte value
 */
void wrapperInitializeAllElementsToXKernel(float* vec, const float X, const int numElements);

/*
 * This is a wrapper function for computing the **squared** pair-wise euclidean distance between two set of points.
 * It can be viewed as a generalized N-body problem, and is useful in many applications ranging from KDE to KNN.
 * 
 * Note: to enable coalesced memory access, refX and queryX are assumed to have each column as an example. And the dist will have shape (numExamplesRef, numExamplesQuery).
 */
void wrapperComputePairwiseEuclideanDistanceKerenl(const float* refX, const float* queryX, const int numExamplesRef, const int numExamplesQuery, const int numFeatures, float* dist);

/*
 * This is a wrapper function for generating a mask vector based on a given label vector and a given target label. Specifically, this function generate a new vector of the same size with the original label vector. The new vector consists of 1 and 0, for position where the label vector has the same value with the target label, the output will be 1.
 *
 * Note: the result vector is of type float because in many cases, it will be used for further computation.
 */
void wrapperGenerateMaskVectorKernel(const int* labelVec, const int targetLabel, const int numElements, float* maskVec);

#endif
