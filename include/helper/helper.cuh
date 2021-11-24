#ifndef HELPER_HELPER_CUH
#define HELPER_HELPER_CUH

void copyToHostAndDisplayFloat(const float* devicePtr, int row, int col);

/*
 * This is a wrapper function for a kernel call that performs matrix vector subtraction (i.e. broadcasting).
 * Pre-condition
 *  1. The vector must be in the shape of (numRow,).
 *  2. Need to specify the size of the shared memory, which is sizeof(float) * numRow.
 *  3. All pointers must be pointers on the device side.
 */
void wrapperMatrixVectorSubtraction(const float* matrix, const int numRow, const int numCol, const float* vector, float* res);

/*
 * This is a wrapper function for a kernel call that performs element-wise multiplication between two vectors, followed by a element-wise scale.
 * In other words, this function performs (x1 .* x2) / scale.
 */
void wrapperVectorVectorElementWiseMultiplication(const float* vecOne, const float* vecTwo, const int numElements, const float scale, float* res);

/*
 * This function performs matrix transpose using cublas functions.
 */
void transposeMatrix(float* matrix, const int numRow, const int numCol);

#endif