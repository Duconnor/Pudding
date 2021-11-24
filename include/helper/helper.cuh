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

#endif