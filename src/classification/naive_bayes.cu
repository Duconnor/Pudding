#include <assert.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <helper/helper_CUDA.h>
#include <helper/helper.cuh>

void _naiveBayesMultinomialFitGPU(const float* X, const int* y, const int numSamples, const int vocabularySize, const int numClasses, const float alpha, float* classProbability, float* wordProbability) {
    /*
     * This function fits a Naive Bayes model, it involves these steps:
     * 1. For every class:
     *  1.1. Find the total number of occurance for each word in the vocabulary plus the alpha (Laplace smoothing)
     *  1.2. Find the total number of samples belonging to this class.
     *  1.3. Divide the number of samples in this class by the total number of samples -> the class probability.
     *  1.4. Divide the total number of occurance of words by the number of samples in this class -> the word probability.
     * 
     */

    float one = 1.0, zero = 0.0;

    // Malloc space on GPU
    float* deviceX;
    int* deviceY;
    float* deviceWordProbability;
    float* deviceMaskVec; // This mask vec is used to count the occurance of each word, as well as the number of samples belonging to each class
    float* deviceAllOneVec; // This is a all one vector, used to pre-compute the number of words in each sample
    float* deviceWordsCount;

    CUDA_CALL( cudaMalloc(&deviceX, sizeof(float) * numSamples * vocabularySize) );
    CUDA_CALL( cudaMalloc(&deviceY, sizeof(int) * numSamples) );
    CUDA_CALL( cudaMalloc(&deviceWordProbability, sizeof(float) * numClasses * vocabularySize) );
    CUDA_CALL( cudaMalloc(&deviceMaskVec, sizeof(float) * numSamples) );
    CUDA_CALL( cudaMalloc(&deviceAllOneVec, sizeof(float) * vocabularySize) );
    CUDA_CALL( cudaMalloc(&deviceWordsCount, sizeof(float) * numSamples) );

    CUDA_CALL( cudaMemcpy(deviceX, X, sizeof(float) * numSamples * vocabularySize, cudaMemcpyHostToDevice) );
    CUDA_CALL( cudaMemcpy(deviceY, y, sizeof(int) * numSamples, cudaMemcpyHostToDevice) );
    wrapperInitializeAllElementsToXKernel(deviceWordProbability, one, numClasses * vocabularySize); // It's important here to initialize all elements to one so we can do all things in one blas operation
    wrapperInitializeAllElementsToXKernel(deviceAllOneVec, one, vocabularySize);

    // Prepare cublas handle
    cublasHandle_t cublasHandle;
    CUBLAS_CALL( cublasCreate(&cublasHandle) );

    // Pre-compute the number of words in each sample
    // This can be done via a matrix vector multiplcation
    CUBLAS_CALL( cublasSgemv(cublasHandle, CUBLAS_OP_T, vocabularySize, numSamples, &one, deviceX, vocabularySize, deviceAllOneVec, one, &zero, deviceWordsCount, one) );

    // Start the main loop (loop for every class)
    for (int classIdx = 0; classIdx < numClasses; classIdx++) {
        // Generate the mask vector first
        wrapperGenerateMaskVectorKernel(deviceY, classIdx, numSamples, deviceMaskVec);
        // Multiply the mask vector to the pre-computed words count vector yields the total number of words in samples belonging to this class
        float numWordsThisClass = 0.0;
        CUBLAS_CALL( cublasSdot(cublasHandle, numSamples, deviceWordsCount, one, deviceMaskVec, one, &numWordsThisClass) );
        float numSamplesThisClass = 0.0;
        CUBLAS_CALL( cublasSasum(cublasHandle, numSamples, deviceMaskVec, one, &numSamplesThisClass) );
        // Multiply deviceX with deivceMaskVec to obtain the total number of occurance for each word in samples belonging to this class
        // Further divide the result by numWordsThisClass results in the word probability estimation
        float denominator = 1.0 / (numWordsThisClass + (vocabularySize * alpha));
        float beta = alpha * denominator;
        CUBLAS_CALL( cublasSgemv(cublasHandle, CUBLAS_OP_N, vocabularySize, numSamples, &denominator, deviceX, vocabularySize, deviceMaskVec, one, &beta, deviceWordProbability + (classIdx * vocabularySize), one) );
        classProbability[classIdx] = numSamplesThisClass / numSamples;
    }

    // Copy the result back to host
    CUDA_CALL( cudaMemcpy(wordProbability, deviceWordProbability, sizeof(float) * numClasses * vocabularySize, cudaMemcpyDeviceToHost) );

    // Free resources
    CUDA_CALL( cudaFree(deviceX) );
    CUDA_CALL( cudaFree(deviceY) );
    CUDA_CALL( cudaFree(deviceWordProbability) );
    CUDA_CALL( cudaFree(deviceMaskVec) );

    CUBLAS_CALL( cublasDestroy(cublasHandle) );
}
