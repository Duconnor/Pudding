#ifndef PUDDING_CLASSIFICATION_H
#define PUDDING_CLASSIFICATION_H

/**
 * Fit the Naive Bayes classifier using the multinomial event model.
 *
 * NOTE: NO CPU VERSION AVALIABLE FOR THIS FUNCTION.
 *
 * @param X The training data, of shape (numSamples, vocabularySize)
 * @param y The label of training data, of shape (numSamples,)
 * @param numSamples The number of samples in the training dataset
 * @param vocabularySize The size of the vocabulary
 * @param numClasses The number of classes
 * @param alpha The additive Laplace smoothing parameter (0 for no smoothing)
 * @param classProbability The fitted class probability (i.e. p(y)), of shape (numClasses,)
 * @param wordProbability The fitted parameter of the multinomial distribution belonging to the different class (i.e. p(k|y)), of shape (numClasses, vocabularySize)
 */
extern "C" void naiveBayesMultinomialFit(const float* X, const int* y, const int numSamples, const int vocabularySize, const int numClasses, const float alpha, float* classProbability, float* wordProbability);

#endif
