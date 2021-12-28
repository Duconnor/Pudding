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

/**
 * Making predictions using the Naive Bayes classifier with the multinomial event model
 *
 * @param X The input data to classify, of shape (numSamples, vocabularySize)
 * @param classProbability The fitted class probability, the output of function naiveBayesMultinomialFit
 * @param wordProbability The fitted word probability, the output of function naiveBayesMultinomialFit
 * @param numSamples The number of samples in X
 * @param vocabularySize The size of the vocabulary
 * @param numClasses The number of classes
 * @param predictions The predictions made by the Naive Bayes model, the output of this function
 */
extern "C" void naiveBayesMultinomialPredict(const float* X, const float* classProbability, const float* wordProbability, const int numSamples, const int vocabularySize, const int numClasses, int* predictions);

#endif
