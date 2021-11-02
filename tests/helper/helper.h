#ifndef TEST_HELPER_H
#define TEST_HELPER_H

#include <vector>


/*
 * A genric function to flatten a 2D vector into a 1D vector
 * Credit: https://stackoverflow.com/questions/47418120/writing-a-function-that-will-flatten-a-two-dimensional-array
 */
template <typename T>
std::vector<T> flatten(const std::vector<std::vector<T>> & vec) {   
    std::vector<T> result;
    for (const auto & v : vec) {
        result.insert(result.end(), v.begin(), v.end());
    }                                                                                     
    return result;
}

#endif