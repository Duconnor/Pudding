# Enable tests for classification algorithms
add_executable(test_naive_bayes "src/test_naive_bayes.cc")

# The test should be linked to both the main library and the Catch2 testing library and the helper library
target_link_libraries(test_naive_bayes PRIVATE pudding Catch2::Catch2 puddingtesthelper)

# For debug
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(test_naive_bayes PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()

# Register the test
add_test(NAME TestNaiveBayesTest COMMAND test_naive_bayes)