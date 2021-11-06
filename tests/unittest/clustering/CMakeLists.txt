# Enable tests for clustering algorithms
add_executable(testKmeans "src/testKmeans.cc")

# The test should be linked to both the main library and the Catch2 testing library and the helper library
target_link_libraries(testKmeans PRIVATE pudding Catch2::Catch2 puddingtesthelper)

# For debug
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(testKmeans PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()

# Register the test
add_test(NAME testKmeansTest COMMAND testKmeans)