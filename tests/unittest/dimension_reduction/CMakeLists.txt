# Enable tests for dimension reduction algorithms
add_executable(test_pca "src/test_pca.cc")

# The test should be linked to both the main library and the Catch2 testing library and the helper library
target_link_libraries(test_pca PRIVATE pudding Catch2::Catch2 puddingtesthelper)

# For debug
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(test_pca PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()

# Register the test
add_test(NAME TestPCATest COMMAND test_pca)