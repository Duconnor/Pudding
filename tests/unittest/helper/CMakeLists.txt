# Enable tests for clustering algorithms
add_executable(test_helper "src/test_helper.cu")

# The test should be linked to both the main library and the Catch2 testing library and the helper library
target_link_libraries(test_helper PRIVATE pudding Catch2::Catch2 puddingtesthelper)

# For debug
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(test_helper PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()

# Generate code for arch 60
set_property(TARGET test_helper PROPERTY CUDA_ARCHITECTURES 60)

# Register the test
add_test(NAME TestHelperTest COMMAND TestHelper)