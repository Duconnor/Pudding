# Enable tests for estimation algorithms
add_executable(test_kde "src/test_kde.cc")

# The test should be linked to both the main library and the Catch2 testing library and the helper library
target_link_libraries(test_kde PRIVATE pudding Catch2::Catch2 puddingtesthelper)

# For debug
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(test_kde PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()

# Register the test
add_test(NAME TestKDETest COMMAND test_kde)