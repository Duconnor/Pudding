# Grab the Catch2 testing library
include(FetchContent)

FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG        v2.13.1)

FetchContent_MakeAvailable(Catch2)

# Enable test
add_executable(testKmeans "clustering/testKmeans.cc")

# The test should be linked to both the main library and the Catch2 testing library
target_link_libraries(testKmeans PRIVATE pudding Catch2::Catch2)

# Register the test
add_test(NAME testKmeansTest COMMAND testKmeans)