# Grab the Catch2 testing library
include(FetchContent)

FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG        v2.13.1)

FetchContent_MakeAvailable(Catch2)

# Create a test-time helper function library (a header only lib) so that all tests can use it
add_library(puddingtesthelper INTERFACE)
target_include_directories(puddingtesthelper INTERFACE helper/)

# Enable the unit test
add_subdirectory(unittest)