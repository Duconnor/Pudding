# Grab the Catch2 testing library
include(FetchContent)

FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG        v2.13.1)

FetchContent_MakeAvailable(Catch2)

# Enable the unit test
add_subdirectory(unittest)