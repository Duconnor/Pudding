# Minimum version of cmake required
cmake_minimum_required(VERSION 3.18)

# CUDA compiler
set(CMAKE_CUDA_COMPILER nvcc)

# Project information
project(Pudding VERSION 1.0
                DESCRIPTION "Running common machine learning algorithms on Nvidia's GPU"
                LANGUAGES CXX CUDA)

# Testing related
if (CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    include(CTest)
endif()

# Only enable test if this is the main project
if (CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME AND BUILD_TESTING)
    add_subdirectory(tests)
endif()

# Build the shared library
add_subdirectory(src)