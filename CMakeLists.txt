# Minimum version of cmake required
cmake_minimum_required(VERSION 3.18)

# Project information
project(Pudding VERSION 1.0
                DESCRIPTION "Running scikit learn functions on Nvidia's GPU"
                LANGUAGES CXX CUDA)

# Build the shared library
add_subdirectory(src)