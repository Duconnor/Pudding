# Minimum version of cmake required
cmake_minimum_required(VERSION 3.18)

# Set the head files
set(HEADER_LIST "${PROJECT_SOURCE_DIR}/include/pudding/clustering.h")

# Build a shared library
add_library(pudding SHARED clustering/kmeans.cc "${HEADER_LIST}")

# The header files are needed to use the library
target_include_directories(pudding PUBLIC "${PROJECT_SOURCE_DIR}/include")

# At least C++ 11 is needed
target_compile_features(pudding PUBLIC cxx_std_11)