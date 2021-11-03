# Minimum version of cmake required
cmake_minimum_required(VERSION 3.18)

# Set the head files
set(HEADER_LIST "${PROJECT_SOURCE_DIR}/include/pudding/clustering.h")

# We also define some useful helper functions in the header file
set(HELPER_HEADER_LIST "${PROJECT_SOURCE_DIR}/src/helper/helper.h")

# Grap source files
set(SOURCE_LIST "clustering/kmeans.cc")

# Build a shared library
add_library(pudding SHARED "${SOURCE_LIST}" "${HEADER_LIST}" "${HELPER_HEADER_LIST}")

# The header files are needed to use the library
target_include_directories(pudding PUBLIC "${PROJECT_SOURCE_DIR}/include"
                                   PRIVATE "${PROJECT_SOURCE_DIR}/src/helper")

# At least C++ 11 is needed
target_compile_features(pudding PUBLIC cxx_std_11)