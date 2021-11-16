# Minimum version of cmake required
cmake_minimum_required(VERSION 3.18)

# Set the head files
set(HEADER_LIST "${PROJECT_SOURCE_DIR}/include/pudding/clustering.h")

# We also define some useful helper functions in the header file
set(HELPER_HEADER_LIST "${PROJECT_SOURCE_DIR}/include/helper/helper.h"
                       "${PROJECT_SOURCE_DIR}/include/helper/helperCUDA.h"
                       "${PROJECT_SOURCE_DIR}/include/helper/helper.cuh")

# Grap source files
set(SOURCE_LIST "clustering/kmeans.cc"
                "clustering/kmeans.cu"
                "helper/helper.cu")

# Build a shared library
add_library(pudding SHARED "${SOURCE_LIST}" "${HEADER_LIST}" "${HELPER_HEADER_LIST}")

# The header files are needed to use the library
target_include_directories(pudding PUBLIC "${PROJECT_SOURCE_DIR}/include")

# Link necessary libraries
target_link_libraries(pudding cublas)

# At least C++ 11 is needed
target_compile_features(pudding PUBLIC cxx_std_11)

# For debug
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(pudding PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()

# For automatical optimization
target_compile_options(pudding PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas -O3>)