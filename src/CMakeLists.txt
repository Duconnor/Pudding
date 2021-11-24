# Minimum version of cmake required
cmake_minimum_required(VERSION 3.18)

# Set the head files
set(HEADER_LIST "${PROJECT_SOURCE_DIR}/include/pudding/clustering.h"
                "${PROJECT_SOURCE_DIR}/include/pudding/dimension_reduction.h")

# We also define some useful helper functions in the header file
set(HELPER_HEADER_LIST "${PROJECT_SOURCE_DIR}/include/helper/helper.h"
                       "${PROJECT_SOURCE_DIR}/include/helper/helper_CUDA.h"
                       "${PROJECT_SOURCE_DIR}/include/helper/helper.cuh")

# Grap source files
set(SOURCE_LIST "clustering/kmeans.cc"
                "clustering/kmeans.cu"
                "dimension_reduction/pca.cc"
                "dimension_reduction/pca.cu"
                "helper/helper.cu")

# Build a shared library
add_library(pudding SHARED "${SOURCE_LIST}" "${HEADER_LIST}" "${HELPER_HEADER_LIST}")

# The header files are needed to use the library
target_include_directories(pudding PUBLIC "${PROJECT_SOURCE_DIR}/include")

# Link necessary libraries
target_link_libraries(pudding PRIVATE cublas cusolver)

# At least C++ 11 is needed
target_compile_features(pudding PUBLIC cxx_std_11)

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    # For debug
    target_compile_options(pudding PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
else()
    # For automatical optimization
    target_compile_options(pudding PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas -O3>)
endif()

# Generate code for arch 60
set_property(TARGET pudding PROPERTY CUDA_ARCHITECTURES 60)