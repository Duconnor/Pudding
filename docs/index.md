---
layout: default
title: Pudding 🍮
nav_order: 1
---
# Pudding 🍮

Welcome to the official site for the project Pudding!

Pudding enables you to run various machine learning algorithms on Nvidia's GPU. It is written in C/C++ and CUDA. To make it easier for one to use, it also comes with a Python binding.

## Installation

*Currently Pudding has only been tested on Linux.*

### Dependencies

Compiling the Pudding shared libarary requires:
* CMake >= 3.18
* gcc >= 5.4.0
* CUDA 10.1

### Installation Steps

First you need to compile the shared libaray:

```shell
mkdir build
cd build
cmake --DCMAKE_BUILD_TYPE=Release ..
cmake --build .
```

Or you can simply use the provided scripts:
```shell
sh scripts/release.sh
```

You can run tests if you want to check the build:
```shell
cmake --build . -t test
```

If you need the Python binding, go back to the root directory and run:
```shell
cd python
python setup.py install
```

You can run tests if you want (again, assume you are in the root directory and be sure to install the pytest package in Python):
```shell
pytest python/tests
```
