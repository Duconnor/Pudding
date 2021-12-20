# Pudding: Running common machine learning algorithms on Nvidia's GPUs

## Goal
Providing GPU support for some common existing machine learning algorithms.  
Start simple (k-means clustering).  
Unit test for every module.  

## Plan
2021.10.28 - 2022.3.1  
* Before 2021.11.1: learn CMake, read codes from others, read official codes of scikit-learn, try to come up with a plan about how to organize the code.
* 2021.11.1 - 2021.12.1: finish the initial version. I should have implemented at least two clustering algorithms and provide python bindings for them.
* 2021.12.1 - 2022.1.1: continue writing for other possible functions, refactor the code if needed. Also, benchmark all algorithms implemented so far and compare them with the CPU implementation in scikit-learn.
* 2021.1.1 - 2022.3.1: publish the project on GitHub with a README providing basic information (what is this, how to use it), also make a websit for it and host the official website on my github.io page.

## TODOs
Order reflects priority, the one on the top has the highest priority.
- [x] Implement KDE.
- [x] There is a bug in the kmeans implementation, see the corresponding TODO in the kmeans.cu file.
- [x] Add assertion check for every shared memory allocation.
- [x] Use the helper function to transpose matrix in kmeans.cu.
- [x] Benchmark kmeans by gradually increasing the number of points, not the number of clusters.
- [x] Benchmark PCA with respect to scikit-learn.
- [x] If __syncthreads() is used in a conditional code, the condition must evaluates identically across the entire thread block. Fix this bug in kernels used in kmeans.
- [ ] Optimize PCA (maybe use a eco mode?)
- [ ] Change the reference on PCA's site by using the offical citation provided in StackOverflow.
- [ ] Think more carefully about the choice of the block size (i.e. the number of threads within the block).
- [ ] The kmeans function will not work for large data samples (many high dimensional data points with many centers). Currently, I just add a simple assertion to check for this and abort the program if there is not enough shared memory.
