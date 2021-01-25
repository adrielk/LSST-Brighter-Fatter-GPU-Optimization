# LSST-Brighter-Fatter-GPU-Optimization
An implementation of the brighter-fatter correction function on the GPU using CUDA.

Much of the brighter-fatter correction algorithm can be parallelized on the GPU. Using [CUDA C++](https://developer.nvidia.com/cuda-toolkit), I implemented the brighter-fatter correction algorithm used in the (Large Synoptic Survey Telescope) LSST codebase to run on the GPU. The original [LSST code](https://github.com/lsst/ip_isr/blob/master/python/lsst/ip/isr/isrFunctions.py) is written in Python and made use of easily parallelizable NumPy functions such as gradient, diff, abs, and sum. The convolution involved in the brighter-fatter correction algorithm was parallelized using a CUDA 2D convolution filter from the [NVIDIA 2D Image and Signal Performance Primitives (NPP) library](https://docs.nvidia.com/cuda/npp/group__image__filter.html#CommonFilterParameters).


How to run files:

1. Install [CUDA Toolkit and CUDA Samples](https://docs.nvidia.com/cuda/cuda-samples/index.html#getting-started-with-cuda-samples) (Must have compatible GPU)
2. Move brighterFatter_GPU directory into NVIDIA_CUDA-11.0_Samples/7_CUDALibraries/
3. cd into brighterFatter_GPU directory and run Makefile
4. Add large textfiles from this [Google Doc](https://drive.google.com/drive/folders/1fT29teYEGMKKnsA0HxbufQlmHB_CMpWU?usp=sharing) to brighterFatter_GPU directory
4. Run brighterFatter_GPU executable by typing: ./brighterFatter_GPU


## Notes:
- The main code is in the brighterFatter_GPU.cpp file
- Helper CUDA kernels are in matrixOpsFuncs.cu and matrix_ops.cuh
