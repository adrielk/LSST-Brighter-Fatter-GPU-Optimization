# LSST-Brighter-Fatter-GPU-Optimization
An implementation of the brighter-fatter correction function on the GPU using CUDA.

Much of the brighter-fatter correction algorithm can be parallelized on the GPU. Using [CUDA C++](https://developer.nvidia.com/cuda-toolkit), I implemented the brighter-fatter correction algorithm used in the (Large Synoptic Survey Telescope) LSST codebase to run on the GPU. The original [LSST code](https://github.com/lsst/ip_isr/blob/master/python/lsst/ip/isr/isrFunctions.py) is written in Python and made use of easily parallelizable NumPy functions such as gradient, diff, abs, and sum. The convolution involved in the brighter-fatter correction algorithm was parallelized using a CUDA 2D convolution filter from the [NVIDIA 2D Image and Signal Performance Primitives (NPP) library](https://docs.nvidia.com/cuda/npp/group__image__filter.html#CommonFilterParameters). I was able to achieve an 11x speed up with my GPU optimized implementation of brighter-fatter correction.

## Technical Report:
The [technical report](https://home.fnal.gov/~mwang/lsst/adriel/RTN-015.pdf) discusses the results and methods in more detail

### How to run GPU code:

1. Install [CUDA Toolkit and CUDA Samples](https://docs.nvidia.com/cuda/cuda-samples/index.html#getting-started-with-cuda-samples) (Must have compatible GPU)
2. Move brighterFatter_GPU directory into NVIDIA_CUDA-11.0_Samples/7_CUDALibraries/
3. cd into brighterFatter_GPU directory and run Makefile
4. Add large textfiles from this [Google Doc](https://drive.google.com/drive/folders/1fT29teYEGMKKnsA0HxbufQlmHB_CMpWU?usp=sharing) to brighterFatter_GPU directory
4. Run brighterFatter_GPU executable by typing: ./brighterFatter_GPU

### How to run CPU code:

1. Install OpenCV https://docs.opencv.org/master/d7/d9f/tutorial_linux_install.html
2. cd into brighterFatter_CPU directory and run Makefile
3. Add large textfiles from this [Google Doc](https://drive.google.com/drive/folders/1fT29teYEGMKKnsA0HxbufQlmHB_CMpWU?usp=sharing) to brighterFatter_GPU directory
4. Run brighterFatter_CPU executable by typing: ./brighterFatter_CPU

## Notes:
- The main code is in the brighterFatter_GPU.cpp file
- Helper CUDA kernels are in matrixOpsFuncs.cu and matrix_ops.cuh

## Speed-up Results and Error:
- [Results](https://docs.google.com/spreadsheets/d/1lHfqa3vAcOzV9VrahLMfkYolysIajMqwmt9a368h-rA/edit?usp=sharing)

