/**
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

//Author: Adriel Kim
//1-31-2021

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#  define WINDOWS_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#  pragma warning(disable:4819)
#endif



#include <ImagesCPU.h>
#include <ImagesNPP.h>
#include <ImageIO.h>
#include <Exceptions.h>

#include <string.h>
#include <fstream>
#include <iostream>

#include <cuda_runtime.h>
#include <npp.h>

#include <helper_string.h>
#include <helper_cuda.h>

//Additional includes
#include <sys/time.h>
#include <bits/stdc++.h>
#include <stdlib.h>
#include <typeinfo>
#include <stdlib.h>
#include <math.h>
#include <set>

#include "matrix_ops.cuh"

long long start_timer();
long long stop_timer(long long start_time, const char* name);
void fillKernelArray(std::string kernelName, Npp32f* kernelArr, int kernelSize);

/*Image size constants */
const int R = 4176;
const int C = 2048; 
const int N = (R * C);

inline int cudaDeviceInit(int argc, const char **argv)
{
    int deviceCount;
    checkCudaErrors(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0)
    {
        std::cerr << "CUDA error: no devices supporting CUDA." << std::endl;
        exit(EXIT_FAILURE);
    }

    int dev = findCudaDevice(argc, argv);

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    std::cerr << "cudaSetDevice GPU" << dev << " = " << deviceProp.name << std::endl;

    checkCudaErrors(cudaSetDevice(dev));

    return dev;
}

bool printfNPPinfo(int argc, char *argv[])
{
    const NppLibraryVersion *libVer   = nppGetLibVersion();

    printf("NPP Library Version %d.%d.%d\n", libVer->major, libVer->minor, libVer->build);

    int driverVersion, runtimeVersion;
    cudaDriverGetVersion(&driverVersion);
    cudaRuntimeGetVersion(&runtimeVersion);

    printf("  CUDA Driver  Version: %d.%d\n", driverVersion/1000, (driverVersion%100)/10);
    printf("  CUDA Runtime Version: %d.%d\n", runtimeVersion/1000, (runtimeVersion%100)/10);

    // Min spec is SM 1.0 devices
    bool bVal = checkCudaCapabilities(1, 0);
    return bVal;
}

/*
Loads kernel from textfile into an array.
*/
void fillKernelArray(std::string kernelName, Npp32f* kernelArr, int kernelSize) {
    std::fstream file;
    std::string word, t, q, filename;
    // filename of the file
    filename = kernelName;

    // opening file
    file.open(filename.c_str());
    //float sum = 0;

    // extracting words from the file
    if (file.is_open()) {
        for (int i = 0; i < kernelSize;i++) {
            file >> word;
            int n = word.length();
            char char_array[n + 1];
            strcpy(char_array, word.c_str());
            char* pEnd;
            double wordfloat = strtod(char_array, &pEnd);
            kernelArr[i] = wordfloat;
            //sum += wordfloat;
        }

    }

    /*Normally, you would normalize the kernel. 
    But in this case, it seems that this has already been 
    done for the kernel I retrieved from the LSST test cases.*/
    /*for (int i = 0;i < kernelSize;i++) {
        kernelArr[i] = kernelArr[i] / sum;
    }*/

    file.close();
}


/*Gets mean squared error between two images*/
double meanSquaredImages(float* img1, float* img2, int imgSize) {
    double diff = 0;
    for (int i = 0;i < imgSize;i++) {
        double pix1 = img1[i];
        double pix2 = img2[i];
        double pixDiff = pix1 - pix2;
        diff += (pixDiff*pixDiff);
    }
    double mse = diff / ((double)imgSize);
    return mse;
}

/*Gets maximum error percent*/
//double maxError(float* img1, float* img_true, int imgSize) {
//    double maxError = abs(img1[0]-img_true[0]);
//    double true_val = img_true[0];
//    double max_p = 0;
//    for (int i = 1;i < imgSize;i++) {
//        double err = abs(img1[i] - img_true[i]);
//        double p_val = err / img_true[i];
//        if (p_val > max_p && img_true[i] !=0){
//            max_p = p_val;
//            maxError = err;
//            true_val = img_true[i];
//        }
//    }
//    return max_p;
//}
double maxError(float* img1, float* img_true, int imgSize) {
    double maxError = abs(img1[0] - img_true[0]);
    double true_val = img_true[0];
    double predicted = img1[0];
    double max_percent = 0;
    for (int i = 1;i < imgSize;i++) {
        double err = abs(img1[i] - img_true[i]);
        double truth = img_true[i];
        double percent = err / truth;
        /*if (err > maxError) {
            maxError = err;
            true_val = img_true[i];
        }*/
        if (percent > max_percent) {
            max_percent = percent;
            true_val = img_true[i];
            predicted = img1[i];
        }
    }

    //std::cout << "True value: " << true_val << std::endl;
    //std::cout << "Predicted value: " << predicted << std::endl;
    //double p_error = maxError / true_val;
    return max_percent;//p_error;
}

void printMaxAndMin(float* img, int imgSize) {
    double min = img[0];
    double max = img[0];
    for (int i = 0;i < imgSize;i++) {
        if (img[i] < min) {
            min = img[i];
        }
        if (img[i] > max) {
            max = img[i];
        }
    }
    std::cout << "MAX: " << max << "/ MIN: " << min << std::endl;
}

//Error surrounding bright areas, where it's most relevant
double maxBrightPixelError(float* img1, float* img_true, int imgSize) {
    double maxError = abs(img1[0] - img_true[0]);
    double true_val = img_true[0];
    double predicted = img1[0];
    double max_percent = 0;
    for (int i = 1;i < imgSize;i++) {
        double err = abs(img1[i] - img_true[i]);
        double truth = img_true[i];
        double percent = err / truth;

        /*abs(img_true[i]>0.1) filters out very small values, 
        which tend to produce extreme errors since even very
        small differences can result in large error percentages*/
        if (percent > max_percent && abs(img_true[i])>0.001) {
            max_percent = percent;
            true_val = img_true[i];
            predicted = img1[i];
        }
    }
    printMaxAndMin(img_true, imgSize);
  
    return max_percent;//p_error;
}



/*Gets average value of a pixel in a given image/array*/
long avgPixelValue(float* img, int imgSize) {
    long long sum = 0;
    for (int i = 0;i < imgSize;i++) {
        sum += (long long)img[i];
    }
    long avg = (long)sum / ((long)imgSize);
    return avg;
}

/*Counts number of zeroes in an array*/
float zeroCount(float* img, int imgSize) {
    int count = 0;
    for (int i = 0;i < imgSize;i++) {
        if (img[i] <= 0) {
            count++;
        }
    }
    return count;
}


/*Fills array with random numbers. For testing purposes*/
void fillWithRandomNumbers(float* arr, int arrSize) {
    for (int i = 0; i < arrSize;i++) {
        arr[i] = rand() % 100 + 1;
    }

}

/*Tests if two arrays are equal. For testing purposes*/
bool checkEquality(float* arr1, float* arr2, int arrSize) {
    bool equal = true;
    for (int i = 0;i < arrSize;i++) {
        if (arr1[i] != arr2[i]) {
            equal = false;
        }
    }
    return equal;
}

/*CPU implementation of getting difference between images (sum of absolute difference between images)*/
float MatrixAbsDiffSum(float* arr1, float* arr2, int arrSize) {
    float absDiff = 0;

    for (int i = 0;i < arrSize;i++) {
        absDiff += abs(arr1[i] - arr2[i]);
    }
    return absDiff;
}



int main(int argc, char *argv[])
{
    printf("%s Starting...\n\n", argv[0]);

    try
    {
       
        long long start_time = start_timer();
        int imgDimX = 2048;
        int imgDimY = 4176;
        int imgSize =  imgDimX * imgDimY;

        npp::ImageCPU_32f_C1 oHostSrc(imgDimX,imgDimY);//32-bit image


        float* originalImg = new float[imgSize];//input image
        float* compareImg= new float[imgSize];//Resulting image for comparison to our output
        float* corrOrigin = new float[imgSize]; //resulting correlation image, which produced by the brighter-fatter algorithm and added to the orignal image
        std::ofstream result("gpu_result_diff.txt");//loading result into text file
        std::ofstream actual("gpu_OG_diff.txt");

        unsigned int srcWidth = oHostSrc.width();
        unsigned int srcHeight = oHostSrc.height();



        /*Filling the arrays above with values from text files:

          Ideally, I would have binded this code to Python and pass in the image arrays directly.
          However, to simplify things I simply saved images as text files and load them into arrays manually
          This takes time but it should be not counted as part of the actual brighter-fatter correction time
          since I'm assuming that images are already loaded.

        */
        std::fstream file;
        std::string word, t, q, filename, compareFilename, corrFilename;
        
        // filename of the file
        filename = "inputImgOG.txt";//input image textfile
        compareFilename = "finalImgOG.txt";//output image for comparison
        corrFilename = "corr.txt"; //output correlation image for comparison

        file.open(filename.c_str());

        //Loading input image textfile into array
        if (file.is_open()) {
            for (int i = 0; i < imgSize;i++) {
                file >> word;
                int n = word.length();
                char char_array[n + 1];
                strcpy(char_array, word.c_str());
                char* pEnd;
                float wordfloat = strtod(char_array, &pEnd);
                oHostSrc.data()[i] = wordfloat;
                originalImg[i] = wordfloat;
            }

        }
        std::cout << "Input image loaded" << std::endl;
        file.close();

        //Loading output comparison image
        file.open(compareFilename.c_str());
        if (file.is_open()) {
            for (int i = 0;i < imgSize;i++) {
                file >> word;
                int n = word.length();
                char char_array[n + 1];
                strcpy(char_array, word.c_str());
                char* pEnd;
                float wordfloat = strtod(char_array, &pEnd);
                compareImg[i] = wordfloat;
            }
        }
        std::cout << "Output image to be compared loaded" << std::endl;
        file.close();

    
        file.open(corrFilename.c_str());
        if (file.is_open()) {
            for (int i = 0;i < imgSize;i++) {
                file >> word;
                int n = word.length();
                char char_array[n + 1];
                strcpy(char_array, word.c_str());
                char* pEnd;
                float wordfloat = strtod(char_array, &pEnd);
                corrOrigin[i] = wordfloat;
            }
        }
        std::cout << "Correlation image to be compared loaded" << std::endl;

        file.close();


        


        //--Brighter-Fatter Correction Begins-------------------------------------------------------------------------        

        long long convTimer = start_timer();

        npp::ImageNPP_32f_C1 oDeviceSrc(oHostSrc);

        NppiSize kernelSize = { 17,17 };

        NppiSize oSizeROI = { imgDimX, imgDimY };
        NppiSize oSrcSize = { imgDimX, imgDimY };
        //NppiPoint oSrcOffset = {0,0};
        //NppiBorderType eBorderType = NPP_BORDER_REPLICATE;

        npp::ImageNPP_32f_C1 oDeviceDst(oSizeROI.width, oSizeROI.height);//Device memory for convolved (output) image
        npp::ImageCPU_32f_C1 oHostDst(oDeviceDst.size());//Host memory destination for resulting image
        NppiPoint oAnchor = { kernelSize.width / 2, kernelSize.height / 2 }; //oAnchor is the point on the kernel we use to determine its position relative to the image
        NppStatus eStatusNPP;

        std::cout << "kernelSize.width: " << kernelSize.width / 2 << "/kernelSize.height: " << kernelSize.height / 2 << std::endl;
        int hostKSize = kernelSize.width * kernelSize.height;
        Npp32f hostKernel[hostKSize];
        std::cout << "Host kernel size: " << hostKSize << std::endl;
        fillKernelArray("bfKernel.txt", hostKernel, hostKSize);

        std::cout << "Loaded PGM Image Data First row vs Image Data from Text File: " << hostKSize << std::endl;


        //Npp32f* deviceKernel;
        Npp32f* deviceKernel;
        size_t deviceKernelPitch;
 
        cudaMalloc((void**)&deviceKernel, kernelSize.width * kernelSize.height * sizeof(Npp32f));
        cudaMemcpy(deviceKernel, hostKernel, kernelSize.width * kernelSize.height * sizeof(Npp32f), cudaMemcpyHostToDevice);
        Npp32f divisor = 1; // no scaling
        
        std::cout << "Calculated size: " << kernelSize.width * kernelSize.height * sizeof(Npp32f) << std::endl;
        std::cout << "Device kernel size: " << sizeof(deviceKernel) << std::endl;
        std::cout << "hostKernel size: " << sizeof(hostKernel) << std::endl;

        //How pitch is calculated: (size of pixel) * (image width)
        int devPitch = sizeof(float) * imgDimX;//number of bytes per row
        int dstPitch = sizeof(float) * imgDimX;//number of bytes per row

        std::cout << "devPitch:" << devPitch << "   dstPitch:" << dstPitch << std::endl;
  
        
        std::cout << "Source image Line Step (bytes) " << devPitch << std::endl;
        std::cout << "Destination Image line step (bytes): " << dstPitch << std::endl;
        //std::cout << "ROI: " << oSizeROI << std::endl;
        //std::cout << "Device Kernel: " << deviceKernel << std::endl;
        //std::cout << "Kernel Size: " << kernelSize << std::endl;
        //std::cout << "X and Y offsets of kernel origin frame: " << oAnchor << std::endl;


        std::cout << "Convolution Step For 32-bit: " << std::endl;
        
        int startX = 8;
        int endX = -9;//in Python, this is the 9th element from the right-hand side (startY:endY, where endY is non-inclusive.(so up to 10th element from the end)
        int startY = 8;
        int endY = -9;

        int maxIter = 10;
        
        //GPU CONSTANTS
        const int threadsPerBlock = 1024;
        const int blocksPerGrid = 8352;

        //dimension constants
        int grad_R = (R - startY + endY);//#of rows of resulting gradient image
        int grad_C = (C - startX + endX);//# cols
        int grad_N = grad_R * grad_C; 
        int first_N = (grad_R - 2) * (grad_C - 2);

        int diff_R_0 = (R - startY + endY);//#rows of resulting diff image
        int diff_C_0 = (C - (startX + 1) + (endX - 1));
        int diff_R_1 = (R - (startY + 1) + (endY - 1));
        int diff_C_1 = (C - startX + endX);
        
        //relevant diff array size vars
        int diffOut20_N = diff_R_0 * diff_C_0;
        int diffOut21_N = diff_R_1 * diff_C_1;
        int diffOut20_N_full = N - C;
        int diffOut21_N_full = N - R;
        int diffOut20_N_full2 = N - (2 * C);
        int diffOut21_N_full2 = N - (2 * R);
        int second_N = diff_R_1 * diff_C_0;

        //--------------------------------------------------------------------//
        //**DEVICE VARIABLES**//       
        float* gradTmp_dev_0 = 0;
        float* gradTmp_dev_1 = 0;
        float* gradOut_dev_0 = 0;
        float* gradOut_dev_1 = 0;
        float* first = 0;

        float* diffOut20 = 0;
        float* diffOut21 = 0;
        float* diffOut20_temp = 0;
        float* diffOut20_temp2 = 0;
        float* diffOut21_temp = 0;
        float* diffOut21_temp2 = 0;
        float* second = 0;
    
        float* corr = 0;
        float* corr_host = (float*)malloc(N * sizeof(float));//host memory to finish computation of diff and comparison with threshold

        float* tmpArray = 0;
        float* tmpArray_spliced = 0;

        float* outArray = 0;
        float* outArray_spliced = 0;
        float* prev_image = 0; 

        float* diff_arr = 0;
        float* diff_arrHost = (float*)malloc(blocksPerGrid * sizeof(float));//host memory to finish computation of diff and comparison with threshold
        int threshold = 1000;//this value was retrieved from available test files. (Of course, this may not be constant)

        //image
        float* image = 0;
        //Final vars
        float* final_image = (float*)malloc(N * sizeof(float));//host memory to finish computation of diff and comparison with threshold
        
        

        //DEVICE MEMORY ALLOCATION//
        cudaError_t cudaStatus;

        cudaStatus = cudaSetDevice(0);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
            goto Error;
        }

        //original image allocation
        cudaStatus = cudaMalloc((void**)&image, N * sizeof(float));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed! 11");
            goto Error;
        }

        //memory allocation of gradient variables -- start
        cudaStatus = cudaMalloc((void**)&gradTmp_dev_0, grad_N * sizeof(float));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed! 1");
            goto Error;
        }

        cudaStatus = cudaMalloc((void**)&gradTmp_dev_1, grad_N * sizeof(float));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed! 2");
            goto Error;
        }
        
        cudaStatus = cudaMalloc((void**)&gradOut_dev_0, grad_N * sizeof(float));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed! 3");
            goto Error;
        }

        cudaStatus = cudaMalloc((void**)&gradOut_dev_1, grad_N * sizeof(float));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed! 4");
            goto Error;
        }

        cudaStatus = cudaMalloc((void**)&first, first_N* sizeof(float));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed! 5");
            goto Error;
        }

        cudaStatus = cudaMalloc((void**)&tmpArray_spliced, grad_N * sizeof(float));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed! 4");
            goto Error;
        }

        cudaStatus = cudaMalloc((void**)&outArray_spliced, grad_N * sizeof(float));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed! 4");
            goto Error;
        }
        //^^^ end of gradient variables


        //start of diff variables ----vvv
        cudaStatus = cudaMalloc((void**)&diffOut20, diffOut20_N* sizeof(float));//final value, second derivative matrix
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed! 6");
            goto Error;
        }

        cudaStatus = cudaMalloc((void**)&diffOut20_temp, diffOut20_N_full * sizeof(float));//temp for intermediate first derivative step
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed! 6");
            goto Error;
        }
        cudaStatus = cudaMalloc((void**)&diffOut20_temp2, diffOut20_N_full2 * sizeof(float));//final value, second derivative matrix
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed! 6");
            goto Error;
        }

        cudaStatus = cudaMalloc((void**)&diffOut21, diffOut21_N* sizeof(float));//final value, second derivative matrix
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed! 7");
            goto Error;
        }

        cudaStatus = cudaMalloc((void**)&diffOut21_temp, diffOut21_N_full * sizeof(float)); //temp for intermediate first derivative step
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed! 7");
            goto Error;
        }

        cudaStatus = cudaMalloc((void**)&diffOut21_temp2, diffOut21_N_full2 * sizeof(float)); //temp for intermediate first derivative step
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed! 7");
            goto Error;
        }


        cudaStatus = cudaMalloc((void**)&second, second_N* sizeof(float));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed! 8");
            goto Error;
        }
        //end of diff variables ^^^------------


        //corr allocation
        cudaStatus = cudaMalloc((void**)&corr, N* sizeof(float));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed! 9");
            goto Error;
        }


        cudaStatus = cudaMalloc((void**)&prev_image, N * sizeof(float));//prev_image same size as input image, (tmpArray)
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed! 10");
            goto Error;
        }

        cudaStatus = cudaMalloc((void**)&tmpArray, N * sizeof(float));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed! 11");
            goto Error;
        }

        cudaStatus = cudaMalloc((void**)&outArray, N * sizeof(float));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed! 12");
            goto Error;
        }


        //difference step variables v v v v-----
        cudaStatus = cudaMalloc((void**)&diff_arr, blocksPerGrid * sizeof(float));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed! 13");
            goto Error;
        }
        //---------------------------------


        cudaStatus = cudaMemcpy(image, originalImg, sizeof(float) * N, cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "init image cudaMemcpy failed!");
            goto Error;
        }
        cudaStatus = cudaMemcpy(tmpArray, oDeviceSrc.data(), sizeof(float) * N, cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "init tmpArray cudaMemcpy failed!");
            goto Error;
        }


	    for(int i = 0 ;i<maxIter;i++){

	        std::cout<<"Iteration: "<<i<<std::endl;

            /*Single Channel Filter for 32 bit image and 32 bit kernel 
            (precision loss possible, since kernel from LSST code is originally 64 bit)

            
                Docs: https://docs.nvidia.com/cuda/npp/group__image__filter.html#gaf354d8f2b2c134503abaee1f004cef37
            */
            
            /*output of convolution array = oDeviceDst.data()*/

            eStatusNPP = nppiFilter_32f_C1R(tmpArray, devPitch, oDeviceDst.data(),
                dstPitch, oSizeROI, deviceKernel, kernelSize, oAnchor);
            
            outArray = oDeviceDst.data();

            /*Single Channel Filter for 64 bit image and 64 bit kernel

                Docs: https://docs.nvidia.com/cuda/npp/group__image__filter.html#gaf354d8f2b2c134503abaee1f004cef37
            */
            
            /*eStatusNPP = nppiFilter_64f_C1R(tmpArray, devPitch, oDeviceDst.data(),
                dstPitch, oSizeROI, deviceKernel, kernelSize, oAnchor);
            */

            /*32 bit image and 32 bit kernel with border control
            
                Docs: https://docs.nvidia.com/cuda/npp/group__image__filter__border__32f.html
            */

            /*eStatusNPP = nppiFilterBorder_32f_C1R(tmpArray, devPitch, oSrcSize, oSrcOffset, oDeviceDst.data()
                , dstPitch, oSizeROI, deviceKernel, kernelSize, oAnchor, eBorderType);
             */
             //to do: adjust oSizeROI to avoid edges

            cudaDeviceSynchronize();//intended to ensure all previous steps are completed before the next step

            /*
                Splicing array for subsequent gradient compuation.
                GPU implementation of:

                    tmp_spliced = tmpArray[startY:endY, startX:endX]
                    out_spliced = outArray[startY:endY, startX:endX]

                    Dimensions of spliced arrays: Rows = 4159, Columns = 2031

            */
            CudaWrapper::MatrixGradientSplice_Device(tmpArray_spliced, tmpArray);
            CudaWrapper::MatrixGradientSplice_Device(outArray_spliced, outArray);

            cudaDeviceSynchronize();
            /*gradient computation. GPU implementation of:

                gradTmp = numpy.gradient(tmp_spliced)
                gradOut = numpy.gradient(out_spliced)

                Dimensions of gradients: Rows = 4159, Columns = 2031

            */
            CudaWrapper::MatrixGradientSplice_2D_RowDevice(gradTmp_dev_0, tmpArray_spliced);
            CudaWrapper::MatrixGradientSplice_2D_ColDevice(gradTmp_dev_1, tmpArray_spliced);
            CudaWrapper::MatrixGradientSplice_2D_RowDevice(gradOut_dev_0, outArray_spliced);
            CudaWrapper::MatrixGradientSplice_2D_ColDevice(gradOut_dev_1, outArray_spliced);

            cudaDeviceSynchronize();

            /*first array calculation. GPU implementation of:

                first = (gradTmp[0]*gradOut[0] + gradTmp[1]*gradOut[1])[1:-1, 1:-1]
                
                Dimensions of first array: Rows = 4157, Columns = 2029

            */
            CudaWrapper::MatrixGradientProductSum_Device(first, gradTmp_dev_0, gradTmp_dev_1, gradOut_dev_0, gradOut_dev_1);
            
           /*diff computation. GPU implementation of:

                diffOut20 = numpy.diff(outArray, 2, 0)[startY:endY, startX + 1:endX - 1]
                diffOut21 = numpy.diff(outArray, 2, 1)[startY + 1:endY - 1, startX:endX]

                Dimensions of diff arrays: Rows = 4157, Columns = 2029

           */

            CudaWrapper::MatrixSecondDiff_RowDevice(diffOut20, outArray, diffOut20_temp, diffOut20_temp2);
            CudaWrapper::MatrixSecondDiff_ColDevice(diffOut21, outArray, diffOut21_temp, diffOut21_temp2);

            float* diffTemp0_host = (float*)malloc(diffOut20_N_full * sizeof(float));
            float* diffTemp1_host = (float*)malloc(diffOut21_N_full * sizeof(float));
            float* diffTemp0_host2 = (float*)malloc(diffOut20_N_full2 * sizeof(float));
            float* diffTemp1_host2 = (float*)malloc(diffOut21_N_full2 * sizeof(float));

            cudaDeviceSynchronize();

            /*second array calculation. GPU implementation of:

                second = tmpArray[startY + 1:endY - 1, startX + 1:endX - 1]*(diffOut20 + diffOut21)

                Dimensions of second arrays: Rows = 4157, Columns = 2029

           */
            CudaWrapper::MatrixDiffProductSum_Device(second, tmpArray, diffOut20, diffOut21);

            cudaDeviceSynchronize();

            /*corr array calculation. GPU implementation of:

                corr[startY + 1:endY - 1, startX + 1:endX - 1] = 0.5*(first + second)
           */
            CudaWrapper::MatrixCorrProductSum_Device(corr, first, second);
        
            cudaDeviceSynchronize();

            /*tmpArray addition step: GPU implementation of:

                tmpArray[:, :] = image.getArray()[:, :]
                tmpArray[nanIndex] = 0.
                tmpArray[startY:endY, startX:endX] += corr[startY:endY, startX:endX]
            */
            CudaWrapper::MatrixCopyCorr_Device(tmpArray, image, corr); 

            /*Sum of absolute difference between images calculation. Breaks loop if a certain threshold is met. 
            GPU implementation of:

                if iteration > 0:
                    diff = numpy.sum(numpy.abs(prev_image - tmpArray))

                    if diff < threshold:
                        break
                    prev_image[:, :] = tmpArray[:, :]
            
            */
            if (i > 0) {

                /*Absolute difference sum between "current" and "previous" image. GPU Implementation of: 
                    diff = numpy.sum(numpy.abs(prev_image - tmpArray))
                */
                CudaWrapper::MatrixAbsDiffSumDevice(diff_arr, prev_image, tmpArray);

                //Memory copy from device to host in order to finish previous computation on the CPU
                cudaStatus = cudaMemcpy(diff_arrHost, diff_arr, sizeof(float) * blocksPerGrid, cudaMemcpyDeviceToHost);
                if (cudaStatus != cudaSuccess) {
                    fprintf(stderr, "diff_arrHost cudaMemcpy failed!");
                    goto Error;
                }

                double diff_final = 0;
                for (int k = 0; k < blocksPerGrid; k++) {
                    diff_final += diff_arrHost[k];
                }
                std::cout << "diff:" << diff_final << std::endl;

                //If the absolute difference sum is less than a certain threshold, we no longer need to iterate
                if (diff_final < threshold){
                    break;
                }

                //CudaWrapper::MatrixCopy_Device(prev_image, tmpArray);
                //Copying tmpArray to prev_image, a device to device copy.
                cudaStatus = cudaMemcpy(prev_image, tmpArray, sizeof(float) * N, cudaMemcpyDeviceToDevice);
                if (cudaStatus != cudaSuccess) {
                    fprintf(stderr, "dtod cudaMemcpy failed!");
                    goto Error;
                }

            }


	    }

        /*Adding correlation image to original input image: 
         GPU implementation of:

            image.getArray()[startY + 1:endY - 1, startX + 1:endX - 1] += corr[startY + 1:endY - 1, startX + 1:endX - 1]

        */
        CudaWrapper::matrixIncrementSplice_Device(image, corr);


        std::cout << "NppiFilter error status " << eStatusNPP << std::endl; // prints 0 (no errors) //
 

        cudaStatus = cudaMemcpy(final_image, image, sizeof(float) * N, cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "final_image cudaMemcpy failed!");
            goto Error;
        }

        cudaStatus = cudaMemcpy(corr_host, corr, sizeof(float) * N, cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "corr cudaMemcpy failed!");
            goto Error;
        }

        
        /*The following loads the resulting difference between input and output images into a textfile for visualizing*/
        if (result.is_open())
        {
            for (int row = 0;row < imgDimY;row++) {
                for (int col = 0;col < imgDimX;col++) {
                    int index = row * imgDimX + col;
                    result << abs(final_image[index]-originalImg[index]) << " ";
                }
                result <<" "<<std::endl;
            }
           
            result.close();
        }
        else std::cout << "Unable to open file";
        
        if (actual.is_open())
        {
            for (int row = 0;row < imgDimY;row++) {
                for (int col = 0;col < imgDimX;col++) {
                    int index = row * imgDimX + col;
                    actual << abs(compareImg[index]- originalImg[index]) << " ";
                }
                actual << " " << std::endl;
            }

            actual.close();
        }
        else std::cout << "Unable to open file";
        
      

        //freeing memory
        Error:
        std::cout << "Cuda memory freed" << std::endl;
        cudaFree(gradTmp_dev_0);
        cudaFree(gradTmp_dev_1);
        cudaFree(gradOut_dev_0);
        cudaFree(gradOut_dev_1);
        cudaFree(first);

        cudaFree(diffOut20);
        cudaFree(diffOut21);
        cudaFree(diffOut20_temp);
        cudaFree(diffOut20_temp2);
        cudaFree(diffOut21_temp);
        cudaFree(diffOut21_temp2);
        cudaFree(second);

        cudaFree(corr);

        cudaFree(tmpArray);
        cudaFree(tmpArray_spliced);
        cudaFree(outArray);
        cudaFree(prev_image);

        cudaFree(image);

        cudaFree(diff_arr);
        free(diff_arrHost);

        nppiFree(oDeviceSrc.data());
        nppiFree(oDeviceDst.data());
        //------------------

        cudaDeviceSynchronize();//maybe helps for more accurate timing?
        long long totalConvTime = stop_timer(convTimer, "NPP convolution time:");

        //std::cout << "Inner OUTPUTS of Brighter-Fattered processed DST image: " << std::endl;
        //for (int row = 50;row < 52;row++) {
        //    for (int col = 50;col < 55;col++) {
        //        int index = row * C + col;

        //        std::cout << "CORR Origin: " << "row: " << row << "; col: " << col << "; value: " << corrOrigin[index] << std::endl;
        //        std::cout<<"CORR Added: " << "row: " << row << "; col: " << col << "; value: " << corr_host[index] << std::endl;
        //        std::cout<<"INPUT: " << "row: " << row << "; col: " << col << "; value: " << originalImg[index] << std::endl;
        //        std::cout <<"OUTPUT: "<< "row: " << row << "; col: " << col << "; value: " << final_image[index] << std::endl;
        //        std::cout <<"ACTUAL: "<< "row: " << row << "; col: " << col << "; value: " << compareImg[index] << std::endl;
        //        std::cout << std::endl;

        //    }
        //}

        double mse_image = meanSquaredImages(final_image, compareImg, imgSize);
        double mse_corr = meanSquaredImages(corr_host, corrOrigin, imgSize);
        double maxError_img = maxError(final_image, compareImg, imgSize);
        double maxError_corr = maxBrightPixelError(corr_host, corrOrigin, imgSize);

        std::cout << "Mean squared error of final result: " << mse_image << std::endl;
        std::cout << "Mean squared error of correction matrix: " << mse_corr << std::endl;
        std::cout << "Max error percentage of final result: " << maxError_img << std::endl;
        std::cout << "Max error percentage of correction matrix: " << maxError_corr << std::endl;

    
        long long totalTime = stop_timer(start_time, "Total time:");

        free(corr_host);
        free(final_image);

        exit(EXIT_SUCCESS);
    }
    catch (npp::Exception &rException)
    {
        std::cerr << "Program error! The following exception occurred: \n";
        std::cerr << rException << std::endl;
        std::cerr << "Aborting." << std::endl;

        exit(EXIT_FAILURE);
    }
    catch (...)
    {
        std::cerr << "Program error! An unknow type of exception occurred. \n";
        std::cerr << "Aborting." << std::endl;

        exit(EXIT_FAILURE);
        return -1;
    }

    return 0;
}

// Returns the current time in microseconds
long long start_timer() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000000 + tv.tv_usec;
}

// converts a long long ns value to float seconds
float usToSec(long long time) {
    return ((float)time) / (1000000);
}

// Prints the time elapsed since the specified time
long long stop_timer(long long start_time, const char* name) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    long long end_time = tv.tv_sec * 1000000 + tv.tv_usec;
    float elapsed = usToSec(end_time - start_time);
    printf("%s: %.5f sec\n", name, elapsed);
    return end_time - start_time;
}
