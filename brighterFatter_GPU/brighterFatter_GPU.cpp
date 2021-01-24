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

int R = 4176;
int C = 2048;
int N = R * C;

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
    float sum = 0;

    // extracting words from the file
    if (file.is_open()) {
        for (int i = 0; i < kernelSize;i++) {
            file >> word;
            int n = word.length();
            char char_array[n + 1];
            strcpy(char_array, word.c_str());
            char* pEnd;
            float wordfloat = strtod(char_array, &pEnd);
            kernelArr[i] = wordfloat;
            sum += wordfloat;
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
long long meanSquaredImages(float* img1, float* img2, int imgSize) {
    long long diff = 0;
    for (int i = 0;i < imgSize;i++) {
        long long pix1 = img1[i];
        long long pix2 = img2[i];
        long long pixDiff = pix1 - pix2;
        diff += pixDiff*pixDiff;
    }
    long long mse = diff / imgSize;
    return mse;
}

/*Gets average value of a pixel in a given image/array*/
long long avgPixelValue(float* img, int imgSize) {
    long long sum = 0;
    for (int i = 0;i < imgSize;i++) {
        sum += img[i];
    }
    long long avg = sum / imgSize;
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

        //Code from stack overflow - begin
        std::string fileExtension = ".pgm";
        std::string dirFilename = "fitstest";
        std::string saveFilename = dirFilename + "_8bitBFconvolved";

        npp::ImageCPU_32f_C1 oHostSrc(imgDimX,imgDimY);//32-bit image


        float* originalImg = new float[imgSize];//input image
        float* compareImg= new float[imgSize];//Resulting image for comparison to our output
        float* textConvolved= new float[imgSize]; //Resulting image from GPU implementation
        float* corrOrigin = new float[imgSize]; //resulting correlation image, which produced by the brighter-fatter algorithm and added to the orignal image

        unsigned int srcWidth = oHostSrc.width();
        unsigned int srcHeight = oHostSrc.height();



        /*Filling the arrays above with values from text files*/
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

        //Testing print of original image
        /*std::cout << "Inner INPUTS of original SRC image: " << std::endl;
        for (int row = 50;row < 52;row++) {
            for (int col = 50;col < 55;col++) {
                int index = row * C + col;
                std::cout << "row: " << row << "; col: " << col << "; value: " << oHostSrc.data()[index] << std::endl;
            }
        }*/

        long long convTimer = start_timer();

        npp::ImageNPP_32f_C1 oDeviceSrc(oHostSrc);

        NppiSize kernelSize = { 17,17 };

        //does this not convolve the edges?(Double check this)
        NppiSize oSizeROI = { imgDimX, imgDimY };//{ imgDimX - kernelSize.width+1, imgDimY - kernelSize.height+1 };//{imgDimX, imgDimY};
        
        npp::ImageNPP_32f_C1 oDeviceDst(oSizeROI.width, oSizeROI.height);//Device memory for convolved (output) image
        npp::ImageCPU_32f_C1 oHostDst(oDeviceDst.size());//Host memory destination for resulting image
        NppiPoint oAnchor = { kernelSize.width / 2, kernelSize.height / 2 }; //oAnchor is the point on the kernel we use to determine its position relative to the image
        NppStatus eStatusNPP;

        std::cout << "kernelSize.width: " << kernelSize.width / 2 << "/kernelSize.height: " << kernelSize.height / 2 << std::endl;
        int hostKSize = kernelSize.width * kernelSize.height;
        //Npp32f hostKernel[hostKSize];//= { 0, -1, 0, -1, 5, -1, 0, -1, 0 };//{ 0,0,0,0,1,0,0,0,0 };//Identity kernel to test alignment//{ 0, -1, 0, -1, 5, -1, 0, -1, 0 };//this is emboss//{ -1, 0, 1, -1, 0, 1, -1, 0, 1 }; // convolving with this should do edge detection
        Npp32f hostKernel[hostKSize];
        std::cout << "Host kernel size: " << hostKSize << std::endl;
        fillKernelArray("bfKernel.txt", hostKernel, hostKSize);

        std::cout << "Loaded PGM Image Data First row vs Image Data from Text File: " << hostKSize << std::endl;


        Npp32f* deviceKernel;
        size_t deviceKernelPitch;
 
        cudaMalloc((void**)&deviceKernel, kernelSize.width * kernelSize.height * sizeof(Npp32f));
        cudaMemcpy(deviceKernel, hostKernel, kernelSize.width * kernelSize.height * sizeof(Npp32f), cudaMemcpyHostToDevice);
        Npp32f divisor = 1; // no scaling

        std::cout << "Calculated size: " << kernelSize.width * kernelSize.height * sizeof(Npp32f) << std::endl;
        std::cout << "Device kernel size: " << sizeof(deviceKernel) << std::endl;
        std::cout << "hostKernel size: " << sizeof(hostKernel) << std::endl;

        int devPitch = oDeviceSrc.pitch();
        int dstPitch = oDeviceDst.pitch();

        std::cout << "devPitch:" << devPitch << "   dstPitch:" << dstPitch << std::endl;
  
        //How pitch is calculated: how many bytes in a row? Calcualte by getting bytes in a pixel * image width
        
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

        int second_N = diff_R_1 * diff_C_0;


        //**DEVICE VARIABLES**//       
        float* grad_host = (float*)malloc(grad_N * sizeof(float));
        float* gradTmp_dev_0 = 0;
        float* gradTmp_dev_1 = 0;
        float* gradOut_dev_0 = 0;
        float* gradOut_dev_1 = 0;
        float* first = 0;
        float* first_host = (float*)malloc(first_N * sizeof(float));

        float* diff_host = (float*)malloc(diffOut20_N * sizeof(float));
        float* diffOut20 = 0;
        float* diffOut21 = 0;
        float* diffOut20_temp = 0;
        float* diffOut21_temp = 0;
        float* second = 0;
        float* second_host = (float*)malloc(second_N* sizeof(float));
    
        float* corr = 0;
        float* corr_host = (float*)malloc(N * sizeof(float));//host memory to finish computation of diff and comparison with threshold

        float* tmpArray = 0;
        float* tmpArray_host = (float*)malloc(N * sizeof(float));//new float[imgSize];//for debugging
        float* tmpArray_spliced = 0;
        float* spliced_arr_host = (float*)malloc(N * sizeof(float));
        float* device_src = new float[imgSize];
        bool copy_succ = 0;

        float* outArray = 0;
        float* outArray_host = (float*)malloc(N * sizeof(float));
        float* outArray_spliced = 0;
        float* prev_image = 0; //previous tmpArray (float check initilized to zero?
        float* prev_image_host = (float*)malloc(N * sizeof(float));//new float[imgSize];//for debugging

        float* diff_arr = 0;
        float* diff_arrHost = (float*)malloc(blocksPerGrid * sizeof(float));//host memory to finish computation of diff and comparison with threshold
        int threshold = 1000;//this value was retrieved from available test files. (Of course, this may not be constant)

        //image
        float* image = 0;//(float*)malloc(N * sizeof(float));
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
        cudaStatus = cudaMalloc((void**)&image, N * sizeof(float));//tmpArray has the original device input copied over to it, but this is iteratively modified for the BF procedure
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

        cudaStatus = cudaMalloc((void**)&diffOut21, diffOut21_N* sizeof(float));//final value, second derivative matrix
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed! 7");
            goto Error;
        }

        cudaStatus = cudaMalloc((void**)&diffOut21_temp, diffOut21_N_full * sizeof(float)); //temp for intermeidate first derivative step
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




        //start of corr allocation vvv-----
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

        cudaStatus = cudaMalloc((void**)&tmpArray, N * sizeof(float));//tmpArray has the original device input copied over to it, but this is iteratively modified for the BF procedure
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed! 11");
            goto Error;
        }

        cudaStatus = cudaMalloc((void**)&outArray, N * sizeof(float));//tmpArray has the original device input copied over to it, but this is iteratively modified for the BF procedure
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
        cudaStatus = cudaMemcpy(tmpArray, originalImg, sizeof(float) * N, cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "init tmpArray cudaMemcpy failed!");
            goto Error;
        }
       

	    for(int i = 0 ;i<maxIter;i++){

	        std::cout<<"Iteration: "<<i<<std::endl;
            //previous outArray: oDeviceDst.data()
            eStatusNPP = nppiFilter_32f_C1R(tmpArray, devPitch, oDeviceDst.data(),
                dstPitch, oSizeROI, deviceKernel, kernelSize, oAnchor);

            cudaDeviceSynchronize();//intended to ensure all previous steps are completed before the next step

            outArray = oDeviceDst.data();//this is just a pointer, for readability.(This is never changed)

            cudaStatus = cudaMemcpy(outArray_host, outArray, sizeof(float) * N, cudaMemcpyDeviceToHost);
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "init tmpArray cudaMemcpy failed!");
                goto Error;
            }


            /*
                Splicing array for subsequent gradient compuation.
                GPU implementation of:

                    tmp_spliced = tmpArray[startY:endY, startX:endX]
                    out_spliced = outArray[startY:endY, startX:endX]

            */
            CudaWrapper::MatrixGradientSplice_Device(tmpArray_spliced, tmpArray);
            CudaWrapper::MatrixGradientSplice_Device(outArray_spliced, outArray);

            cudaDeviceSynchronize();


            //saving spliced version of gradTmp_dev_0 for debugging
            cudaStatus = cudaMemcpy(spliced_arr_host, tmpArray_spliced, sizeof(float) * grad_N, cudaMemcpyDeviceToHost);
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "init tmpArray cudaMemcpy failed!");
                goto Error;
            }

            /*gradient computation. GPU implementation of:

                gradTmp = numpy.gradient(tmp_spliced)
                gradOut = numpy.gradient(out_spliced)
            */
            CudaWrapper::MatrixGradientSplice_2D_RowDevice(gradTmp_dev_0, tmpArray_spliced);
            CudaWrapper::MatrixGradientSplice_2D_ColDevice(gradTmp_dev_1, tmpArray_spliced);
            CudaWrapper::MatrixGradientSplice_2D_RowDevice(gradOut_dev_0, outArray_spliced);
            CudaWrapper::MatrixGradientSplice_2D_ColDevice(gradOut_dev_1, outArray_spliced);

            cudaDeviceSynchronize();

            cudaStatus = cudaMemcpy(tmpArray_host, tmpArray, sizeof(float) * N, cudaMemcpyDeviceToHost);
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "diff_arrHost cudaMemcpy failed!");
                goto Error;
            }

            cudaStatus = cudaMemcpy(grad_host, gradTmp_dev_0, sizeof(float) * grad_N, cudaMemcpyDeviceToHost);
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "diff_arrHost cudaMemcpy failed!");
                goto Error;
            }
            cudaStatus = cudaMemcpy(grad_host, gradTmp_dev_1, sizeof(float) * grad_N, cudaMemcpyDeviceToHost);
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "diff_arrHost cudaMemcpy failed!");
                goto Error;
            }
            cudaStatus = cudaMemcpy(grad_host, gradOut_dev_0, sizeof(float) * grad_N, cudaMemcpyDeviceToHost);
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "diff_arrHost cudaMemcpy failed!");
                goto Error;
            }
            cudaStatus = cudaMemcpy(grad_host, gradOut_dev_1, sizeof(float) * grad_N, cudaMemcpyDeviceToHost);
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "diff_arrHost cudaMemcpy failed!");
                goto Error;
            }

            /*first array calculation. GPU implementation of:

                first = (gradTmp[0]*gradOut[0] + gradTmp[1]*gradOut[1])[1:-1, 1:-1]
            */
            CudaWrapper::MatrixGradientProductSum_Device(first, gradTmp_dev_0, gradTmp_dev_1, gradOut_dev_0, gradOut_dev_1);
            
            cudaStatus = cudaMemcpy(first_host, first, sizeof(float) * first_N, cudaMemcpyDeviceToHost);
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "diff_arrHost cudaMemcpy failed!");
                goto Error;
            }


           /*diff computation. GPU implementation of:

                diffOut20 = numpy.diff(outArray, 2, 0)[startY:endY, startX + 1:endX - 1]
                diffOut21 = numpy.diff(outArray, 2, 1)[startY + 1:endY - 1, startX:endX]
           */
            CudaWrapper::MatrixSecondDiff_RowDevice(diffOut20, outArray, diffOut20_temp);
            CudaWrapper::MatrixSecondDiff_ColDevice(diffOut21, outArray, diffOut21_temp);


            cudaStatus = cudaMemcpy(diff_host, diffOut20, sizeof(float) * diffOut20_N, cudaMemcpyDeviceToHost);
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "diff_arrHost cudaMemcpy failed!");
                goto Error;
            }


            cudaDeviceSynchronize();

            /*second array calculation. GPU implementation of:

                second = tmpArray[startY + 1:endY - 1, startX + 1:endX - 1]*(diffOut20 + diffOut21)
           */
            CudaWrapper::MatrixDiffProductSum_Device(second, tmpArray, diffOut20, diffOut21);

            cudaStatus = cudaMemcpy(second_host, second, sizeof(float) * second_N, cudaMemcpyDeviceToHost);
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "diff_arrHost cudaMemcpy failed!");
                goto Error;
            }

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

            cudaStatus = cudaMemcpy(tmpArray_host, tmpArray, sizeof(float) * N, cudaMemcpyDeviceToHost);
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "diff_arrHost cudaMemcpy failed!");
                goto Error;
            }
            cudaStatus = cudaMemcpy(first_host, first, sizeof(float) * first_N, cudaMemcpyDeviceToHost);
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "diff_arrHost cudaMemcpy failed!");
                goto Error;
            }

            cudaStatus = cudaMemcpy(corr_host, corr, sizeof(float) * N, cudaMemcpyDeviceToHost);
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "diff_arrHost cudaMemcpy failed!");
                goto Error;
            }


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

                cudaStatus = cudaMemcpy(prev_image_host, prev_image, sizeof(float) * N, cudaMemcpyDeviceToHost);
                if (cudaStatus != cudaSuccess) {
                    fprintf(stderr, "diff_arrHost cudaMemcpy failed!");
                    goto Error;
                }
                cudaStatus = cudaMemcpy(tmpArray_host, tmpArray, sizeof(float) * N, cudaMemcpyDeviceToHost);
                if (cudaStatus != cudaSuccess) {
                    fprintf(stderr, "diff_arrHost cudaMemcpy failed!");
                    goto Error;
                }
                
                double diff_final = 0;
                for (int i = 0; i < blocksPerGrid; i++) {
                    diff_final += diff_arrHost[i];
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


        std::cout << "NppiFilter error status " << eStatusNPP << std::endl; // prints 0 (no errors) //-6 is NPP_SIZE_ERROR (ROI Height or ROI width are negative)
 

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
        cudaFree(second);

        cudaFree(corr);

        cudaFree(tmpArray);
        cudaFree(outArray);
        cudaFree(prev_image);

        cudaFree(diff_arr);
        free(diff_arrHost);
        //------------------

        cudaDeviceSynchronize();//maybe helps for more accurate timing?
        long long totalConvTime = stop_timer(convTimer, "NPP convolution time:");


        int badappleCount = 0;
        std::set<int> badRows;
        std::cout << "RESULTING HOST IMAGE DATA from oDeviceSrc:"<<std::endl;
        for (int i = 0;i < imgSize;i++) {//just prints first row of values

            //float p32bit = final_image[i]//oHostDst.data()[i];
            
            textConvolved[i] = final_image[i];//oHostDst.data()[i];
 
            if (abs(final_image[i] - compareImg[i]) > 1) {//classified as "bad" if predicted differs by more than 10 from actual (I THINK)


                //std::cout << "8-bit convolved data: " << p8bit << std::endl;
                //std::cout << "32-bit convolved data:" << p32bit << "\n" << std::endl;
                int row = i / (imgDimX);
                //int col = i % imgDimX //- 1;
                //std::cout << "Row: " << row << " / " << "Column: " << col << std::endl;
                badRows.insert(row);

                badappleCount++;
            }
        }


        int discardRows = imgDimX * 30;//discard 30 rows
        int inspectRows = imgDimX * 472;

        std::cout << "Inner OUTPUTS of Brighter-Fattered processed DST image: " << std::endl;
        for (int row = 50;row < 52;row++) {
            for (int col = 50;col < 55;col++) {
                int index = row * C + col;

                std::cout << "CORR Origin: " << "row: " << row << "; col: " << col << "; value: " << corrOrigin[index] << std::endl;
                std::cout<<"CORR Added: " << "row: " << row << "; col: " << col << "; value: " << corr_host[index] << std::endl;
                std::cout<<"INPUT: " << "row: " << row << "; col: " << col << "; value: " << originalImg[index] << std::endl;
                std::cout <<"OUTPUT: "<< "row: " << row << "; col: " << col << "; value: " << final_image[index] << std::endl;
                std::cout <<"ACTUAL: "<< "row: " << row << "; col: " << col << "; value: " << compareImg[index] << std::endl;
                std::cout << std::endl;

            }
        }



        double badappleCount_db = badappleCount;

        //bad apples are outlying values that screw everything up. Could be edge pixels.
        std::cout << "Bad apple count: " << badappleCount << std::endl;
        std::cout << "Bad apple percentage of image: " << badappleCount_db / imgSize * 100 << std::endl;
        //std::cout << "Bad rows contains:";
        //for (std::set<int>::iterator it = badRows.begin(); it != badRows.end(); ++it)
        //    std::cout << ' ' << *it;
        
        long long mse = meanSquaredImages(textConvolved, compareImg, imgSize);
        long long avgConvolvedPixel = avgPixelValue(compareImg, inspectRows);
        long long percentage = (sqrt(mse) / avgConvolvedPixel) *100;
        float convZeroCount = zeroCount(compareImg, imgSize);

        std::cout << "avgConvolvedPixel: " << avgConvolvedPixel << std::endl;
        std::cout << "Zero count: " << convZeroCount << std::endl;
        //float mseWithOriginal = meanSquaredImages(originalImg, compareImg, imgSize);
        std::cout << "Mean squared error: " << mse << std::endl;
        std::cout << "Root MSE: " << sqrt(mse) << std::endl;
        std::cout << "Root MSE as a percentage of avg destination pixel: " << percentage << std::endl;
        //std::cout << "MSE between original input image and compareImg: " << mseWithOriginal << std::endl;
        //std::cout << "Root MSE between original input image and compareImg: " << sqrt(mseWithOriginal) << std::endl;
        

        //end code from SO
        nppiFree(oDeviceSrc.data());
        nppiFree(oDeviceDst.data());
    
        long long totalTime = stop_timer(start_time, "Total time:");


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
