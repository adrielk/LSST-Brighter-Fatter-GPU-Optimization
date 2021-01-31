//Author: Adriel Kim
//1-31-2021
/*Description: 
Contains wrappers of kernels used to optimize brighter-fatter correction.
Kernels parallelize functions such as numpy.gradient, numpy.diff, numpy.abs, numpy.sum, and matrix splicing.
*/

#include "matrix_ops.cuh"
#include <iostream>

#define R 4176 //Image rows
#define C 2048 //Image columns
#define N (R*C)



/*Constants for gradient calculation:

    gradTmp = numpy.gradient(tmpArray[startY:endY, startX:endX])
    gradOut = numpy.gradient(outArray[startY:endY, startX:endX])

*/
#define startX 8
#define endX (C-9)
#define startY 8
#define endY (R-9)
#define grad_R (endY - startY)//used in gradient kernel
#define grad_C (endX - startX)//used in gradient kernel
#define grad_N (grad_R * grad_C)//size of gradient array
//---------------------------------------------------------------^^^gradient kernel


//#define start_flat ((startY)*R + startX)
//#define end_flat ((endY)*R + endX)


/*Constants for "first" array calculation:
    
    first = (gradTmp[0]*gradOut[0] + gradTmp[1]*gradOut[1])[1:-1, 1:-1]

*/
#define first_R (grad_R - 2)
#define first_C (grad_C - 2)
#define first_N (first_R * first_C)

/*Constants for diff calculation and "second" array calculation:

      diffOut20 = numpy.diff(outArray, 2, 0)[startY:endY, startX + 1:endX - 1]
      diffOut21 = numpy.diff(outArray, 2, 1)[startY + 1:endY - 1, startX:endX]
      second = tmpArray[startY + 1:endY - 1, startX + 1:endX - 1]*(diffOut20 + diffOut21)

*/
#define diff_R_0 (endY - startY)
#define diff_C_0 ((endX - 1) - (startX + 1))
#define diff_R_1 ((endY - 1) - (startY + 1))
#define diff_C_1 (endX - startX)
#define N_axis0_2 ((R*C)-C)//# of elements after 1st diff, axis = 0 (row)
#define N_axis1_2 ((R*C)-R) //# of elements after 1st diff, axis = 1 (col)
#define diffOut20_N (diff_R_0 * diff_C_0)//#after 2nd diff
#define diffOut21_N (diff_R_1 * diff_C_1)



const int threadsPerBlock = 1024;//threads in a block. A chunk that shares the same shared memory.
const int blocksPerGrid = 8352;//imin(32, (N + threadsPerBlock - 1) / threadsPerBlock);

using namespace std;


//Kernel simple examples//
__global__ void matrixAddKernel(float* c, const float* a, const float* b) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    while (tid < N) {
        c[tid] = a[tid] + b[tid];
        //adds total number of running threads to tid, the current index.
        tid += blockDim.x * gridDim.x;
    }
}
__global__ void matrixSubtractKernel(float* c, const float* a, const float* b) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    while (tid < N) {
        c[tid] = a[tid] - b[tid];
        //adds total number of running threads to tid, the current index.
        tid += blockDim.x * gridDim.x;
    }
}
__global__ void matrixMultiplyKernel(float* c, const float* a, const float* b) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    while (tid < N) {
        c[tid] = a[tid] * b[tid];
        tid += blockDim.x * gridDim.x;
    }
}

__global__ void matrixDivideKernel(float* c, const float* a, const float* b) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    while (tid < N) {
        c[tid] = (a[tid] / b[tid]);
        tid += blockDim.x * gridDim.x;
    }
}



//Brighter-fatter correction relevant kernels://

/*Horizontal axis (axis = 0) 1st diff. Implementation of numpy.diff*/
__global__ void matrixDiffKernel_2D_Horiz(float* c, const float* a) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x; //current index
    int tid2 = tid + C;//"adjacent" row index
    while (tid2 < N) {
        c[tid] = (a[tid2] - a[tid]);

        tid += blockDim.x * gridDim.x;
        tid2 = tid + C;
    }

}


__global__ void matrixDiffSplice_2D_Horiz(float* c, const float* a) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x; //current index
    //note: this splices the second diff, so the matrix is 4174 x 2048. We want to splice down to 4157 x 2029

    while (tid < N_axis0_2 - C) {

        int tid_row = tid / C;
        int tid_col = tid % C;

        //splice condition, based on resulting "coordinate space". (-2 is to correct for matrix shrinkage after two diffs)
        if (tid_row >= startY && tid_row < (endY-2) && tid_col>=(startX+1) && tid_col<(endX-1)) {
            int spliced_row = tid_row - (startY);
            int spliced_col = tid_col - (startX + 1);
            int spliced_index = spliced_row * (diff_C_0)+spliced_col;
            c[spliced_index] = a[tid];
        }

        tid += blockDim.x * gridDim.x;

    }


}

/*Horizontal axis (axis = 0) 2nd diff. Implementation of numpy.diff*/
__global__ void matrixDiffKernel_2D_Horiz_Iter2(float* c, const float* a) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x; //current index
    
    while (tid < N_axis0_2 - C) {//test this with and without this...
    
        int tid_next = tid + C;
        c[tid] = (a[tid_next] - a[tid]);

        tid += blockDim.x * gridDim.x;

    }

}


/*Vertical axis (axis = 1) 1st diff. Implementation of numpy.diff*/
__global__ void matrixDiffKernel_2D_Vert(float* c, const float* a) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x; //current index
    
    while (tid < N) {
        
        int tid_row = tid / C;
        int tid_row_next = (tid + 1) / C;
        if (tid_row == tid_row_next) {//elements are along the same row
            c[tid - tid_row] = a[tid + 1] - a[tid];
        }

        tid += blockDim.x * gridDim.x;
    }

}


__global__ void matrixDiffSplice_2D_Vert(float* c, const float* a) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x; //current index
    //note: this splices the second diff, so the matrix is 4176 x 2046. We want to splice down to 4157 x 2029
    int cols = C - 2;

    while (tid < N) {

        int tid_row = tid / cols;
        int tid_col = tid % cols;

        if (tid_row >= (startY + 1) && tid_row < (endY - 1) && tid_col >= startX && tid_col < endX-2) {
            int spliced_row = tid_row - (startY+1);
            int spliced_col = tid_col - (startX);
            int spliced_index = spliced_row * (diff_C_0)+spliced_col;
            c[spliced_index] = a[tid];
        }
        tid += blockDim.x * gridDim.x;
    }


}

/*Vertical axis (axis = 1) 2nd diff. Implementation of numpy.diff*/ //DUH, just reuse the first iteration
__global__ void matrixDiffKernel_2D_Vert_Iter2(float* c, const float* a) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x; //current index
    int cols = C - 1;//loses a column for second iteration

    while (tid < N) {

        int tid_row = tid / cols;
        int tid_row_next = (tid + 1) / cols;
        if (tid_row == tid_row_next) {//elements are along the same row
            c[tid - tid_row] = a[tid + 1] - a[tid];
        }

        tid += blockDim.x * gridDim.x;
    }

}


/*GPU implementation of:
    diff = numpy.sum(numpy.abs(prev_image - tmpArray))
*/
__global__ void matrixAbsDiffSum(float* c, const float* a, const float* b) {
    __shared__ float cache[threadsPerBlock];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIndex = threadIdx.x;

    float temp = 0;
    while (tid < N) {
        temp += abs(a[tid] - b[tid]);//absolute differnece
        tid += blockDim.x * gridDim.x;
    }

    cache[cacheIndex] = temp;

    __syncthreads();

    int i = blockDim.x / 2;
    while (i != 0) {
        if (cacheIndex < i) {
            cache[cacheIndex] += cache[cacheIndex + i];
        }
        __syncthreads();
        i = i / 2;
    }
    if (cacheIndex == 0) {
        c[blockIdx.x] = cache[0];
    }
}

//Array splicing for gradient step
__global__ void matrixGradientSplice(float* grad, float* tmpArray) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x; //current index

    //we are "iterating" over every tmpArray element
    while (tid < N) {
        int tid_row_N = tid / C;
        int tid_col_N = tid % C;
        int tid_linear_N = (tid_row_N * C) + tid_col_N;

        if (tid_row_N >= startY && tid_row_N < endY && tid_col_N >= startX && tid_col_N < endX) {
            int tid_row_grad = tid_row_N - startY;//shifts starting point. so first splice instance maps to 0
            int tid_col_grad = tid_col_N - startX;//same here ^^
            int tid_linear_grad = (tid_row_grad * grad_C) + tid_col_grad;//grad_C + tid_col_grad;

            grad[tid_linear_grad] = tmpArray[tid_linear_N];
        }

        tid += blockDim.x * gridDim.x;

    }


}

/*2D implementation of numpy.gradient along row axis (axis = 0).

    Used for GPU implementation of:
        gradTmp = numpy.gradient(tmpArray[startY:endY, startX:endX])
        gradOut = numpy.gradient(outArray[startY:endY, startX:endX])

*/
__global__ void matrixGradientKernel_2D_Row(float* c, const float* a) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x; //current index
    
    int tid_row = tid / grad_C;
    int tid_col = tid % grad_C;
    
    if (tid_row == 0) {//if first row the gradient is simply the difference between the following row and first row
        c[tid] = a[tid + grad_C] - a[tid];
    }
    else if (tid_col == grad_C-1) {//tid is a last row index
        c[tid] = a[tid] - a[tid - grad_C];
    }
    
    
    while (tid < grad_N - (2*grad_C)) {
        
        int tid_row_next = tid_row+2;
        int linear_index_next = (tid_row_next*grad_C)+tid_col;
        int linear_index_current = (tid_row * grad_C)+tid_col;
        c[tid + grad_C] = (a[linear_index_next] - a[linear_index_current])/2;
        
        tid += blockDim.x * gridDim.x;
        tid_row = tid/grad_C;
        tid_col = tid%grad_C;
    }

}

/*2D implementation of numpy.gradient along vertical axis (axis = 1).

    Used for GPU implementation of:
        gradTmp = numpy.gradient(tmpArray[startY:endY, startX:endX])
        gradOut = numpy.gradient(outArray[startY:endY, startX:endX])

*/
__global__ void matrixGradientKernel_2D_Col(float* c, const float* a) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x; //current index
    
    //if column == 0 (first column)
    if (tid % grad_C == 0) {//zero modulus indicates element in first column
        c[tid] = a[tid + 1] - a[tid];
    }
    else if ((tid + 1)%grad_C == 0) {//element at the last column is right before an element in the first column
        c[tid] = a[tid] - a[tid - 1];
    }

    while ( (tid%grad_C) < (grad_C - 2)  && tid < grad_N) {//while column is less than grad_C - 2
        //+1 is index offset, due to exception case for first column indices
        c[tid + 1] = (a[tid + 2] - a[tid]) / 2;

        tid += blockDim.x * gridDim.x;
    }
    

}

/*
   GPU implementation of :
        first = (gradTmp[0]*gradOut[0] + gradTmp[1]*gradOut[1])[1:-1, 1:-1]
*/
__global__ void matrixGradientProductSum(float* c, float* gradTmp_0,float* gradTmp_1 ,float* gradOut_0, float* gradOut_1) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    while (tid < grad_N) {

        int tid_row_grad = tid / grad_C;
        int tid_col_grad = tid % grad_C;

        if (tid_row_grad >= 1 && tid_row_grad < (grad_R - 1) && tid_col_grad >= 1 && tid_col_grad < (grad_C - 1)) {
            int tid_row_first = tid_row_grad - 1;
            int tid_col_first = tid_col_grad - 1;
            int linear_index = (tid_row_first * (diff_C_0)) + tid_col_first;//first_C + tid_col_first;

            c[linear_index] = gradTmp_0[tid] * gradOut_0[tid] + gradTmp_1[tid] * gradOut_1[tid];

        }
        tid += blockDim.x * gridDim.x;
    }

   
}

/*
   GPU implementation of :
        second = tmpArray[startY + 1:endY - 1, startX + 1:endX - 1]*(diffOut20 + diffOut21)
*/
__global__ void matrixDiffProductSum(float* dev_c, float* tmpArray, float* diffOut20, float* diffOut21) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    //note: the final spliced array (dev_c) will have the same size as diffOut20_N
    while (tid < N) {

        int tid_row = tid / C;
        int tid_col = tid % C;

        if (tid_row >= (startY + 1) && tid_row < (endY - 1) && tid_col >= (startX + 1) && tid_col < (endX - 1)) {
            int tid_row_spliced = tid_row - (startY + 1);
            int tid_col_spliced = tid_col - (startX + 1);
            int linear_index = (tid_row_spliced * diff_C_0) + tid_col_spliced;

            dev_c[linear_index] = tmpArray[tid]*(diffOut20[linear_index]+diffOut21[linear_index]);//dev_c and diffOut20 and diffOut21 are the same size!
        }

        tid += blockDim.x * gridDim.x;
    }

}

/*
   GPU implementation of :
        corr[startY + 1:endY - 1, startX + 1:endX - 1] = 0.5*(first + second)
*/
__global__ void matrixCorrProductSum(float* dev_c, float* first, float* second) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    //corr has same full size as original input array, according to lsst code: "corr = numpy.zeros_like(image.getArray())"
    while (tid < N) {
        
        int tid_row = tid / C;
        int tid_col = tid % C;

        if (tid_row >= (startY + 1) && tid_row < (endY - 1) && tid_col >= (startX + 1) && tid_col < (endX - 1)) {
            int tid_row_offset = tid_row - (startY + 1);
            int tid_col_offset = tid_col - (startX + 1);
            int linear_index = (tid_row_offset * diff_C_0) + tid_col_offset;//tid_row_offset * C + tid_col_offset;

            dev_c[tid] = 0.5 * (first[linear_index] + second[linear_index]);//hopefully this is right index
        }
        else {
            dev_c[tid] = 0;
        }

        tid += blockDim.x * gridDim.x;
    }

}

/*For copying matrices*/
__global__ void matrixCopy(float* dev_c, float* dev_a) {
   int tid = threadIdx.x + blockIdx.x * blockDim.x;
   
   while (tid < N) {

       if (isnan(dev_a[tid]) == true) {
           dev_c[tid] = 0;
       }
       else {
           dev_c[tid] = dev_a[tid];
       }

       tid += blockDim.x * gridDim.x;
   }

}

/*GPU implementation of:

    tmpArray[:, :] = image.getArray()[:, :]
    tmpArray[nanIndex] = 0.
    tmpArray[startY:endY, startX:endX] += corr[startY:endY, startX:endX]

*/
__global__ void matrixCopyCorr(float* dev_c, float* image, float* corr){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    while(tid<N){
        int tid_row = tid/C;
        int tid_col = tid%C;
        
        if(isnan(dev_c[tid]) == false){
            dev_c[tid] = image[tid];
        }else{
            dev_c[tid] = 0;
        }

        if(tid_row >= startY && tid_row < endY && tid_col>=startX && tid_col< endX){
            dev_c[tid] = dev_c[tid]+corr[tid];
        }

        tid += blockDim.x*gridDim.x;
    }
}


/*GPU implementation of:

    image.getArray()[startY + 1:endY - 1, startX + 1:endX - 1] += corr[startY + 1:endY - 1, startX + 1:endX - 1]
*/
__global__ void matrixIncrementSplice(float* imgSrc, float* corr) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    //corr has same full size as original input array, according to lsst code: "corr = numpy.zeros_like(image.getArray())"
    while (tid < N) {

        int tid_row = tid / C;
        int tid_col = tid % C;

        if (tid_row >= (startY + 1) && tid_row < (endY - 1) && tid_col >= (startX + 1) && tid_col < (endX - 1)) {
            imgSrc[tid] += corr[tid];
        }

        tid += blockDim.x * gridDim.x;
    }
   
}


namespace CudaWrapper {

    void MatrixGradientSplice_Device(float* grad, float* tmpArray) {
        matrixGradientSplice << <blocksPerGrid, threadsPerBlock >> > (grad, tmpArray);
    }

    void matrixIncrementSplice_Device(float* imgSrc, float* corr) {
        matrixIncrementSplice << <blocksPerGrid, threadsPerBlock >> > (imgSrc, corr);
    }

    void MatrixCopyCorr_Device(float* tmpArray,float* image, float* corr) {
        matrixCopyCorr << <blocksPerGrid, threadsPerBlock >> > (tmpArray, image, corr);
    }

    void MatrixCopy_Device(float* dev_c, float* dev_a) {
        matrixCopy << <blocksPerGrid, threadsPerBlock >> > (dev_c, dev_a);
    }

    void MatrixCorrProductSum_Device(float* dev_c, float* first, float* second) {
        matrixCorrProductSum<<<blocksPerGrid, threadsPerBlock>>>(dev_c, first, second);
    }

    void MatrixDiffProductSum_Device(float* dev_c, float* tmpArray, float* diffOut20, float* diffOut21) {
        matrixDiffProductSum << <blocksPerGrid, threadsPerBlock >> > (dev_c, tmpArray, diffOut20, diffOut21);
    }

    //temp is missing one row, temp2 is missing 2 rows.
    void MatrixSecondDiff_RowDevice(float* dev_c, float* dev_a, float* temp, float* temp2) {
        matrixDiffKernel_2D_Horiz << <blocksPerGrid, threadsPerBlock >> > (temp, dev_a);
        cudaDeviceSynchronize();
        matrixDiffKernel_2D_Horiz << <blocksPerGrid, threadsPerBlock >> > (temp2, temp);
        cudaDeviceSynchronize();
        matrixDiffSplice_2D_Horiz << <blocksPerGrid, threadsPerBlock >> > (dev_c, temp2);//just splices temp2 down to 4157 x 2029
    }
    
    //temp is missing one column, temp2 is missing 2 columns
    void MatrixSecondDiff_ColDevice(float* dev_c, float* dev_a, float* temp, float* temp2) {
        matrixDiffKernel_2D_Vert << <blocksPerGrid, threadsPerBlock >> > (temp, dev_a);
        cudaDeviceSynchronize();

        matrixDiffKernel_2D_Vert_Iter2 << <blocksPerGrid, threadsPerBlock >> > (temp2, temp);
        cudaDeviceSynchronize();
        matrixDiffSplice_2D_Vert << <blocksPerGrid, threadsPerBlock >> > (dev_c, temp2);//just splices temp2 down to 4157 x 2029

    }

    void MatrixGradientProductSum_Device(float* dev_c, float* gradTmp_dev_0, float* gradTmp_dev_1, float* gradOut_dev_0, float* gradOut_dev_1) {
        matrixGradientProductSum << <blocksPerGrid, threadsPerBlock >> > (dev_c, gradTmp_dev_0, gradTmp_dev_1, gradOut_dev_0, gradOut_dev_1);
    }

    void MatrixGradientSplice_2D_RowDevice(float* dev_c, float* dev_a) {
        matrixGradientKernel_2D_Row << <blocksPerGrid, threadsPerBlock >> > (dev_c, dev_a);
    }  
    
    void MatrixGradientSplice_2D_ColDevice(float* dev_c, float* dev_a) {
        matrixGradientKernel_2D_Col << <blocksPerGrid, threadsPerBlock >> > (dev_c, dev_a);
    }


    void MatrixAbsDiffSumDevice(float* dev_c, float* dev_a, float* dev_b) {
        matrixAbsDiffSum << <blocksPerGrid, threadsPerBlock >> > (dev_c, dev_a, dev_b);
        //note this would require a device copy to finish up the rest of the computation (outside of this kernel wrapper)
    }

}