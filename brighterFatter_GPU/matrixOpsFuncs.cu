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

/*1D array implementation of numpy.diff*/
__global__ void matrixDiffKernel(float* c, const float* a) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x; //current index
    int tid2 = threadIdx.x + 1 + blockIdx.x * blockDim.x;//adjacent index
    while (tid2 < N) {
        c[tid] = (a[tid2] - a[tid]);
        tid += blockDim.x * gridDim.x;
        tid2 = tid + C;
    }

}

/*For splicing resulting diff array. Not used*/
__global__ void matrixDiffRowSplice(float* dev_c, float* diffOut20) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x; //current index
    
    while (tid < N_axis0_2 - C) {
        int tid_row = tid / C;
        int tid_col = tid % C;

        if (tid_row >= startY && tid_row < endY && tid_col >= (startX + 1) && tid_col < (endX - 1)) {
            int tid_row_diffSplice = tid_row - startY;
            int tid_col_diff = tid_col - (startX + 1);
            int tid_linear = (tid_row_diffSplice * C) + tid_col_diff;
            
            dev_c[tid_linear] = diffOut20[tid];
        }


        tid += blockDim.x * gridDim.x;
    }

}

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
    
        int tid_row_diff = tid / C;
        int tid_col_diff = tid % C;
        int tid_next = tid + C;

        //splice condition, based on resulting "coordinate space". (-2 is to correct for matrix shrinkage after two diffs)
        //if (tid_row_diff >= startY && tid_row_diff < endY-2 && tid_col_diff >= (startX + 1) && tid_col_diff < (endX - 1)) {
            
            //corrected coordinates relative to spliced array
            //int tid_row_diffSplice = tid_row_diff - startY;
            //int tid_col_diffSplice = tid_col_diff - (startX + 1);
            //int tid_linear = (tid_row_diffSplice * (diff_C_0)) + tid_col_diffSplice;
        
            c[tid] = (a[tid_next] - a[tid]);
        
        //}


        tid += blockDim.x * gridDim.x;

    }

}


/*Vertical axis (axis = 1) 1st diff. Implementation of numpy.diff*/
__global__ void matrixDiffKernel_2D_Vert(float* c, const float* a) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x; //current index
    
    while (tid < N) {
        
        int tid_row = tid / C;
        int tid_col = tid % C;
        int tid_row_next = (tid + 1) / C;
        //int tid_col_next = (tid + 1) % C;
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
        int tid_col = tid % cols;
        int tid_row_next = (tid + 1) / cols;
        //int tid_col_next = (tid + 1) % C;
        if (tid_row == tid_row_next) {//elements are along the same row
            c[tid - tid_row] = a[tid + 1] - a[tid];
        }

        tid += blockDim.x * gridDim.x;
    }

}
//__global__ void matrixDiffKernel_2D_Vert_Iter2(float* c, const float* a) {
//    int tid = threadIdx.x + blockIdx.x * blockDim.x; //current index
//    int cols = C - 1;
//
//    while (tid < N){//N_axis1_2 - R) { Beware, this bound is sketchy, since rows cannot simply be cut off since they are ingrained in a 1D array...
//        int tid_row = tid / cols;
//        int tid_col = tid % cols;
//        int tid_row_next = (tid + 1) / cols;
//        //int tid_col_next = (tid + 1) % cols;
//        //establish splice boundary. Must also bound tid_row_next and check if it's on the same row as current index
//        //if (tid_row >= (startY + 1) && tid_row < (endY - 1) && tid_col >= startX && tid_col < endX - 2 && tid_row_next <(endY-1) && tid_row_next == tid_row) {
//
//            
//            if (tid_row_next == tid_row) {//must be on same row for this
//                //int tid_row_diffSpliced = tid_row - (startY + 1);
//                //int tid_col_diffSpliced = tid_col - (startX);
//                //int tid_linear = (tid_row_diffSpliced * (diff_C_0)) + tid_col_diffSpliced;
//
//                //c[tid_linear - tid_row_diffSpliced] = a[tid + 1] - a[tid];
//
//                c[tid - tid_row] = a[tid + 1] - a[tid];
//
//                //debug by setting to index to tid (see outArray's indices)
//                //then try what skadron suggested
//            }
//
//
//        
//        //}
//        
//        tid += blockDim.x * gridDim.x;
//    }
//
//}

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


/*1D numpy.gradient implementation. Not used*/
__global__ void matrixGradientKernel(float* c, const float* a) {
    int baseIndex = threadIdx.x + blockIdx.x * blockDim.x;

    //int tid = baseIndex + 1;//threadIdx.x + 1 + blockIdx.x * blockDim.x;//plus one, shifts so we start at c[1]
    //int tid2 = tid + 2;

    if (baseIndex == 0) {
        c[baseIndex] = (a[1] - a[0]);
    }
    else if (baseIndex == N - 1) {//note, that it's important N is defined exactly as the image size....
        c[baseIndex] = (a[baseIndex] - a[baseIndex - 1]);
    }

    while (baseIndex < N - 2) {

        //+1 is an index offset, due to the exception case for the first element
        c[baseIndex + 1] = (a[baseIndex + 2] - a[baseIndex]) / 2;
        baseIndex += blockDim.x * gridDim.x;

    }

}

/*1D implementation of numpy.gradient while splicing. Not used*/
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
           int tid_linear_grad = (tid_row_grad * grad_C)+tid_col_grad;//grad_C + tid_col_grad;

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

    void MatrixDiffRowSplice_Device(float* dev_c, float* diffOut20) {
        matrixDiffRowSplice << <blocksPerGrid, threadsPerBlock >> > (dev_c, diffOut20);
    }

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
        //dev_c stores 2nd spliced diff, dev_temp stores first derivative, dev_a is the original input image
        matrixDiffKernel_2D_Horiz << <blocksPerGrid, threadsPerBlock >> > (temp, dev_a);
        cudaDeviceSynchronize();
        //matrixDiffKernel_2D_Horiz_Iter2 << <blocksPerGrid, threadsPerBlock >> > (dev_c, temp);
        matrixDiffKernel_2D_Horiz << <blocksPerGrid, threadsPerBlock >> > (temp2, temp);//be aware of bounds... not sure if it'd be an actual problem tho.
        cudaDeviceSynchronize();
        matrixDiffSplice_2D_Horiz << <blocksPerGrid, threadsPerBlock >> > (dev_c, temp2);//just splices temp2 down to 4157 x 2029
    }
    
    //temp is missing one column, temp2 is missing 2 columns
    void MatrixSecondDiff_ColDevice(float* dev_c, float* dev_a, float* temp, float* temp2) {
        matrixDiffKernel_2D_Vert << <blocksPerGrid, threadsPerBlock >> > (temp, dev_a);
        cudaDeviceSynchronize();
        //matrixDiffKernel_2D_Vert_Iter2 << <blocksPerGrid, threadsPerBlock >> > (dev_c, temp);
        //matrixDiffKernel_2D_Vert << <blocksPerGrid, threadsPerBlock >> > (temp2, temp);
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






    /*Everything below this line was not used for the brighter-fatter computation*/
    
    float MatrixAbsDiffSum(float* floatMatrix, float* floatMatrix2) {

        float* outputs = (float*)malloc(blocksPerGrid * sizeof(float));
        float total_abs_diff = 0;

        float* dev_a = 0;
        float* dev_b = 0;
        float* dev_c = 0;
        cudaError_t cudaStatus;


        cudaStatus = cudaSetDevice(0);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
            goto Error;
        }

        //Allocate GPU buffers for three vectors (two input, one output)
        cudaStatus = cudaMalloc((void**)&dev_c, blocksPerGrid * sizeof(float));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed! 1");
            goto Error;
        }

        cudaStatus = cudaMalloc((void**)&dev_a, N * sizeof(float));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed! 2");
            goto Error;
        }

        cudaStatus = cudaMalloc((void**)&dev_b, N * sizeof(float));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed! 3");
            goto Error;
        }

        cudaStatus = cudaMemcpy(dev_a, floatMatrix, sizeof(float) * N, cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed! 1");
            goto Error;
        }

        cudaStatus = cudaMemcpy(dev_b, floatMatrix2, sizeof(float) * N, cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed! 2");
            goto Error;
        }

        matrixAbsDiffSum << <blocksPerGrid, threadsPerBlock >> > (dev_c, dev_a, dev_b);

        cudaStatus = cudaMemcpy(outputs, dev_c, sizeof(float) * blocksPerGrid, cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed! 3");
            goto Error;
        }

        for (int i = 0;i < blocksPerGrid;i++) {
            total_abs_diff += outputs[i]; //CPU has to finish up the job...
        }


    Error:
        cout << "Cuda memory freed" << endl;
        cudaFree(dev_c);
        cudaFree(dev_a);
        cudaFree(dev_b);

        return total_abs_diff;


    }


    void MatrixDiffDevice(float* dev_c, float* dev_a, int axis) {
        if (axis == 0) {
            matrixDiffKernel_2D_Horiz << <blocksPerGrid, threadsPerBlock >> > (dev_c, dev_a);
            matrixDiffKernel_2D_Horiz_Iter2 << <blocksPerGrid, threadsPerBlock >> > (dev_c, dev_c);
        }
        else {//axis == 1
            matrixDiffKernel_2D_Vert << <blocksPerGrid, threadsPerBlock >> > (dev_c, dev_a);
            matrixDiffKernel_2D_Vert_Iter2 << <blocksPerGrid, threadsPerBlock >> > (dev_c, dev_c);
        }
    }

    //this is a 2-iteration diff, as used by the LSST brighter-fatter function.
    void MatrixDiff(float* floatMatrix, float* outputs, int axis) {
        float* dev_a = 0;
        float* dev_c = 0;
        cudaError_t cudaStatus;


        cudaStatus = cudaSetDevice(0);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
            goto Error;
        }

        //Allocate GPU buffers for three vectors (two input, one output)
        cudaStatus = cudaMalloc((void**)&dev_c, N * sizeof(float));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed! 1");
            goto Error;
        }

        cudaStatus = cudaMalloc((void**)&dev_a, N * sizeof(float));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed! 2");
            goto Error;
        }

        cudaStatus = cudaMemcpy(dev_a, floatMatrix, sizeof(float) * N, cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed! 1");
            goto Error;
        }

        if (axis == 0) {
            printf("First iteration\n");
            matrixDiffKernel_2D_Horiz << <blocksPerGrid, threadsPerBlock >> > (dev_c, dev_a);
            printf("Second iteration\n");
            matrixDiffKernel_2D_Horiz_Iter2 << <blocksPerGrid, threadsPerBlock >> > (dev_c, dev_c);
        }
        else {//axis == 1
            printf("First iteration\n");
            matrixDiffKernel_2D_Vert << <blocksPerGrid, threadsPerBlock >> > (dev_c, dev_a);
            //printf("dev_c[0]: %d\n", dev_c[0]);
            printf("Second iteration\n");
            matrixDiffKernel_2D_Vert_Iter2 << <blocksPerGrid, threadsPerBlock >> > (dev_c, dev_c);
            //printf("dev_c[0]: %d\n", dev_c[0]);

        }

        cudaStatus = cudaMemcpy(outputs, dev_c, sizeof(float) * N, cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed! 3");
            goto Error;
        }

        cout << ("Wrapper function finished TEST") << endl;

    Error:
        cout << "Cuda memory freed" << endl;
        cudaFree(dev_c);
        cudaFree(dev_a);
    }

    /*void MatrixGradientDevice(float* dev_a, float* dev_c) {
        //this function assumes device variables (cuda mallac and all that stuff were done)
        matrixGradientKernel << <blocksPerGrid, threadsPerBlock >> > (dev_c, dev_a);

    }*/

    void MatrixGradientDevice(float* dev_c, float* dev_a) {
        matrixGradientKernel << <blocksPerGrid, threadsPerBlock >> > (dev_c, dev_a);
    }

    void MatrixGradient(float* floatMatrix, float* outputs) {
        float* dev_a = 0;
        float* dev_c = 0;
        cudaError_t cudaStatus;


        cudaStatus = cudaSetDevice(0);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
            goto Error;
        }

        //Allocate GPU buffers for three vectors (two input, one output)
        cudaStatus = cudaMalloc((void**)&dev_c, N * sizeof(float));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed! 1");
            goto Error;
        }

        cudaStatus = cudaMalloc((void**)&dev_a, N * sizeof(float));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed! 2");
            goto Error;
        }

        cudaStatus = cudaMemcpy(dev_a, floatMatrix, sizeof(float) * N, cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed! 1");
            goto Error;
        }

        matrixGradientKernel << <blocksPerGrid, threadsPerBlock >> > (dev_c, dev_a);

        cudaStatus = cudaMemcpy(outputs, dev_c, sizeof(float) * N, cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed! 3");
            goto Error;
        }

        cout << ("Wrapper function finished TEST") << endl;

    Error:
        cout << "Cuda memory freed" << endl;
        cudaFree(dev_c);
        cudaFree(dev_a);
    }

    void MatrixAdd(float* floatMatrix, float* floatMatrix2, float *outputs) {
        
        float* dev_a = 0;
        float* dev_b = 0;
        float* dev_c = 0;
        /*
        cudaError_t stat = GetCudaStatus();

        DeviceAllocation(dev_a, stat);
        DeviceAllocation(dev_b, stat);
        DeviceAllocation(dev_c, stat);

        HostToDevice(dev_a, floatMatrix, stat);
        HostToDevice(dev_b, floatMatrix2, stat);
        */

        cudaError_t cudaStatus;


        cudaStatus = cudaSetDevice(0);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
            goto Error;
        }

        //Allocate GPU buffers for three vectors (two input, one output)
        cudaStatus = cudaMalloc((void**)&dev_c, N * sizeof(float));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed! 1");
            goto Error;
        }

        cudaStatus = cudaMalloc((void**)&dev_a, N * sizeof(float));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed! 2");
            goto Error;
        }

        cudaStatus = cudaMalloc((void**)&dev_b, N * sizeof(float));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed! 3");
            goto Error;
        }

        cudaStatus = cudaMemcpy(dev_a, floatMatrix, sizeof(float) * N, cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed! 1");
            goto Error;
        }

        cudaStatus = cudaMemcpy(dev_b, floatMatrix2, sizeof(float) * N, cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed! 2");
            goto Error;
        }
        
        matrixAddKernel << <blocksPerGrid, threadsPerBlock >> > (dev_c, dev_a, dev_b);

        /*
        DeviceToHost(dev_c, outputs, stat);

        FreeDevice(dev_c);
        FreeDevice(dev_a);
        FreeDevice(dev_b);
        */

        
        cudaStatus = cudaMemcpy(outputs, dev_c, sizeof(float) * N, cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed! 3");
            goto Error;
        }

        cout<<("Wrapper function finished TEST")<<endl;

        Error:
        cout << "Cuda memory freed" << endl;
        cudaFree(dev_c);
        cudaFree(dev_a);
        cudaFree(dev_b);
        
    }



    cudaError_t GetCudaStatus() {
        cudaError_t cudaStatus;
        cudaStatus = cudaSetDevice(0);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
            //return;
        }
        return cudaStatus;
    }

    void FreeDevice(float* dev) {
        cout << "Cuda memory freed" << endl;
        cudaFree(dev);
    }

    void DeviceToHost(float* dev, float* outputs, cudaError_t cudaStatus) {
        //cudaError_t cudaStatus;
        //cudaStatus = cudaSetDevice(0);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
            goto Error;
        }


        cudaStatus = cudaMemcpy(outputs, dev, sizeof(float) * N, cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed! 3");
            goto Error;
        }
        Error:
        cout << "Cuda memory freed" << endl;
        cudaFree(dev);

    }

    void DeviceAllocation(float* dev, cudaError_t cudaStatus) {
        //cudaError_t cudaStatus;
        //cudaStatus = cudaSetDevice(0);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
            goto Error;
        }

        cudaStatus = cudaMalloc((void**)&dev, N * sizeof(float));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed! 1");
            goto Error;
        }

        Error:
        cout << "Cuda memory freed" << endl;
        cudaFree(dev);
    }

    void HostToDevice(float* dev, float* input, cudaError_t cudaStatus) {
        //cudaError_t cudaStatus;
        //cudaStatus = cudaSetDevice(0);//just make sure using a different cuda status all the time isn't deterimental....
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
            goto Error;
        }

        cudaStatus = cudaMemcpy(dev, input, sizeof(float) * N, cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed! 1");
            goto Error;
        }

        Error:
        cout << "Cuda memory freed" << endl;
        cudaFree(dev);
    }
}