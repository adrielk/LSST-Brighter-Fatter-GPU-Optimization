#include "matrix_ops.cuh"
#include <iostream>

#define R 4176
#define C 2048
#define N (R*C)//# of elements in matrices (This may become a variable instead of a constant, so no longer defined here but as a parameter..)
#define N_axis0_2 ((R*C)-C)//# of elements after 1st derivative, axis = 0 (row)
#define N_axis1_2 ((R*C)-R) //# of elements after 1st derivative, axis = 1 (col)

//for splicing:(not sure if there is a better, more dynamic way to do this??


//for 2-D gradient kernel-----------------------
#define startX 8
#define endX (C-9)
#define startY 8
#define endY (R-9)
#define grad_R (endY - startY)//used in gradient kernel
#define grad_C (endX - startX)//used in gradient kernel
#define grad_N (grad_R * grad_C)//size of gradient array(double check for debugging)
//---------------------------------------------------------------^^^gradient kernel


#define start_flat ((startY)*R + startX)
#define end_flat ((endY)*R + endX)


//splicing "first" array
//first = (gradTmp[0]*gradOut[0] + gradTmp[1]*gradOut[1])[1:-1, 1:-1]

//variables relevant to gradient kernels
#define first_R (grad_R - 2)
#define first_C (grad_C - 2)
#define first_N (first_R * first_C)//size of "first" array, matrix slice storing product sum of gradient...

//variables relevant to diff kernels
#define diff_R_0 (endY - startY)
#define diff_C_0 ((endX - 1) - (startX + 1))
#define diff_R_1 ((endY - 1) - (startY + 1))
#define diff_C_1 (endX - startX)
#define diffOut20_N (diff_R_0 * diff_C_0)
#define diffOut21_N (diff_R_1 * diff_C_1)

//variables relevant to corr kernels
/*
int corr_R = R - (startY + 1) + (endY - 1);
int corr_C = C - (startX + 1) + (endX - 1);
int corr_N = corr_R * corr_C;
*/
//#define corr_R ((endY - 1) - (startY + 1))
//#define corr_C ((endX - 1) - (startY + 1))
//#define corr_N (corr_R * corr_C) (THIS IS WRONG b/c corr has same shape as input image!)

//#define first_start_flat (first_startY)*first_R + first_startX //flatten indices, relative to gradient matrices
//#define first_end_flat (first_endY)*first_R + first_endX



//test different numbers to optimize matrixAbsDiffSum calculation
const int threadsPerBlock = 1024;//threads in a block. A chunk that shares the same shared memory.
const int blocksPerGrid = 8352;//imin(32, (N + threadsPerBlock - 1) / threadsPerBlock);//this will be our output array size for sumKernel.

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

//naive 1-D array implementation
//To do: 2D diff kernel, with axis = 0 or axis = 1. (may have to just make separate kernels, or an if)
//R = # of "rows" in the array
//C = # of "columns" in the array
//axis = 0, "horizontal/row axis/diff between rows"
//axis = 1, "vertical/column axis/diff between columns"
__global__ void matrixDiffKernel(float* c, const float* a) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x; //current index
    int tid2 = threadIdx.x + 1 + blockIdx.x * blockDim.x;//adjacent index
    while (tid2 < N) {
        c[tid] = (a[tid2] - a[tid]);
        tid += blockDim.x * gridDim.x;//Study why this incrementer is here again...?? (and possibly reduce the redundancies of it(if they exist?)
        tid2 = tid + C;
    }

}







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

//axis = 0/diff between rows
__global__ void matrixDiffKernel_2D_Horiz(float* c, const float* a) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x; //current index
    int tid2 = tid + C;//"adjacent" row index
    while (tid2 < N) {
        c[tid] = (a[tid2] - a[tid]);

        tid += blockDim.x * gridDim.x;
        tid2 = tid + C;
    }

}

//Difference from first iteration: # of rows decrease by 1, and total is N - # of cols
//splice by diffOut20 (second deriv, axis = 0)

/*
#define diff_R_0 (endY - startY)
#define diff_C_0 (endX - 1) - (startX + 1)
#define diff_R_1 (endY - 1) - (startY + 1)
#define diff_C_1 (endX - startX)
#define diffOut20_N (diff_R_0 * diff_C_0)
#define diffOut21_N (diff_R_1 * diff_C_1)*/

//input is from 1st derivative, with original size, but a row missing.
__global__ void matrixDiffKernel_2D_Horiz_Iter2(float* c, const float* a) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x; //current index
    
    while (tid < N_axis0_2 - C) {
    
        int tid_row_diff = tid / C;
        int tid_col_diff = tid % C;
        int tid_next = tid + C;

        //splice condition, based on resulting "coordinate space".
        if (tid_row_diff >= startY && tid_row_diff < endY && tid_col_diff >= (startX + 1) && tid_col_diff < (endX - 1)) {
            
            //corrected coordinates relative to spliced array
            int tid_row_diffSplice = tid_row_diff - startY;
            int tid_col_diffSplice = tid_col_diff - (startX + 1);
            int tid_linear = (tid_row_diffSplice * C) + tid_col_diffSplice;

            c[tid_linear] = (a[tid_next] - a[tid]);
        }


        tid += blockDim.x * gridDim.x;

    }

    /* (THIS IS FUNCTIONALLY EQUIVALENT, but above is more robust)
    int tid2 = tid + C;//"adjacent" row index
    while (tid2 < N_axis0_2) {

        int tid_row = tid / C;
        int tid_col = tid % C;



        //splice condition (BADDD)
        if (tid_row >= startY && tid_row < endY && tid_col >= (startX+1) && tid_col < (endX - 1)) {
            int tid_row_offset = tid_row - startY;
            int tid_col_offset = tid_col - (startX+1);
            int tid_linearized = (tid_row_offset * C) + tid_col_offset;

            c[tid_linearized] = (a[tid2] - a[tid]);

        }
        //c[tid] = (a[tid2] - a[tid]);



        tid += blockDim.x * gridDim.x;
        tid2 = tid + C;
    }
    */
}







//axis = 1/diff between columns (note: differences are only between elements in the same row! so make an exception when traversing to another row!)
__global__ void matrixDiffKernel_2D_Vert(float* c, const float* a) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x; //current index
    
    while (tid < N) {
        
        int tid_row = tid / C;
        int tid_col = tid % C;
        int tid_row_next = (tid + 1) / C;
        int tid_col_next = (tid + 1) % C;
        if (tid_col < (C-1) && tid_row == tid_row_next) {//elements are along the same row
            c[tid - tid_row] = a[tid + 1] - a[tid];
        }

        tid += blockDim.x * gridDim.x;
    }

    /*int tid2 = tid + 1;//"adjacent" row index

    while (tid2 < N) {

        if (tid2 % C != 0) {//This if checks that we are not taking the difference between an element in the subsequent row. (Elements must be from the same row!)
            int index_correction_num = tid / C;
            c[tid - index_correction_num] = (a[tid2] - a[tid]);//might have some "empty" spots with this...check memory allocation
        }
        tid += blockDim.x * gridDim.x;//i believe these "jump" around so that we can still finish the computation despite the lack of computational units.
        tid2 = tid + C;
    }*/

}

//Difference from first iteration: # of columns has shrunken by 1, and total number of elements = N - # of rows
__global__ void matrixDiffKernel_2D_Vert_Iter2(float* c, const float* a) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x; //current index
    int cols = C - 1;
    //int tid2 = tid + 1;//"adjacent" row index

    while (tid < N_axis1_2) {
        int tid_row = tid / cols;
        int tid_col = tid % cols;
        int tid_row_next = (tid + 1) / cols;
        int tid_col_next = (tid + 1) % cols;

        if (tid_col < (cols - 1) && tid_row == tid_row_next) {//elements are along the same row
            
            int tid_row_diff = (tid - tid_row) / (cols - 1);
            int tid_col_diff = (tid - tid_row) % (cols - 1);

            if (tid_row_diff >= (startY + 1) && tid_row_diff < (endY - 1) && tid_col_diff >= startX && tid_col_diff < endX-2) {//endX-2 (-2 is to account for shrinkage)
                int tid_row_diffSpliced = tid_row_diff - (startY + 1);
                int tid_col_diffSpliced = tid_col_diff - (startX);
                int tid_linear = (tid_row_diffSpliced * (cols-1)) + tid_col_diffSpliced; //(cols - 1) + tid_col_diffSpliced;

                c[tid_linear] = a[tid + 1] - a[tid];
            }
        
        
        }

        tid += blockDim.x * gridDim.x;
    }

    /*
    while (tid2 < N_axis1_2) {

        //this is most likely faulty. I need to account for the changing array size. Maybe it's easier to implement the splicing separately....
        if (tid2 % (C - 1) != 0) {
            int index_correction = tid / (C - 1);
            int tid_corrected = tid - index_correction;

            int tid_row = tid_corrected / (C - 1);
            int tid_col = tid_corrected % (C - 1);

            //splice condition
            if (tid_row >= (startY + 1) && tid_row < (endY - 1) && tid_col >= startX && tid_col < endX) {
                int tid_row_offset = tid_row - (startY + 1);
                int tid_col_offset = tid_col - startX;

                int tid_linearized = (tid_row_offset * (C-1)) + tid_col_offset;


                c[tid_linearized] = (a[tid2] - a[tid]);

            }
        }

        tid += blockDim.x * gridDim.x;//i believe these "jump" around so that we can still finish the computation despite the lack of computational units.
        tid2 += blockDim.x * gridDim.x;
    }*/

}






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

//note that the gradient also requires a slightly different procedure, specific to only the first and last values of the array. (can simply be "hard-coded")
//This is done, only left to do is optimize (if that's even necessary?)
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

//takes gradient of 2D matrix along row direction ▼ (axis = 0) (instead of base index, is there like a base row?)
//resulting gradient will be of the same shape as original input

//same row traversal as row diff...

//NOTE, this array is differnet sized than N (FIX SPLCIING."a" is spliced before hand!)****
//start in tmpArray space (of size N)
//Check if within spliced bounds
//Convert splice to offset coordinates
//"c" must be size of spliced array.
__global__ void matrixGradientKernel_2D_Row(float* c, const float* a) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x; //current index
    
    int tid_row = tid / grad_C;
    int tid_col = tid % grad_C;
    
    if (tid_row == 0) {//if first row (gradient is simply the difference)
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
        //c[tid + grad_C] = (a[tid + (2 * grad_C)] - a[tid]) / 2;

        
        tid += blockDim.x * gridDim.x;
        tid_row = tid/grad_C;
        tid_col = tid%grad_C;
    }


    /*
    //int tid2 = tid + C;//"adjacent" row index
    int tid_row_N = tid / C;
    int tid_col_N = tid % C;
    int tid_linear_N = tid_row_N * C + tid_col_N;
    //int tid_linear_N_next = (tid_row_N + 1) * C + tid_col_N;

    //we will only do something if this main condition is met.(Otherwise, treat as if it doesn't exist)
    if (tid_row_N >= startY && tid_row_N < endY && tid_col_N >= startX && tid_col_N < endX) {
        int tid_row_grad = tid_row_N - startY;//shifts starting point. so first splice instance maps to 0
        int tid_col_grad = tid_col_N - startX;//same here ^^
        int tid_linear_grad = tid_row_grad * grad_C + tid_col_grad;
        //^^ now, we're in the splice "coordinate space" as if it were its own array

        if (tid_row_grad == 0) {//if first row, the gradient is simply the difference between subsequent row
            int tid_linear_N_next = (tid_row_N + 1) * C + tid_col_N;
            c[tid_linear_grad] = a[tid_linear_N_next] - a[tid_linear_N];
        }
        else if (tid_row_grad == (grad_R - 1)) {//if last row, gradient is difference between previous and current row
            int tid_linear_N_prev = (tid_row_N - 1) * C + tid_col_N;
            c[tid_linear_grad] = a[tid_linear_N] - a[tid_linear_N_prev];
        }

        while (tid_linear_grad < grad_N - (2*grad_C)) {
            int tid_linear_grad_offset = grad_C;//offset so as to not conflict with previously inserted element in first rows
            int tid_linear_N_next2 = (tid_row_N + 2) * C + tid_col_N;//element 2 rows from tid_linear_N;

            c[tid_linear_grad + tid_linear_grad_offset] = (a[tid_linear_N_next2] - a[tid_linear_N])/2;//mean of current and value in subsequent 2 rows
            
            tid_linear_grad += blockDim.x * gridDim.x;//not entirely sure if this is valid...just be aware.
        }
       

    }*/

}

//takes gradient of 2D matrix along column direction -> (axis = 1)
//NOTE: SAME SPLICING ISSUE AS KERNEL ABOVE. MUST FIX AND TEST :(****
__global__ void matrixGradientKernel_2D_Col(float* c, const float* a) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x; //current index
    //int tid2 = tid + 1;//"adjacent" row index
    /*
    int tid_row_N = tid / C;
    int tid_col_N = tid % C;
    int tid_linear_N = tid_row_N * C + tid_col_N;

    if (tid_row_N >= startY && tid_row_N < endY && tid_col_N >= startX && tid_col_N < endX) {
        int tid_row_grad = tid_row_N - startY;//shifts starting point. so first splice instance maps to 0
        int tid_col_grad = tid_col_N - startX;//same here ^^
        int tid_linear_grad = tid_row_grad * grad_C + tid_col_grad;

        //if column == 0 (first column)
        if (tid_col_grad == 0) {
            int tid_linear_N_next = (tid_row_N)*C + (tid_col_N + 1);
            c[tid_linear_grad] = a[tid_linear_N_next] - a[tid_linear_N];
        }
        else if (tid_col_grad == (C - 1)) {
            int tid_linear_N_prev = (tid_row_N)*C + (tid_col_N - 1);
            c[tid_linear_grad] = a[tid_linear_N] - a[tid_linear_N_prev];
        }

        while (tid_col_grad < (grad_C - 2)) {
            int tid_linear_grad_offset = 1;//offsets by 1 column, so as to not collide with column 0
            c[tid_linear_grad + tid_linear_grad_offset] = (a[tid_linear_N + 2] - a[tid_linear_N]) / 2;//difference between element 2 cols from current and current element

            tid_linear_grad += blockDim.x * gridDim.x;
            tid_col_grad = tid_linear_grad % grad_C;
        }


    }
    */

    
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

        tid += blockDim.x * gridDim.x;//hmmm not entirely sure still
    }
    

}

//traversal size iis based off first array size (first_N)
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

    /*
    int tid_row = tid/first_C;
    int tid_col = tid % first_C;

    //skip last row and last column
    //first parens, checks not first column nor last column
    //second parens, checks not first row nor last row
    while (tid < grad_N) {
        
        if (tid_row > 0 && tid_row < first_R - 1 && tid_col > 0 && tid_col < first_C - 1) {
            int tid_row_offset = (tid_row - 1);
            int tid_col_offset = (tid_col - 1);
            int linear_index = tid_row_offset * first_C + tid_col_offset;

            c[linear_index] = gradTmp_0[tid] * gradOut_0[tid] + gradTmp_1[tid] * gradOut_1[tid];

        }
        tid += blockDim.x * gridDim.x;
        tid_row = tid / first_C;
        tid_col = tid % first_C;

    }*/
}







//this is to be used for testing purposes(separates splicing and actual operation)
/*
__global__ void matrixDiffProductSumSplice(float* dev_c, float* tmpArray) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    //note: this final spliced array will have the same size as diffOut20_N
    while (tid < N) {
        
        int tid_row = tid / C;
        int tid_col = tid % C;

        if (tid_row >= (startY + 1) && tid_row < (endY - 1) && tid_col >= (startX + 1) && tid_col < (endX - 1)) {
            int tid_row_spliced = tid_row - (startY + 1);
            int tid_col_spliced = tid_col - (startX + 1);
            int linear_index = tid_row_spliced * diff_C_0 + tid_col_spliced;//careful what you use for your column size...

            dev_c[linear_index] = tmpArray[tid];

        }

        tid += blockDim.x * gridDim.x;
    }
}
*/


__global__ void matrixDiffProductSum(float* dev_c, float* tmpArray, float* diffOut20, float* diffOut21) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    //note: this final spliced array will have the same size as diffOut20_N
    while (tid < N) {

        int tid_row = tid / C;
        int tid_col = tid % C;

        if (tid_row >= (startY + 1) && tid_row < (endY - 1) && tid_col >= (startX + 1) && tid_col < (endX - 1)) {
            int tid_row_spliced = tid_row - (startY + 1);
            int tid_col_spliced = tid_col - (startX + 1);
            int linear_index = (tid_row_spliced * 2029) + tid_col_spliced;//careful what you use for your column size...

            //dev_c[linear_index] = tmpArray[tid] * (diffOut20[tid] + diffOut21[tid]);
            dev_c[linear_index] = tmpArray[tid]*(diffOut20[linear_index]+diffOut21[linear_index]);//dev_c and diffOut20 and diffOut21 are the same size!
        }

        tid += blockDim.x * gridDim.x;
    }



    /*while (tid < diffOut20_N) {
        dev_c[tid] = tmpArray_spliced[tid] * (diffOut20[tid] + diffOut21[tid]);
        
        tid += blockDim.x * gridDim.x;
    }*/

    /*
    while (tid < N) {

        int tid_row = tid / C;
        int tid_col = tid % C;

        if (tid_row >= (startY+1) && tid_row < (endY-1) && tid_col >= (startX+1) && tid_col < (endX-1)) {
            int tid_row_offset = tid_row - (startY+1);
            int tid_col_offset = tid_col - (startX+1);
            int linear_index = tid_row_offset * C + tid_col_offset;

            dev_c[linear_index] = tmpArray[tid] * (diffOut20[tid] + diffOut21[tid]);

        }
        tid += blockDim.x * gridDim.x;
    }*/
}

/*
__global__ void matrixCorrProductSum(float* dev_c, float* first, float* second){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    while(tid < N){
        int tid_row = tid/C;
        int tid_col = tid%C;
        
        if(tid_row>=startY+1 && tid_row < endY-1 && tid_col >= startX+1 && tid_col < endX-1){
            int tid_row_offset = tid_row - (startY+1);
            int tid_col_offset = tid_col - (startX+1);
            int linear_index = tid_row_offset * diff_C_0 + tid_col_offset;
            
            dev_c[tid] = 0.5 * (first[linear_index] + second[linear_index]);
            
        }
        else{
            dev_c[tid] = 0;
        }
        
        tid += blockDim.x * gridDim.x;
    }
}*/


__global__ void matrixCorrProductSum(float* dev_c, float* first, float* second) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    //corr has same full size as original input array, according to lsst code: "corr = numpy.zeros_like(image.getArray())"
    while (tid < N) {
        
        int tid_row = tid / C;
        int tid_col = tid % C;

        if (tid_row >= (startY + 1) && tid_row < (endY - 1) && tid_col >= (startX + 1) && tid_col < (endX - 1)) {
            int tid_row_offset = tid_row - (startY + 1);
            int tid_col_offset = tid_col - (startX + 1);
            int linear_index = (tid_row_offset * 2029) + tid_col_offset;//tid_row_offset * C + tid_col_offset;

            dev_c[tid] = 0.5 * (first[linear_index] + second[linear_index]);//hopefully this is right index
        }
        else {
            dev_c[tid] = 0;
        }

        tid += blockDim.x * gridDim.x;
    }

}

//copy kernels

__global__ void matrixCopy(float* dev_c, float* dev_a) {
   int tid = threadIdx.x + blockIdx.x * blockDim.x;
   
   while (tid < N) {

       if (isnan(dev_a[tid]) == true) {//nan numbers will be copied over as zero.
           dev_c[tid] = 0;//test this with bogus to see if you get a bug...
       }
       else {
           dev_c[tid] = dev_a[tid];
       }

       tid += blockDim.x * gridDim.x;
   }

}

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


/*
//just a simple index shift, and array size modifcation from the original gradient function.
__global__ void matrixGradientSplice(float* c, const float* a) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int baseIndex = tid - start_flat;


    if (baseIndex == 0) {
        c[baseIndex] = (a[1] - a[0]);
    }
    else if (baseIndex == grad_N - 1) {//note, that it's important N is defined exactly as the image size....
        c[baseIndex] = (a[baseIndex] - a[baseIndex - 1]);
    }

    while (baseIndex >= 0 && baseIndex < grad_N - 2) {//possibility of base index being negative...

        c[baseIndex + 1] = (a[baseIndex + 2] - a[baseIndex]) / 2;
        baseIndex += blockDim.x * gridDim.x;

    }

}
*/

/*
//c is variable for first array, slightly shrunked from "a" and "b", the numpy gradient arrays
//not sure if this kernel still iteratates relative to original array.... hmmm (or it's constrained by condiiton)
__global__ void matrixFirstSplice(float* c, const float* gradTmp, const float *gradOut) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int baseIndex = tid - first_start_flat;

    while (baseIndex >= 0 && baseIndex < first_N) {//possibility of base index being negative...

        //nanni...(this isn't what python is doing...) XXXXX
        c[baseIndex] = c[tid] * (gradTmp[0] * gradOut[0] + gradTmp[1] * gradOut[1]);
        baseIndex += blockDim.x * gridDim.x;

    }

}*/


//later, you should actually review namespaces, class synax, wrappers in c++....
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

    //note: image is source input
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

    void MatrixSecondDiff_RowDevice(float* dev_c, float* dev_a, float* temp) {
        //dev_c stores 2nd spliced derivative, dev_temp stores first derivative, dev_a is the original input image
        matrixDiffKernel_2D_Horiz << <blocksPerGrid, threadsPerBlock >> > (temp, dev_a);
        cudaDeviceSynchronize();
        matrixDiffKernel_2D_Horiz_Iter2 << <blocksPerGrid, threadsPerBlock >> > (dev_c, temp);
        //cudaDeviceSynchronize();
        //matrixDiffRowSplice<<<blocksPerGrid, threadsPerBlock>>>(dev_c, temp);
    }

    void MatrixSecondDiff_ColDevice(float* dev_c, float* dev_a, float* temp) {
        matrixDiffKernel_2D_Vert << <blocksPerGrid, threadsPerBlock >> > (temp, dev_a);
        cudaDeviceSynchronize();
        matrixDiffKernel_2D_Vert_Iter2 << <blocksPerGrid, threadsPerBlock >> > (dev_c, temp);
    }

    void MatrixGradientProductSum_Device(float* dev_c, float* gradTmp_dev_0, float* gradTmp_dev_1, float* gradOut_dev_0, float* gradOut_dev_1) {
        
        matrixGradientProductSum << <blocksPerGrid, threadsPerBlock >> > (dev_c, gradTmp_dev_0, gradTmp_dev_1, gradOut_dev_0, gradOut_dev_1);
        //matrixGradientProductSum(float* c, float* gradTmp_0, float* gradTmp_1, float* gradOut_0, float* gradOut_1)
    
    }

    void MatrixGradientSplice_2D_RowDevice(float* dev_c, float* dev_a) {
        matrixGradientKernel_2D_Row << <blocksPerGrid, threadsPerBlock >> > (dev_c, dev_a);
    }  
    
    
    void MatrixGradientSplice_2D_ColDevice(float* dev_c, float* dev_a) {
        matrixGradientKernel_2D_Col << <blocksPerGrid, threadsPerBlock >> > (dev_c, dev_a);
    }

    /*void MatrixGradientSpliceDevice(float* dev_c, float* dev_a) {
        matrixGradientSplice << <blocksPerGrid, threadsPerBlock >> > (dev_c, dev_a);
    }*/

    void MatrixAbsDiffSumDevice(float* dev_c, float* dev_a, float* dev_b) {
        matrixAbsDiffSum << <blocksPerGrid, threadsPerBlock >> > (dev_c, dev_a, dev_b);
        //note this would require a device copy to finish up the rest of the computation (outside of this kernel wrapper)
        //NOTE, however this is implemented, it will always require a device copy...
        //So if this happens to be slow, try: https://www.eximiaco.tech/en/2019/06/10/implementing-parallel-reduction-in-cuda/ 
    }

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









    //JUNK that might work later??? Although, not really necessary.-----------------------------------
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
