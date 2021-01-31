//Author: Adriel Kim
//1-31-2021
//Header file for CUDA kernel wrappers

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

namespace CudaWrapper{

	void MatrixAbsDiffSumDevice(float* dev_c, float* dev_a, float* dev_b);

	//functions relevant to gradient step of Brighter-Fatter
	void MatrixGradientSplice_Device(float* grad, float* tmpArray);
	void MatrixGradientSplice_2D_RowDevice(float* dev_c, float* dev_a);
	void MatrixGradientSplice_2D_ColDevice(float* dev_c, float* dev_a);
	void MatrixGradientProductSum_Device(float* dev_c, float* gradTmp_dev_0, float* gradTmp_dev_1, float* gradOut_dev_0, float* gradOut_dev_1);

	//functions relevant to the diff step of the Brighter_Fatter
	void MatrixSecondDiff_RowDevice(float* dev_c, float* dev_a, float* temp, float* temp2);
	void MatrixSecondDiff_ColDevice(float* dev_c, float* dev_a, float* temp, float* temp2);
	void MatrixDiffProductSum_Device(float* dev_c, float* tmpArray, float* diffOut20, float* diffOut21);

	//functions relevant to corr step of Brighter-Fatter
	void MatrixCorrProductSum_Device(float* dev_c, float* first, float* second);
	
	//data copy functions:
	void MatrixCopy_Device(float* dev_c, float* dev_a);
	void MatrixCopyCorr_Device(float* tmpArray, float* image, float* corr);

	//final step
	void matrixIncrementSplice_Device(float* imgSrc, float* corr);


}