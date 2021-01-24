#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

namespace CudaWrapper{
	void MatrixAdd(float* floatMatrix, float* floatMatrix2, float* outputs);

	void MatrixGradient(float* floatMatrix, float* outputs);
	void MatrixGradientDevice(float* dev_c, float* dev_a);

	void MatrixDiff(float* floatMatrix, float* outputs, int axis);
	void MatrixDiffDevice(float* dev_c, float* dev_a, int axis);

	float MatrixAbsDiffSum(float* floatMatrix, float* floatMatrix2);
	void MatrixAbsDiffSumDevice(float* dev_c, float* dev_a, float* dev_b);

	//functions relevant to gradient step of Brighter-Fatter
	void MatrixGradientSplice_Device(float* grad, float* tmpArray);
	void MatrixGradientSplice_2D_RowDevice(float* dev_c, float* dev_a);//need to test
	void MatrixGradientSplice_2D_ColDevice(float* dev_c, float* dev_a);//need to test
	void MatrixGradientProductSum_Device(float* dev_c, float* gradTmp_dev_0, float* gradTmp_dev_1, float* gradOut_dev_0, float* gradOut_dev_1);

	//functions relevant to the diff step of the Brighter_Fatter
	void MatrixDiffRowSplice_Device(float* dev_c, float* diffOut20);
	void MatrixSecondDiff_RowDevice(float* dev_c, float* dev_a, float* temp);
	void MatrixSecondDiff_ColDevice(float* dev_c, float* dev_a, float* temp);
	void MatrixDiffProductSum_Device(float* dev_c, float* tmpArray, float* diffOut20, float* diffOut21);

	//fucntions relevant to corr step of Brighter-Fatter
	void MatrixCorrProductSum_Device(float* dev_c, float* first, float* second);
	
	//data copy functions:
	void MatrixCopy_Device(float* dev_c, float* dev_a);
	void MatrixCopyCorr_Device(float* tmpArray, float* image, float* corr);

	//final function, updating source image:
	void matrixIncrementSplice_Device(float* imgSrc, float* corr);

	//Just some crap below that doesn't really work... can look at later if have time...
	void FreeDevice(float* dev);
	void DeviceToHost(float* dev, float* outputs, cudaError_t cudaStatus);
	void DeviceAllocation(float* dev, cudaError_t cudaStatus);
	void HostToDevice(float* dev, float* input, cudaError_t cudaStatus);
	cudaError_t GetCudaStatus();

	//create wrappers for rest of teh critical functions.(diff, graidnet, abs sum (to be developed)
	//also figure out how to make this compile with the weird complex makefile boxfilternpp r e q u i r e s...
}