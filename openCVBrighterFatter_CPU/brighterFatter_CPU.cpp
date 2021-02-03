//Author: Adriel Kim
//1-31-2021
//CPU Implementation of Brighter-Fatter Correction using OpenCV

#include <opencv2/opencv.hpp>

#include <stdlib.h>
#include<iostream>
#include<fstream>
#include<sys/time.h>
#include <string.h>
#include <fstream>
#include <bits/stdc++.h>
#include <stdlib.h>
#include <set>


using namespace cv;
using namespace std;

long long start_timer();
long long stop_timer(long long start_time, const char* name);
void fillKernelArray(std::string kernelName, float* kernelArr, int kernelSize);

//img1 is true, while img2 is generated
double pixelErrorThresholdPercent(float* img1, float* img2, int imgSize, int imgDimX, std::set<int>& badRows, std::set<int>& badCols) {
    int problemCount = 0;
    for (int i = 0;i < imgSize;i++) {
        double pix1 = img1[i];
        double pix2 = img2[i];
        double pixDiff = abs(pix1 - pix2);
        double percentError = (pixDiff / pix1) * 100;
        if (percentError >= 2) {
            problemCount++;
            int row = i / (imgDimX);
            int col = i % imgDimX - 1;
            //std::cout << "(ROW, COL): " << "(" << row << "," << col << ")" << std::endl;
            badRows.insert(row);
            badCols.insert(col);
        }
    }
    return ((double)problemCount / (double)imgSize) * 100;
}

/*Gets mean squared error between two images*/
double meanSquaredImages(float* img1, float* img2, int imgSize) {
    double diff = 0;
    for (int i = 0;i < imgSize;i++) {
        double pix1 = img1[i];
        double pix2 = img2[i];
        double pixDiff = pix1 - pix2;
        diff += (pixDiff * pixDiff);
    }
    double mse = diff / ((double)imgSize);
    return mse;
}

/*Gets maximum error percent*/
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

double maxErrorAbsolute(float* img1, float* img_true, int imgSize) {
    double maxError = abs(img1[0] - img_true[0]);

    for (int i = 1;i < imgSize;i++) {
        double err = abs(img1[i] - img_true[i]);
        if (err > maxError) {
            maxError = err;
        }
    }

    return maxError;
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
        if (percent > max_percent&& abs(img_true[i]) > 0.01) {
            max_percent = percent;
            true_val = img_true[i];
            predicted = img1[i];
        }
    }
    printMaxAndMin(img_true, imgSize);

    return max_percent;//p_error;
}


long long avgPixelValue(float* img, int imgSize) {
    long long sum = 0;
    for (int i = 0;i < imgSize;i++) {
        sum += img[i];
    }
    long long avg = sum / imgSize;
    return avg;
}

void fillKernelArray(std::string kernelName, float*kernelArr, int kernelSize) {
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
            char* char_array = new char[n + 1];//to avoid n+1 not being a constant index, we dynamically allocate memory
            strcpy(char_array, word.c_str());
            char* pEnd;
            float wordDouble = strtod(char_array, &pEnd);
            kernelArr[i] = wordDouble;
            sum += wordDouble;
            /*if (i == 144)//testing out identity kernel.
                kernelArr[i] = 1;//wordDouble;
            else
                kernelArr[i] = 0;*/
            delete[] char_array;//hmmm ok..
            //std::cout << kernelArr[i] << std::endl;
        }

    }

    //for (int i = 0;i < kernelSize;i++) {
    //    kernelArr[i] = kernelArr[i] / sum;
    //}

    file.close();
}

//numpy.gradient CPU implementation - row axis
void RowGradient(float* gradient_arr, float* spliced_arr, int startY, int endY, int startX, int endX, int imgWidth) {

    float spliced_height = endY - startY;
    float spliced_width = endX - startX;
    float spliced_size = (spliced_height) * (spliced_width);//make sure these points are concrete

    //exclusive gradient of first row
    for (int i = 0;i < spliced_width;i++) {//traverses first row of spliced array
        int next_row = i + spliced_width;
        gradient_arr[i] = spliced_arr[next_row] - spliced_arr[i];
    }
    //exclusive gradient of second row
    for (int i = spliced_size - spliced_width;i < spliced_size;i++) {//traverses last row of spliced array
        int prev_row = i - spliced_width;
        gradient_arr[i] = spliced_arr[i] - spliced_arr[prev_row];
    }

    int row_offset = spliced_width;//row offset so as to not collide with first row gradient
    //gradient of rest of array, which is average every two rows
    for (int row = 0;row < spliced_height - 2;row++) {
        for (int col = 0; col < spliced_width;col++) {
            int linear_index = (row * spliced_width) + col; // current index of interest
            int next_index = ((row + 2) * spliced_width) + col;//index in next two rows from current

            gradient_arr[linear_index + row_offset] = (spliced_arr[next_index] - spliced_arr[linear_index]) / 2;
        }
    }

}

//numpy.gradient CPU implementation - column axis
void ColumnGradient(float* gradient_arr, float* spliced_arr, int startY, int endY, int startX, int endX, int imgWidth) {
    float spliced_height = endY - startY;
    float spliced_width = endX - startX;
    float spliced_size = (spliced_height) * (spliced_width);//make sure these points are concrete

    for (int row = 0; row < spliced_height;row++) {//gradient of first column of spliced array (column-major traversal)
        int linear_index = row * spliced_width;
        int adj_index = linear_index + 1;
        gradient_arr[linear_index] = spliced_arr[adj_index] - spliced_arr[linear_index];
    }

    for (int row = 0;row < spliced_height;row++) {//gradient of last column of spliced array
        int last_row_index = row * spliced_width + (spliced_width - 1);
        int prev_index = last_row_index - 1;
        gradient_arr[last_row_index] = spliced_arr[last_row_index] - spliced_arr[prev_index];
    }
    
    int col_offset = 1;

    for (int col = 0; col < spliced_width-2;col++) {
        for (int row = 0; row < spliced_height;row++) {
            int linear_index = (row * spliced_width) + col;
            int next_index = linear_index + 2;

            gradient_arr[linear_index + col_offset] = (spliced_arr[next_index] - spliced_arr[linear_index]) / 2;
        }
    }

}

//Array splicer. Behaves like Python array slicing for matrices
float* SpliceArray(float* arr, int startY, int endY, int startX, int endX, int imgWidth) {
    int spliced_height = endY - startY;
    int spliced_width = endX - startX;
    int spliced_size = (spliced_height) * (spliced_width);
    float* spliced_arr = new float[spliced_size];
    int spliced_index = 0;
    
    //populating spliced array, to be use in gradient
    
    for (int row = startY;row < endY;row++) {
        for (int col = startX; col < endX;col++) {
            int arr_index = row * imgWidth + col;//imgWidth, aka # of columns of original image;
            int spliced_arr_index = (row - startY) * (spliced_width)+(col - startX);
            spliced_arr[spliced_arr_index] = arr[arr_index];
 
        }
    }

    return spliced_arr;
}

/*
CPU Implementation of:
    gradTmp = numpy.gradient(tmpArray[startY:endY, startX:endX])
    gradOut = numpy.gradient(outArray[startY:endY, startX:endX])

*/
void GetGradient(float* grad_row, float* grad_col, float* arr, int startY, int endY, int startX, int endX, int imgWidth) {

    int spliced_size = (endX - startX) * (endY - startY);
    float* spliced_arr = SpliceArray(arr, startY, endY, startX, endX, imgWidth);

    RowGradient(grad_row, spliced_arr, startY, endY, startX, endX, imgWidth);
    ColumnGradient(grad_col, spliced_arr, startY, endY, startX, endX, imgWidth);

}


/*Gets second diff of matrix along row axis*/
void Get_SecondDiff_Row(float* diffOut,float* arr,int imgWidth, int imgHeight, int startY, int endY_val, int startX, int endX_val) {
    int diff_index = 0;
    int diff_height = imgHeight - 1;
    float* temp = new float[imgWidth * diff_height];
    for (int row = 0;row <diff_height;row++) {
        for (int col = 0; col < imgWidth;col++) {
            int linear_index = row * imgWidth + col;
            int row_adj = (row + 1) * imgWidth + col;
            temp[diff_index] = arr[row_adj] - arr[linear_index];
            diff_index++;
        }
    }


    diff_index = 0;
    int secondDiff_height = diff_height - 1;
    float* temp2 = new float[imgWidth * secondDiff_height];
    for (int row = 0;row < secondDiff_height;row++) {
        for (int col = 0;col < imgWidth;col++) {
            int linear_index = row * imgWidth + col;
            int row_adj = (row + 1) * imgWidth + col;
            temp2[diff_index] = temp[row_adj] - temp[linear_index];
            diff_index++;
        }
    }

    //have not check if this works properly..
    diff_index = 0;
    //splicing
    for (int row = startY;row < endY_val-2;row++) {//-2 is to account for temp being shrunken 2 rows
        for (int col = (startX + 1);col < (endX_val - 1);col++) {
            int linear_index = row * imgWidth + col;
            diffOut[diff_index] = temp2[linear_index];
            diff_index++;
        }
    }


    
}

/*Gets second diff of matrix along column axis*/
void Get_SecondDiff_Column(float* diffOut, float* arr, int imgWidth, int imgHeight, int startY, int endY_val, int startX, int endX_val) {
    int diff_width = imgWidth - 1;
    float* temp = new float[diff_width * imgHeight];

    for (int col = 0;col < diff_width;col++) {
        for (int row = 0;row < imgHeight;row++) {
            int linear_index = row * imgWidth + col;
            int col_adj = linear_index + 1;
            int diff_index = row * diff_width + col;
            temp[diff_index] = arr[col_adj] - arr[linear_index];
        }
    }


    int secondDiff_width = diff_width - 1;
    float* temp2 = new float[secondDiff_width * imgHeight];

    for (int col = 0;col < secondDiff_width;col++) {
        for (int row = 0;row < imgHeight;row++) {
            int linear_index = row * diff_width + col;
            int col_adj = linear_index + 1;
            int diff_index = row * secondDiff_width + col;
            temp2[diff_index] = temp[col_adj] - temp[linear_index];
        }
    }

    int diff_index = 0;
    for (int row = startY+1;row < endY_val-1;row++) {
        for (int col = startX;col < endX_val-2;col++) {//-2 is to adjust for temp2 being shrunken 2 cols
            int linear_index = row * secondDiff_width + col;
            diffOut[diff_index] = temp2[linear_index];
            diff_index++;
        }
    }


}

/* CPU implementation of:

    diffOut20 = numpy.diff(outArray, 2, 0)[startY:endY, startX + 1:endX - 1]
    diffOut21 = numpy.diff(outArray, 2, 1)[startY + 1:endY - 1, startX:endX]
*/
void Get_SecondDiff(float* diffOut, float* arr ,int startY, int endY, int startX, int endX, int imgWidth, int imgHeight, int axis) {
    
    if (axis == 0) {//row-wise 
        Get_SecondDiff_Row(diffOut, arr, imgWidth, imgHeight,startY, endY, startX,endX);
    }
    else if (axis == 1) {//column-wise
        Get_SecondDiff_Column(diffOut, arr, imgWidth, imgHeight,startY, endY, startX, endX);
    }
}

/*
CPU equivalent of:
    diff = numpy.sum(numpy.abs(prev_image - tmpArray))
*/
double MatrixAbsDiffSum(float* arr1, float* arr2, int arrSize) {
    double absDiff = 0;
    
    for (int i = 0;i < arrSize;i++) {
        absDiff += abs(arr1[i] - arr2[i]);
    }
    return absDiff;
}

int main(int argc, char** argv)
{
    // Declare variables
    long long loading_timer = start_timer();
    int imgDimX = 2048;
    int imgDimY = 4176;//2048;
    int imgSize = imgDimX * imgDimY;
    std::fstream file;
    std::string word, t, q, filename, compareFilename, corrFilename;
    float* originalImg = new float[imgSize];
    float* compareImg = new float[imgSize];//this is the image we will want to compare to verify our results. 
    float* dstImg = new float[imgSize];
    float* corrOrigin = new float[imgSize];


    int kernel_size;
    kernel_size = 17;// 3 + 2 * (ind % 5);
    int kSize = kernel_size * kernel_size;
    float* kdata = new float[kSize]; 
    fillKernelArray("bfKernel.txt", kdata, kSize);

    // filename of the file
    filename = "inputImgOG3.txt";
    compareFilename = "finalImgOG3.txt";
    corrFilename = "corr3.txt";

    // opening file
    file.open(filename.c_str());
    //double sum = 0;


    // extracting words from the file
    if (file.is_open()) {
        for (int i = 0; i < imgSize;i++) {
            file >> word;
            int n = word.length();
            char char_array[n + 1];
            strcpy(char_array, word.c_str());
            char* pEnd;
            float wordDouble = strtod(char_array, &pEnd);
            originalImg[i] = wordDouble;
        }

    }
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

    long long loadend_timer = stop_timer(loading_timer,"Time to load image data from text file: ");


    file.open(compareFilename.c_str());
    if (file.is_open()) {
        for (int i = 0;i < imgSize;i++) {
            file >> word;
            int n = word.length();
            char char_array[n + 1];
            strcpy(char_array, word.c_str());
            char* pEnd;
            float wordDouble = strtod(char_array, &pEnd);
            compareImg[i] = wordDouble;
        }
    }
    std::cout << "Input image to be compared loaded onto double array" << std::endl;
    file.close();



    long long timer = start_timer();


    long long memTimer = start_timer();

    Mat src(imgDimY, imgDimX, CV_32FC1, originalImg);//time this separately


    Mat dst;
    //Mat kernel;
    Point anchor;
    double delta;
    int ddepth;
    const char* window_name = "filter2D Demo";
    const char* imageName = "fitstest.pgm";
    // Loads an image
    //src = imread(imageName, IMREAD_GRAYSCALE);//cv::samples::findFile(imageName),0); // Load an image
    if (src.empty())
    {
        printf(" Error opening image\n");
        printf(" Program Arguments: [image_name -- default lena.jpg] \n");
        return EXIT_FAILURE;
    }

    
    // Initialize arguments for the filter
    anchor = Point(-1, -1);//center by default
    delta = 0;
    ddepth = -1;//-1 means that the output image will have same depth as input image


    // Update kernel size for a normalized box filter
    
    Mat kernel(kernel_size, kernel_size, CV_32F, kdata);//seems to be an issue with loading the kernel data.....(do i need to dereference?)

    long long memTimerEnd = stop_timer(memTimer, "Memory Time: ");


    long long justFilter = start_timer();



    int startX = 8;
    int endX = -9;//in Python, this is the 9th element from the right-hand side (startY:endY, where endY is non-inclusive.(so up to 10th element from the end)
    int startY = 8;
    int endY = -9;

    int endX_value = imgDimX + endX;//actual end point column of splice
    int endY_value = imgDimY + endY;//actual end point row of splice

    //double check this...
    //size of resulting gradient array
    int grad_R = (imgDimY - startY + endY);
    int grad_C = (imgDimX - startX + endX);
    int grad_N = grad_R * grad_C; //in python, index by row,co;
    int first_N = (grad_R - 2) * (grad_C - 2);//this is spliced from gradient array.


    int diff_R_0 = (imgDimY - startY + endY);
    int diff_C_0 = (imgDimX - (startX + 1) + (endX - 1));
    int diff_R_1 = (imgDimY - (startY + 1) + (endY - 1));
    int diff_C_1 = (imgDimX - startX + endX);

    //relevant diff array size vars
    int diffOut20_N = diff_R_0 * diff_C_0;
    int diffOut21_N = diff_R_1 * diff_C_1;
    int diffOut20_N_full = imgSize - imgDimX;//diffOut20_N will not be used (unless we can optimize the kernel)
    int diffOut21_N_full = imgSize - imgDimY;//same reason as above

    int second_N = diff_R_1 * diff_C_0;
    // = (R - (startY + 1) + (endY - 1)) * (C - (startX + 1) + (endX - 1))

    //relevant corr vars
    int corr_R = imgDimY - (startY + 1) + (endY - 1);
    int corr_C = imgDimX - (startX + 1) + (endX - 1);
    int corr_N = corr_R * corr_C;

    float threshold = 1000;
    int maxIter = 10;//10;
    float diff = 0;
    
    float* image = new float[imgSize];
    float* tmpArray = new float[imgSize];// modify src directly to match tmpArray per iteration
    float* outArray = new float[imgSize];
    float* prev_image = new float[imgSize];

    float* gradTmp_row = new float[grad_N];
    float* gradTmp_col = new float[grad_N];
    float* gradOut_row = new float[grad_N];
    float* gradOut_col = new float[grad_N];
    float* first = new float[first_N];

    float* diffOut20 = new float[diffOut20_N];
    float* diffOut21 = new float[diffOut21_N];
    float* second = new float[second_N];

    float* corr = new float[imgSize];

    //for storing original image
    for (int i = 0;i < src.rows;i++) {
        for (int j = 0;j < src.cols;j++) {
            int linear_index = i * imgDimX + j;
            if (isnan(src.at<float>(i, j)) == false)
                image[linear_index] = src.at<float>(i, j);
            else {
                image[linear_index] = 0;
                //cout << "GOT A NAN" << endl;
            }
            //cout << "image value:" << i << " = " << image[linear_index];
        }
    }





    for (int i = 0;i < maxIter;i++) {
        filter2D(src, dst, ddepth, kernel, anchor, delta, BORDER_DEFAULT);


        for (int row = 0;row < src.rows;row++) {
            for (int col = 0;col < src.cols;col++) {
                int linear_index = row * imgDimX + col;
                if (isnan(src.at<float>(row, col)) == false)
                    tmpArray[linear_index] = src.at<float>(row, col);
                else{
                    tmpArray[linear_index] = 0;
                    //cout << "GOT NAN" << endl;
                }

            }
        }

        for (int row = 0;row < dst.rows;row++) {
            for (int col = 0;col < dst.cols;col++) {
                int linear_index = row * imgDimX + col;
                if(isnan(dst.at<float>(row,col)) == false)
                    outArray[linear_index] = dst.at<float>(row, col);
                else {
                    outArray[linear_index] = 0;
                    //cout << "GOT NAN" << endl;
                }
            }
        }

        /*
        Equivalent to:
                gradTmp = numpy.gradient(tmpArray[startY:endY, startX:endX])
                gradOut = numpy.gradient(outArray[startY:endY, startX:endX])
        */
        GetGradient(gradTmp_row, gradTmp_col, tmpArray, startY, endY_value, startX, endX_value, imgDimX);
        GetGradient(gradOut_row, gradOut_col, outArray, startY, endY_value, startX, endX_value, imgDimX);
        
        /*
         Equivalent to:
            first = (gradTmp[0]*gradOut[0] + gradTmp[1]*gradOut[1])[1:-1, 1:-1]
         */
        int first_index = 0;
        for (int row = 1;row < grad_R - 1;row++) {
            for (int col = 1;col < grad_C - 1;col++) {
                int linear_index = row * grad_C + col;
                first[first_index] = (gradTmp_row[linear_index] * gradOut_row[linear_index]) + (gradTmp_col[linear_index] * gradOut_col[linear_index]);
                first_index++;
            }
        }

        /*
          Equivalent to:
              diffOut20 = numpy.diff(outArray, 2, 0)[startY:endY, startX + 1:endX - 1]
              diffOut21 = numpy.diff(outArray, 2, 1)[startY + 1:endY - 1, startX:endX]
          */
        Get_SecondDiff(diffOut20, outArray, startY, endY_value,startX, endX_value, imgDimX, imgDimY, 0);
        Get_SecondDiff(diffOut21, outArray, startY, endY_value,startX, endX_value, imgDimX, imgDimY, 1);
        

        /*
            Equivalent to:        
                second = tmpArray[startY + 1:endY - 1, startX + 1:endX - 1]*(diffOut20 + diffOut21)
        */
        int second_index = 0;
        for (int row = (startY + 1);row < (endY_value - 1);row++) {
            for (int col = (startX + 1);col < (endX_value - 1);col++) {
                int tmp_index = row * imgDimX + col;
                second[second_index] = tmpArray[tmp_index] * (diffOut20[second_index] + diffOut21[second_index]);
                second_index++;
            }
        }

        
        /*
            Equivalent to:
                corr[startY + 1:endY - 1, startX + 1:endX - 1] = 0.5*(first + second)
        */
        int diffs_index = 0;
        for (int row = startY + 1; row < endY_value - 1;row++) {
            for (int col = startX + 1;col < endX_value - 1;col++) {
                int corr_index = row * imgDimX + col;//corr is same size as imgSize
                corr[corr_index] = 0.5 * (first[diffs_index] + second[diffs_index]);
                diffs_index++;
            }
        }

        /*
        Equivalent to:
            tmpArray[:, :] = image.getArray()[:, :]
            tmpArray[nanIndex] = 0.
            tmpArray[startY:endY, startX:endX] += corr[startY:endY, startX:endX]
         */
        for (int k = 0;k < imgSize;k++) {
            if (isnan(tmpArray[k]) == false)
                tmpArray[k] = image[k];
            else
                tmpArray[k] = 0;
        }
        for (int row = startY; row < endY_value;row++) {
            for (int col = startX;col < endX_value;col++) {
                int linear_index = row * imgDimX + col;
                tmpArray[linear_index] += corr[linear_index];
            }
        }

        /*
        Equivalent to:

             if iteration > 0:
                diff = numpy.sum(numpy.abs(prev_image - tmpArray))

                if diff < threshold:
                    break
                prev_image[:, :] = tmpArray[:, :]
        */

        if (i > 0) {


            double diff = MatrixAbsDiffSum(prev_image, tmpArray, imgSize);

            cout << "iteration = " << i << endl;
            cout << "diff = " << diff << endl;

            if (diff < threshold) {
                break;
            }
            
            for (int k = 0;k < imgSize;k++) {
                prev_image[k] = tmpArray[k];
            }

        }

        //must copy over tmpArray to original src image this way, since it uses an odd object.
        for (int row = 0;row < src.rows;row++) {
            for (int col = 0;col < src.cols;col++) {
                int linear_index = row * imgDimX + col;
                src.at<float>(row, col) = tmpArray[linear_index];
            }
        }


    }

  
    int testing = 0;


    /*
    Equivalent to:
          image.getArray()[startY + 1:endY - 1, startX + 1:endX - 1] += corr[startY + 1:endY - 1, startX + 1:endX - 1]
    */
    cout << "Final Step" << endl;
    for (int row = startY+1;row < endY_value - 1;row++) {
        for (int col = startX + 1;col < endX_value - 1;col++) {
            int linear_index = row * imgDimX + col;
            image[linear_index] = image[linear_index] + corr[linear_index];
        
        }
    }

    long long justFilterEnd = stop_timer(justFilter, "Just filter time: ");

    //FINAL RESULT IS "image"***
    
 /*   for (int row = 50;row < 52;row++) {
        for (int col = 50;col < 55;col++) {
            int index = row * imgDimX + col;
            cout << "CORR: " << " row = " << row << ", col = " << col << "value = " << corr[index] << endl;
            cout << "CORR Origin: " << " row = " << row << ", col = " << col << "value = " << corrOrigin[index] << endl;
            cout << "image" << " row = " << row << ", col = " << col << "value = " << image[index] << endl;
            cout << endl;

        }
    }
    */

    long long timerEnd = stop_timer(timer, "Total Filter time:");


    double mse_image = meanSquaredImages(image, compareImg, imgSize);
    double mse_corr = meanSquaredImages(corr, corrOrigin, imgSize);
    double maxError_img = maxError(image, compareImg, imgSize);
    double maxError_corr = maxErrorAbsolute(corr, corrOrigin, imgSize);//maxBrightPixelError(corr_host, corrOrigin, imgSize);
    double maxError_result = maxErrorAbsolute(image, compareImg, imgSize);

    std::cout << "Mean squared error of final result: " << mse_image << std::endl;
    std::cout << "Mean squared error of correction matrix: " << mse_corr << std::endl;
    std::cout << "Max error percentage of final result: " << maxError_img << std::endl;
    std::cout << "Max error absolute of final result:" << maxError_result << std::endl;
    std::cout << "Max error absolute of correction matrix: " << maxError_corr << std::endl;

    return EXIT_SUCCESS;
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
