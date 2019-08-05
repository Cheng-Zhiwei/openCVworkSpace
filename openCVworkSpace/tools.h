#pragma once
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

using namespace cv;
using namespace std;

/*对应两个图像的像素值是否相等*/
bool matIsEqual(const cv::Mat mat1, const cv::Mat mat2);

/*对比度和亮度的增强*/
Mat& contrastBrightness(Mat& I, double& a, double& b);
Mat& contrastBrightnessByC(Mat& I, double& a, double& b);
Mat& contrastBrightnessByRA(Mat& I, double& c, double& d);
Mat& contrastBrightnessByOfficial(Mat& I, double& c, double& d);

/*伽马校正*/
Mat gammaResived(Mat& img, double& gamma);

/*中值*/
uchar& getMedian(Mat& zone);

/*中值滤波（3*3）*/
void MedianBlur(const Mat& src, Mat& img);

/*利用c指针*/
Mat& ScanImageAndReduceC(Mat& I, const uchar* const table);

/*利用迭代器*/
Mat& ScanImageAndReduceIterator(Mat& I, const uchar* const table);

/*利用randomAccess*/
Mat& ScanImageAndReduceRandomAccess(Mat& I, const uchar* const table);

/*膨胀与腐蚀*/
void ErodingAndDilating(const Mat& src, Mat& img, int& para);

/*阈值操作*/
Mat& thresholdOperation(Mat& I, const int para, const uchar thres);

/*sobel算子*/
void  sobelEdgeDetection(Mat& img_src, Mat& img_edge);

/*Gauss核*/
void getGaussKernel(int& size, double& sigma, Mat& Kernel);