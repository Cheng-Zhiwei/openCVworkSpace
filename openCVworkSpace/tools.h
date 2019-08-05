#pragma once
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

using namespace cv;
using namespace std;

/*��Ӧ����ͼ�������ֵ�Ƿ����*/
bool matIsEqual(const cv::Mat mat1, const cv::Mat mat2);

/*�ԱȶȺ����ȵ���ǿ*/
Mat& contrastBrightness(Mat& I, double& a, double& b);
Mat& contrastBrightnessByC(Mat& I, double& a, double& b);
Mat& contrastBrightnessByRA(Mat& I, double& c, double& d);
Mat& contrastBrightnessByOfficial(Mat& I, double& c, double& d);

/*٤��У��*/
Mat gammaResived(Mat& img, double& gamma);

/*��ֵ*/
uchar& getMedian(Mat& zone);

/*��ֵ�˲���3*3��*/
void MedianBlur(const Mat& src, Mat& img);

/*����cָ��*/
Mat& ScanImageAndReduceC(Mat& I, const uchar* const table);

/*���õ�����*/
Mat& ScanImageAndReduceIterator(Mat& I, const uchar* const table);

/*����randomAccess*/
Mat& ScanImageAndReduceRandomAccess(Mat& I, const uchar* const table);

/*�����븯ʴ*/
void ErodingAndDilating(const Mat& src, Mat& img, int& para);

/*��ֵ����*/
Mat& thresholdOperation(Mat& I, const int para, const uchar thres);

/*sobel����*/
void  sobelEdgeDetection(Mat& img_src, Mat& img_edge);

/*Gauss��*/
void getGaussKernel(int& size, double& sigma, Mat& Kernel);