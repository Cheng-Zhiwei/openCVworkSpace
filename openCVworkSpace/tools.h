#pragma once
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
using namespace cv;
using namespace std;

bool matIsEqual(const cv::Mat mat1, const cv::Mat mat2);
Mat& contrastBrightness(Mat& I, double& a, double& b);
Mat& contrastBrightnessByC(Mat& I, double& a, double& b);
Mat& contrastBrightnessByRA(Mat& I, double& c, double& d);
Mat& contrastBrightnessByOfficial(Mat& I, double& c, double& d);