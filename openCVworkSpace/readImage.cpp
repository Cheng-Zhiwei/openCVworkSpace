/* 01 read img*/

#include <opencv2/core.hpp>;
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <string>

using namespace cv;
using namespace std;

int main(int argc, char** argv) {

	String imgName("E:/fig1.jpg");

	if (argc > 1) {
		imgName = argv[1];
	}

	Mat image  = imread(imgName, IMREAD_COLOR);

	if (image.empty()) {
		cout << "can't open or find the image" << std::endl;
		return -1;
	}
	namedWindow("display the window", WINDOW_AUTOSIZE);
	imshow("display the window", image);
	waitKey(0);
	destroyAllWindows();
		return 0;
}




