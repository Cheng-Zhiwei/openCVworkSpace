////////////////////// 01 read img///////////////////////////////

//#include <opencv2/core.hpp>;
//#include <opencv2/imgcodecs.hpp>
//#include <opencv2/highgui.hpp>
//#include <iostream>
//#include <string>
//
//using namespace cv;
//using namespace std;
//
//int main(int argc, char** argv) {//int argc, char** argv，可以去掉不使用
//
//	String imgName("E:/fig1.jpg");
//
//	if (argc > 1) {
//		imgName = argv[1];
//	}
//
//	Mat image  = imread(imgName, IMREAD_COLOR);
//
//	if (image.empty()) {
//		cout << "Could not open or find the image" << std::endl;
//		system("pause");
//		return -1;
//	}
//	namedWindow("display the window", WINDOW_AUTOSIZE);
//	imshow("display the window", image);
//	waitKey(0);
//	destroyAllWindows();
//		return 0;
//}



//////////////////////02-wirte image//////////////////////////////////////
#include <opencv2/opencv.hpp>
#include <string>

using namespace cv;

int main() {

	char namePath[20] = "E:\\fig1.jpg";

	Mat img;
	img = imread(namePath, IMREAD_COLOR);//create the mat and read image

	Mat gray_img;
	cvtColor(img, gray_img, COLOR_BGR2GRAY);//BGR to gray
	imwrite("E:\\fig2.jpg", gray_img);

	namedWindow("1", WINDOW_AUTOSIZE);//adjust size
	namedWindow("2", WINDOW_AUTOSIZE);

	imshow("1", img);
	imshow("2", gray_img);

	waitKey(0);
	destroyAllWindows();

	return 0;
}