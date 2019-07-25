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
//#include <opencv2/opencv.hpp>
//#include <string>

//using namespace cv;
//
//int main() {
//
//	char namePath[20] = "E:\\fig1.jpg";
//
//	Mat img;
//	img = imread(namePath, IMREAD_COLOR);//create the mat and read image
//
//	Mat gray_img;
//	cvtColor(img, gray_img, COLOR_BGR2GRAY);//BGR to gray
//	imwrite("E:\\fig2.jpg", gray_img);
//
//	namedWindow("1", WINDOW_AUTOSIZE);//adjust size
//	namedWindow("2", WINDOW_AUTOSIZE);
//
//	imshow("1", img);
//	imshow("2", gray_img);
//
//	waitKey(0);
//	destroyAllWindows();
//
//	return 0;
//}





/////////////////////创建Mat对象/////////////////////////////
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgcodecs.hpp>
#include <iostream> // 包含cout endl 打印
#include <string>
#include <opencv2/imgproc/imgproc_c.h> //包含IplImage

using namespace cv;

//
//int main() {
//
//	char namePath[20] = "E:\\fig1.jpg";
//
//	Mat img;
//	img = imread(namePath, IMREAD_COLOR);//create the mat and read image
//
//	Mat gray_img;
//	cvtColor(img, gray_img, COLOR_BGR2GRAY);//BGR to gray
//
//
//
//	namedWindow("1", WINDOW_AUTOSIZE);//adjust size
//
//	imshow("1", img);
//
//
//	waitKey(0);
//	destroyAllWindows();
//
//	return 0;
//}



/*方法一，通过Mat函数构造*/
//int main() {
//
//	Mat M(2, 2, CV_8UC3, Scalar(0, 0, 255));
//	std::cout << "M = " << std::endl << " "<< M << std::endl;//C++打印输出
//
//	system("pause");
//	return 0;
//}


/*方法二，使用C数组并通过构造函数初始化*/
//int main() {
//
//	int sz[3] = { 2,2,2 };//构造C数组
//
//	Mat M(2, sz, CV_8UC3, Scalar::all(0));//Scalar所有元素设置为0
//
//	system("pause");
//	return 0;
//}


/*方法三，为已存在的ipiimage指针创建信息头*/
//int main() {
//
//	IplImage* img = cvLoadImage("E:\\fig1.jpg",1)；//4.0版本中无此函数，IpiImage为指针
//
//	Mat mtx（iimg）;//创建信息头
//
//	system("pause");
//	return 0;
//}




/*方法四，利用create（）函数
利用Mat类的create（）成员函数进行Mat类的初始化操作
此方法不能为矩阵设初值，只是在改变尺寸时为矩阵开辟空间
*/

//int main() {
//
//	Mat img = Mat();
//	img.create(4, 4, CV_8UC3);
//
//	std::cout << img;
//
//	system("pause");
//	return 0;
//}



/*方法五，采用matlab式初始化方式
方法五采用Matlab 形式的初始化方式： zeros( ) , ones(), eyes（ ） 。使用以下方
λ指定尺寸和数据类型：*/

//int main() {
//
//	Mat E = Mat::eye(4,4,CV_64F);
//	std::cout << E << std::endl;
//
//	Mat O = Mat::ones(4, 4, CV_64F);
//	std::cout << O << std::endl;
//
//	Mat Z = Mat::zeros(4, 4, CV_64F);
//	std::cout << Z << std::endl;
//
//	system("pause");
//	return 0;
//}


/*方法六采用matlab式初始化方式*/
//int main() {
//
//	Mat C = (Mat_<double>(3, 3) << 0, 1, 2, 3, 4, 5, 6, 7, 8);
//	std::cout << C << std::endl;
//
//	system("pause");
//	return 0;
//}



/*方法七：为已存在的对象创建信息头*/

//int main() {
//
//	Mat C = (Mat_<double>(3, 3) << 0, 1, 2, 3, 4, 5, 6, 7, 8);
//	std::cout << C << std::endl;
//
//	Mat rowClone = C.row(1).clone();
//	std::cout << rowClone;
//
//	system("pause");
//	return 0;
//}


/////////////////////////02 openCV格式化输出方法/////////////////////////

//int main() {
//
//	/*首先利用randu函数生成一个随机数的矩阵*/
//	Mat r = Mat(10, 3, CV_8UC3);
//	randu(r, Scalar::all(0), Scalar::all(255));
//
//	//opencv默认风格
//	std::cout << "r(默认风格)=" << r << std::endl;
//
//	//python style
//	std::cout << "r(python)=" << format(r, Formatter::FMT_PYTHON) << std::endl;
//
//	//numpy style
//	std::cout << "r(numpy)=" << format(r, Formatter::FMT_NUMPY) << std::endl;
//
//	//CSV style
//	std::cout << "r(CSV)" << format(r, Formatter::FMT_CSV) << std::endl;
//
//	//C style
//	std::cout << "r(CSV)" << format(r, Formatter::FMT_C) << std::endl;
//
//	
//	//二维点的定义和输出
//	Point2f p(6, 2);
//	std::cout << "2D_p=" << p << std::endl;
//
//	//三维点的定义和输出
//	Point3f p1(1, 2, 3);
//	std::cout << "3D_p1=" << p << std::endl;
//
//	//基于Mat的std::vector
//	std::vector<float> v;
//	v.push_back(3);
//	v.push_back(5);
//	v.push_back(7);
//	std::cout << "3Dp1=" << Mat(v) << std::endl;//注意：这里V表示的为Mat（v）；
//
//	//
//	std::vector<Point2f> points(20);
//
//	for (size_t i = 0; i < points.size(); ++i) {
//		points[i] = Point2f((float)(i * 5), (float)(i % 7));
//	}
//	
//	std::cout << "A vector of 2D Points = " << points << std::endl << std::endl;
//
//	system("pause");
//	return 0;
//}



#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <sstream>

using namespace std;
using namespace cv;

void help()
{
	cout
		<< "\n--------------------------------------------------------------------------" << endl
		<< "This program shows how to scan image objects in OpenCV (cv::Mat). As use case"
		<< " we take an input image and divide the native color palette (255) with the " << endl
		<< "input. Shows C operator[] method, iterators and at function for on-the-fly item address calculation." << endl
		<< "Usage:" << endl
		<< "./howToScanImages imageNameToUse divideWith [G]" << endl
		<< "if you add a G parameter the image is processed in gray scale" << endl
		<< "--------------------------------------------------------------------------" << endl
		<< endl;
}

Mat& ScanImageAndReduceC(Mat& I, const uchar* table);
Mat& ScanImageAndReduceIterator(Mat& I, const uchar* table);
Mat& ScanImageAndReduceRandomAccess(Mat& I, const uchar * table);

int main(int argc, char* argv[])
{
	help();
	if (argc < 3)
	{
		cout << "Not enough parameters" << endl;
		return -1;
	}

	Mat I, J;
	if (argc == 4 && !strcmp(argv[3], "G"))
		I = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
	else
		I = imread(argv[1], CV_LOAD_IMAGE_COLOR);

	if (!I.data)
	{
		cout << "The image" << argv[1] << " could not be loaded." << endl;
		return -1;
	}

	int divideWith; // convert our input string to number - C++ style
	stringstream s;
	s << argv[2];
	s >> divideWith;
	if (!s)
	{
		cout << "Invalid number entered for dividing. " << endl;
		return -1;
	}

	uchar table[256];
	for (int i = 0; i < 256; ++i)
		table[i] = divideWith * (i / divideWith);

	const int times = 100;
	double t;

	t = (double)getTickCount();

	for (int i = 0; i < times; ++i)
		J = ScanImageAndReduceC(I.clone(), table);

	t = 1000 * ((double)getTickCount() - t) / getTickFrequency();
	t /= times;

	cout << "Time of reducing with the C operator [] (averaged for "
		<< times << " runs): " << t << " milliseconds." << endl;

	t = (double)getTickCount();

	for (int i = 0; i < times; ++i)
		J = ScanImageAndReduceIterator(I.clone(), table);

	t = 1000 * ((double)getTickCount() - t) / getTickFrequency();
	t /= times;

	cout << "Time of reducing with the iterator (averaged for "
		<< times << " runs): " << t << " milliseconds." << endl;

	t = (double)getTickCount();

	for (int i = 0; i < times; ++i)
		ScanImageAndReduceRandomAccess(I.clone(), table);

	t = 1000 * ((double)getTickCount() - t) / getTickFrequency();
	t /= times;

	cout << "Time of reducing with the on-the-fly address generation - at function (averaged for "
		<< times << " runs): " << t << " milliseconds." << endl;

	Mat lookUpTable(1, 256, CV_8U);
	uchar* p = lookUpTable.data;
	for (int i = 0; i < 256; ++i)
		p[i] = table[i];

	t = (double)getTickCount();

	for (int i = 0; i < times; ++i)
		LUT(I, lookUpTable, J);

	t = 1000 * ((double)getTickCount() - t) / getTickFrequency();
	t /= times;

	cout << "Time of reducing with the LUT function (averaged for "
		<< times << " runs): " << t << " milliseconds." << endl;
	return 0;
}

Mat& ScanImageAndReduceC(Mat& I, const uchar* const table)
{
	// accept only char type matrices
	CV_Assert(I.depth() != sizeof(uchar));

	int channels = I.channels();

	int nRows = I.rows * channels;
	int nCols = I.cols;

	if (I.isContinuous())
	{
		nCols *= nRows;
		nRows = 1;
	}

	int i, j;
	uchar* p;
	for (i = 0; i < nRows; ++i)
	{
		p = I.ptr<uchar>(i);
		for (j = 0; j < nCols; ++j)
		{
			p[j] = table[p[j]];
		}
	}
	return I;
}

Mat& ScanImageAndReduceIterator(Mat& I, const uchar* const table)
{
	// accept only char type matrices
	CV_Assert(I.depth() != sizeof(uchar));

	const int channels = I.channels();
	switch (channels)
	{
	case 1:
	{
		MatIterator_<uchar> it, end;
		for (it = I.begin<uchar>(), end = I.end<uchar>(); it != end; ++it)
			*it = table[*it];
		break;
	}
	case 3:
	{
		MatIterator_<Vec3b> it, end;
		for (it = I.begin<Vec3b>(), end = I.end<Vec3b>(); it != end; ++it)
		{
			(*it)[0] = table[(*it)[0]];
			(*it)[1] = table[(*it)[1]];
			(*it)[2] = table[(*it)[2]];
		}
	}
	}

	return I;
}

Mat& ScanImageAndReduceRandomAccess(Mat& I, const uchar* const table)
{
	// accept only char type matrices
	CV_Assert(I.depth() != sizeof(uchar));

	const int channels = I.channels();
	switch (channels)
	{
	case 1:
	{
		for (int i = 0; i < I.rows; ++i)
			for (int j = 0; j < I.cols; ++j)
				I.at<uchar>(i, j) = table[I.at<uchar>(i, j)];
		break;
	}
	case 3:
	{
		Mat_<Vec3b> _I = I;

		for (int i = 0; i < I.rows; ++i)
			for (int j = 0; j < I.cols; ++j)
			{
				_I(i, j)[0] = table[_I(i, j)[0]];
				_I(i, j)[1] = table[_I(i, j)[1]];
				_I(i, j)[2] = table[_I(i, j)[2]];
			}
		I = _I;
		break;
	}
	}

	return I;
}














