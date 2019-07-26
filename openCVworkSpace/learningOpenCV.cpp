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
//#include <opencv2/opencv.hpp>
//#include <opencv2/imgproc.hpp>
//#include <opencv2/highgui.hpp>
//#include <opencv2/highgui/highgui_c.h>
//#include <opencv2/imgcodecs.hpp>
//#include <iostream> // 包含cout endl 打印
//#include <string>
//#include <opencv2/imgproc/imgproc_c.h> //包含IplImage


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



///////////////////////////03_扫描图像//////////////////////////////////////////////////


/*补充知识----C++的引用*/
#include <iostream>

//01基本运用
//using namespace std;
//
//int main() {
//
//	int a = 10;
//	int& b = a;
//
//	cout << a << endl;
//	cout << b << endl;
//	cout << &a << "," <<  &b << endl;
//
//	system("pause");
//	return 0;
//
//}

////引用作为函数的形参
//#include <iostream>
//using namespace std;
//void swap1(int a, int b);
//void swap2(int *p1, int *p2);
//void swap3(int &a, int &b);
//int main() {
//	int num1, num2;
//	cout << "Input two integers: ";
//	cin >> num1 >> num2;
//	swap1(num1, num2);
//	cout << num1 << " " << num2 << endl;
//
//	cout << "Input two integers: ";
//	cin >> num1 >> num2;
//	swap2(&num1, &num2);
//	cout << num1 << " " << num2 << endl;
//
//	cout << "Input two integers: ";
//	cin >> num1 >> num2;
//	swap3(num1, num2);
//	cout << num1 << " " << num2 << endl;
//
//	system("pause");
//	return 0;
//}
////直接传递参数内容
//void swap1(int a, int b) {
//	int temp = a;
//	a = b;
//	b = temp;
//}
////传递指针
//void swap2(int *p1, int *p2) {
//	int temp = *p1;
//	*p1 = *p2;
//	*p2 = temp;
//}
////按引用传参
//void swap3(int &a, int &b) {
//	int temp = a;
//	a = b;
//	b = temp;
//}


////引用作为函数返回值
/*运行程序后得到的结果为nu=20，num2=20，过程如下：
首先num1=10,然后把使用函数，int &n = num1,n和num1代表同一块内存的值，
当n=n+10后，n=20，即num1=20，然后num2 = num1 = 20*/
//#include <iostream>
//using namespace std;
//int& plus10(int &n) {
//	n = n + 10;
//	return n;
//}
//
//int main() {
//	int num1 = 10;
//	int num2 = plus10(num1);
//	cout << num1 << " " << num2 << endl;
//
//	system("pause");
//	return 0;
//}











#include <opencv2/core.hpp>  //核心模块，定义了基本的数据结构和算术函数
#include <opencv2/core/utility.hpp>  //包含getTickFrequency()等函数
#include "opencv2/imgcodecs.hpp"  //图片编解码、加载保存之类的函数
#include <opencv2/highgui.hpp>  //视频捕捉、图像和视频的编码解码、图形交互界面的接口
#include <iostream>  //C++中用于数据的流式输入与输出的头文件
#include <sstream>  //使用stringsteam类型需要用到的头文件

	using namespace std;
	using namespace cv;

	static void help() //静态函数，这个函数输出函数信息及命令行参数信息
	{
		cout
			<< "\n--------------------------------------------------------------------------" << endl
			<< "This program shows how to scan image objects in OpenCV (cv::Mat). As use case"
			<< " we take an input image and divide the native color palette (255) with the " << endl
			<< "input. Shows C operator[] method, iterators and at function for on-the-fly item address calculation." << endl
			<< "Usage:" << endl
			<< "./how_to_scan_images <imageNameToUse> <divideWith> [G]" << endl
			<< "if you add a G parameter the image is processed in gray scale" << endl
			<< "--------------------------------------------------------------------------" << endl
			<< endl;
	}



	/*下面是C操作符[]（指针）、迭代器、即时项目地址计算三种方法函数声明，
	其中的&读作引用，相当于给函数或变量名起了第二个名字，引用初始化某个变量后，
	可以使用该引用名称或原变量名称指向该变量，和指针有一定的区别，具体请参考C++引用*/


	//Mat& 返回Mat类型返回值的引用
	Mat& ScanImageAndReduceC(Mat& I, const uchar* table);//通过C指针方式扫描
	Mat& ScanImageAndReduceIterator(Mat& I, const uchar* table);//通过迭代器
	Mat& ScanImageAndReduceRandomAccess(Mat& I, const uchar * table);//通过at函数

	// argc是命令行总的参数个数，argv[]是argc个参数，其中第0个参数是程序的全名，后面跟的用户输入的参数
	int main(int argc, char* argv[])
	{
		help();
		if (argc < 3)//判断输入参数是否合法，如果个数小于3个，则输出参数不够
		{
			cout << "Not enough parameters" << endl;
			return -1;
		}

		Mat I, J; //I为输入矩阵；J为输出矩阵

		//int strcmp(const char *s1, const char *s2);
	   //返回值：若s1、s2字符串相等，则返回零；若s1大于s2，则返回大于零的数；否则，则返回小于零的数。
		if (argc == 4 && !strcmp(argv[3], "G"))
			I = imread(argv[1], IMREAD_GRAYSCALE);
		else
			I = imread(argv[1], IMREAD_COLOR);

		if (I.empty())
		{
			cout << "The image" << argv[1] << " could not be loaded." << endl;
			return -1;
		}

	
		int divideWith = 0;  // 将我们输入的字符串转换为数字 -C++风格
		stringstream s; //这里用到了stringstream
		s << argv[2]; //将第三个参数复制给字符串
		s >> divideWith;//将字符串转化为数字





		if (!s || !divideWith)
		{
			cout << "Invalid number entered for dividing. " << endl;
			return -1;
		}

		uchar table[256];//建立一张颜色空间缩减的表格，其实就是数组，方便后边查找赋值
		for (int i = 0; i < 256; ++i)
			table[i] = (uchar)(divideWith * (i / divideWith));
		//! [dividewith]



		const int times = 100;//定义常量，即值不会改变的量。常量的值为100，目的是计算执行100次的平均时间
		double t;//平均执行时间


		////////////////////C程序运行时间////////////////////
		t = (double)getTickCount();

		for (int i = 0; i < times; ++i)
		{
			cv::Mat clone_i = I.clone();
			J = ScanImageAndReduceC(clone_i, table);
		}

		//1000 * 总次数 / 一秒内重复的次数 = 时间(ms)
		t = 1000 * ((double)getTickCount() - t) / getTickFrequency();
		t /= times;
		cout << "Time of reducing with the C operator [] (averaged for "
			<< times << " runs): " << t << " milliseconds." << endl;


		////////////////////Iterator程序运行时间////////////////////
		t = (double)getTickCount();

		for (int i = 0; i < times; ++i)
		{
			cv::Mat clone_i = I.clone();
			J = ScanImageAndReduceIterator(clone_i, table);
		}


	
		//1000 * 总次数 / 一秒内重复的次数 = 时间(ms)
		t = 1000 * ((double)getTickCount() - t) / getTickFrequency();
		t /= times;

		cout << "Time of reducing with the iterator (averaged for "
			<< times << " runs): " << t << " milliseconds." << endl;



		////////////////////////////////////////////////////////////
		t = (double)getTickCount();

		for (int i = 0; i < times; ++i)
		{
			cv::Mat clone_i = I.clone();
			ScanImageAndReduceRandomAccess(clone_i, table);
		}

		t = 1000 * ((double)getTickCount() - t) / getTickFrequency();
		t /= times;

		cout << "Time of reducing with the on-the-fly address generation - at function (averaged for "
			<< times << " runs): " << t << " milliseconds." << endl;



		//////////////////////////LUT运行时间//////////////////////////////////////
		//! [table-init]
		Mat lookUpTable(1, 256, CV_8U);
		uchar* p = lookUpTable.ptr();
		for (int i = 0; i < 256; ++i)
			p[i] = table[i];
		//! [table-init]

		t = (double)getTickCount();

		for (int i = 0; i < times; ++i)
			//! [table-use]
			LUT(I, lookUpTable, J);
		//! [table-use]

		t = 1000 * ((double)getTickCount() - t) / getTickFrequency();
		t /= times;

		cout << "Time of reducing with the LUT function (averaged for "
			<< times << " runs): " << t << " milliseconds." << endl;
		return 0;
	}

	//! [scan-c]
	Mat& ScanImageAndReduceC(Mat& I, const uchar* const table)
	{
		// accept only char type matrices
		CV_Assert(I.depth() == CV_8U);

		int channels = I.channels();

		int nRows = I.rows;
		int nCols = I.cols * channels;

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
	//! [scan-c]

	//! [scan-iterator]
	Mat& ScanImageAndReduceIterator(Mat& I, const uchar* const table)
	{
		// accept only char type matrices
		CV_Assert(I.depth() == CV_8U);

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
	//! [scan-iterator]

	//! [scan-random]
	Mat& ScanImageAndReduceRandomAccess(Mat& I, const uchar* const table)
	{
		// accept only char type matrices
		CV_Assert(I.depth() == CV_8U);

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
	//! [scan-random]















