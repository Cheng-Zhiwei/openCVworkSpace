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







//#include <opencv2/core.hpp> 
//#include <opencv2/core/utility.hpp>  
//#include "opencv2/imgcodecs.hpp" 
//#include <opencv2/highgui.hpp>  
//#include <iostream>  
//#include <sstream>  
//
//using namespace std;
//using namespace cv;

//Mat& scanImageByC(Mat& I, const uchar* const table) 
//{
//
//	int channels = I.channels();
//	int rows = I.rows;
//	int cols = I.cols;
//	int ncols = rows * cols*channels;
//
//	int i;
//	uchar* p;
//
//	p = I.ptr<uchar>(0);//获得数组的首地址
//
//	for (i = 0; i < ncols; i++) 
//	{
//
//		p[i] = table[p[i]];
//	}
//
//	return I;
//}
//
//Mat& scanImageByIteration(Mat& I, const uchar* const table) 
//{
//
//	CV_Assert(I.depth() == CV_8U);
//	
//	const int channels = I.channels();
//
//	switch (channels)
//	{
//	case 1: 
//	{
//		MatIterator_<uchar> it, end;
//		for (it = I.begin<uchar>(),end = I.end<uchar>(); it != end; ++it) 
//			*it = table[*it];
//		break;
//	}
//	case 3:
//		MatIterator_<Vec3b> it, end;
//		for (it = I.begin<Vec3b>(), end = I.end<Vec3b>(); it != end; ++it) 
//		{
//			(*it)[0] = table[(*it)[0]];//(*it)[0]为一个具体的数值
//			(*it)[1] = table[(*it)[1]];
//			(*it)[2] = table[(*it)[2]];
//		}
//		break;
//	}
//
//	return I;
//}
//
//
//
//Mat& scanImageByRandom(Mat& I, const uchar* const table) 
//{
//
//	const int channels = I.channels();
//
//	switch (channels)
//	{
//	case 1:
//	{for (int i = 0; i < I.rows; ++i)
//	{
//		for (int j = 0; i < I.cols; ++j) 
//		{
//			I.at<uchar>(i, j) = table[I.at<uchar>(i, j)];
//		}
//		break;
//	}
//	}
//
//	case 3:
//	{
//		Mat_<Vec3b> _I = I;// 定义_I 并赋值为I
//
//		for (int i = 0; i < I.rows; ++i) 
//		{
//			for (int j = 0; j < I.cols; ++j) 
//			{
//				_I(i, j)[0] = table[_I(i, j)[0]];
//				_I(i, j)[1] = table[_I(i, j)[1]];
//				_I(i, j)[2] = table[_I(i, j)[2]];
//			}
//
//			I = _I;
//			break;
//		}
//	}
//	}
//
//	return I;
//}



	//int main() 
	//{

	//	Mat I, J, J1, J2, J3;
	//	I = imread("E:\\fig1.jpg", 1);

	//	/*01_建立查找表*/
	//	int para = 10;
	//	uchar table[256];
	//	int len = 256;

	//	for (int i = 0; i < len; i++) 
	//	{
	//		table[i] = (i / para)*para;
	//		/*printf("%d", table[i]);*/
	//	}

	//	Mat I_clone = I.clone();

	//	J = scanImageByC(I_clone, table);
	//	J1 = scanImageByIteration(I_clone, table);
	//	J2 = scanImageByRandom(I_clone, table);


	//	Mat lookUpTable(1, 256, CV_8U);
	//	uchar* p = lookUpTable.data;
	//	for (int i = 0; i < 256; ++i)
	//		p[i] = table[i];
	//
	//	LUT(I, lookUpTable, J3);



	//	Mat_<Vec3b> J_img = J;
	//	Mat_<Vec3b> J1_img = J1;
	//	Mat_<Vec3b> J2_img = J2;
	//	Mat_<Vec3b> J3_img = J3;

	//	printf("%d\n", J_img(0, 0)[0]);
	//	printf("%d\n", J1_img(0, 0)[0]);
	//	printf("%d\n", J2_img(0, 0)[0]);
	//	printf("%d\n", J3_img(0, 0)[0]);

	//	//imshow("C", J);
	//	//waitKey(0);z

	//	system("pause");
	//	return 0;
	//}





///////////////////////////////////////	图像矩阵掩码操作//////////////////////////////////
//#include <opencv2/imgcodecs.hpp>
//#include <opencv2/highgui.hpp>
//#include <opencv2/imgproc.hpp>
//#include <iostream>
//
//using namespace std;
//using namespace cv;
//
////static void help(char* progName)
////{
////	cout << endl
////		<< "This program shows how to filter images with mask: the write it yourself and the"
////		<< "filter2d way. " << endl
////		<< "Usage:" << endl
////		<< progName << " [image_path -- default ../data/lena.jpg] [G -- grayscale] " << endl << endl;
////}
//
//
//void Sharpen(const Mat& myImage, Mat& Result);
//
//int main()
//{
//
//	Mat src, dst0, dst1;
//
//	src = imread("E:\\openCV_Pictures\\fig5_classical.jpg", IMREAD_GRAYSCALE);
//
//	namedWindow("Input", WINDOW_AUTOSIZE);
//	namedWindow("Output", WINDOW_AUTOSIZE);
//
//	imshow("Input", src);
//
//	double t = (double)getTickCount();
//
//	Sharpen(src, dst0);//函数
//
//	t = ((double)getTickCount() - t) / getTickFrequency();//计时
//	cout << "Hand written function time passed in seconds: " << t << endl;
//
//
//	imshow("Output", dst0);
//	waitKey(0);
//
//
//
//	//构造核
//	Mat kernel = (Mat_<char>(3, 3) << 0, -1, 0,
//		-1, 5, -1,
//		0, -1, 0);
//
//	t = (double)getTickCount();
//	filter2D(src, dst1, src.depth(), kernel);//使用filter2D函数
//	t = ((double)getTickCount() - t) / getTickFrequency();
//	cout << "Built-in filter2D time passed in seconds:     " << t << endl;
//
//	imshow("Output", dst1);
//	waitKey(0);
//	return 0;
//}
//
//
//
////基本方法
//void Sharpen(const Mat& myImage, Mat& Result)
//{
//
//	CV_Assert(myImage.depth() == CV_8U);  // 只接受8位图（uchar）
//
//
//	const int nChannels = myImage.channels();
//	Result.create(myImage.size(), myImage.type());//创建了一个与输入有着相同大小和类型的输出图像
//
//
//	for (int j = 1; j < myImage.rows - 1; ++j)
//	{
//		const uchar* previous = myImage.ptr<uchar>(j - 1);//前一行地址
//		const uchar* current = myImage.ptr<uchar>(j);//当前行地址
//		const uchar* next = myImage.ptr<uchar>(j + 1);//后一行地址
//
//		uchar* output = Result.ptr<uchar>(j);//获取j行的首地址
//
//		for (int i = nChannels; i < nChannels*(myImage.cols - 1); ++i)
//		{
//
//			//saturte_cast，防止数据溢出，例如if a<0, a=0,if a>255, a=255;
//			*output++ = saturate_cast<uchar>(5 * current[i] - 1 * current[i - nChannels] - 1 * current[i + nChannels]
//				- 1 * previous[i] + 0 * previous[i - nChannels] + 0 * previous[i + nChannels]
//				- 1 * next[i] + next[i - nChannels] + next[i + nChannels]);
//
//
//
//		}
//	}
//
//
//	//! [borders]
//	/*在图像的边界上，上面给出的公式会访问不存在的像素位置（比如(0, -1)）。
//	因此我们的公式对边界点来说是未定义的。一种简单的解决方法，是不对这些边界点使用掩码，而直接把它们设为0：*/
//	//外边界的元素都置为0
//	Result.row(0).setTo(Scalar(0));//上边界
//	Result.row(Result.rows - 1).setTo(Scalar(0));//下边界
//	Result.col(0).setTo(Scalar(0));//左边界
//	Result.col(Result.cols - 1).setTo(Scalar(0));//右边界
//	
//}


#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

using namespace std;
using namespace cv;

void Sharpen(const Mat& src, Mat& img)
{
	img.create(src.size(), src.type());
	int const nChanels = src.channels();

	for (int j = 1; j < src.rows-1; ++j)
	{
		const uchar* previous = src.ptr<uchar>(j - 1);
		const uchar* current = src.ptr<uchar>(j);
		const uchar* next = src.ptr<uchar>(j + 1);

		uchar* output = img.ptr<uchar>(j);//获取output的首地址

		for (int i = 1; i < nChanels*src.cols; ++i)
		{
			//先赋值后计算
			*output++ = 5 * current[i] - (previous[i] + next[i] + current[i - 1] + current[i + 1]);

		}
	}



}