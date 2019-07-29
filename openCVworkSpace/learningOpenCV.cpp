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
//	src = imread("E:\\openCV_Pictures\\fig5_classical.jpg", 1);
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
//	imshow("Output1", dst1);
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



////////////////////////////////////////////////////练习版

//#include <opencv2/imgcodecs.hpp>
//#include <opencv2/highgui.hpp>
//#include <opencv2/imgproc.hpp>
//#include <iostream>
//
//using namespace std;
//using namespace cv;
//
//void Sharpen(const Mat& src, Mat& img)
//{
//	img.create(src.size(), src.type());
//	int const nChanels = src.channels();
//
//	for (int j = 1; j < src.rows - 1; ++j)
//	{
//		const uchar* previous = src.ptr<uchar>(j - 1);
//		const uchar* current = src.ptr<uchar>(j);
//		const uchar* next = src.ptr<uchar>(j + 1);
//
//		uchar* output = img.ptr<uchar>(j);//获取output的首地址
//
//		for (int i = 1; i < nChanels*src.cols; ++i)
//		{
//			//先赋值后计算
//			*output++ = saturate_cast < uchar>(5 * current[i] - (previous[i] + next[i] + current[i - 1] + current[i + 1]));
//
//		}
//
//	}
//
//	img.row(0).setTo(Scalar(0));//0行
//	img.row(img.rows - 1).setTo(Scalar(0));//rows-1行即最后一行
//	img.col(0).setTo(Scalar(0));//0列
//	img.col(img.cols - 1).setTo(Scalar(0));//cols-1列，即最后一列
//}
//
//
//
//
//int main()
//{
//	Mat I, J, J1;
//	Mat kernel = (Mat_<char>(3, 3) << 0, -1, 0, -1, 5, -1, 0, -1, 0);
//
//	I = imread("E:\\openCV_Pictures\\fig5_classical.jpg", 1);
//
//	filter2D(I, J1, I.depth(), kernel);
//
//	Sharpen(I, J);
//
//	imshow("input", I);
//	imshow("output", J);
//	imshow("output1", J1);
//	waitKey(0);
//	destroyAllWindows();
//	return 0;
//
//}



//////////////////////////////////对于两幅图片求和/////////////////////////////////

//#include <opencv2/imgcodecs.hpp>
//#include <opencv2/highgui.hpp>
//#include <opencv2/imgproc.hpp>
//#include <iostream>
//
//using namespace std;
//using namespace cv;
//
//void addPictureMethod(Mat& src_0, Mat& src_1, double& a, Mat& dst)
//{
//	
//	dst.create(src_0.size(), src_0.type());//创建一个与原始图像大小和类型相同的图像
//
//	int channels = src_0.channels();
//	int nRows = src_0.rows * channels;
//	int nCols = src_0.cols;
//	nCols = nCols * nRows;
//
//	int j;
//	uchar* p0;
//	uchar* p1;
//	uchar* p2;
//
//	p0 = src_0.ptr<uchar>(0);
//	p1 = src_1.ptr<uchar>(0);
//	p2 = dst.ptr<uchar>(0);
//	
//	for (j = 0; j < nCols; ++j)
//	{
//		p2[j] = (1 - a) * p0[j] + a * p1[j];
//	}
//
//}
//
//
//int main()
//{
//	Mat img_0, img_1, img_1_1, img_hand, img_func;
//
//	double alpha = 0.5; double beta; 
//	beta = (1.0 - alpha);
//
//	img_0 = imread("E:\\openCV_Pictures\\fig5_classical.jpg", 1);
//	img_1 = imread("E:\\openCV_Pictures\\fig6_views.jpg",1);
//
//	resize(img_1,img_1_1,img_0.size(), 0, 0, INTER_LINEAR);
//
//	addPictureMethod(img_0, img_1_1, alpha, img_hand);
//	
//	addWeighted(img_0, alpha, img_1_1, beta, 0.0, img_func);//使用内置函数addWeighted
//
//	imshow("1", img_0);
//	imshow("2", img_1_1);
//	imshow("3", img_hand);
//	imshow("4", img_func);
//	waitKey(0);
//
//	return 0;
//}





///////////////////////////////////////增强图片的对比度////////////////////////////////////



/////////////////////////////////官方案例///////////////

#include "tools.h"


//int main()
//{
//	/// 读入用户提供的图像
//	Mat image = imread("E:\\openCV_Pictures\\fig5_classical.jpg", 1);
//	Mat image_0 = image.clone();
//	Mat new_image = Mat::zeros(image.size(), image.type());
//	Mat new_image_0 = Mat::zeros(image.size(), image.type());
//
//	
//	double alpha, beta;
//	/// 初始化
//	cout << " Basic Linear Transforms " << endl;
//	cout << "-------------------------" << endl;
//	cout << "* Enter the alpha value [1.0-3.0]: ";
//	cin >> alpha;
//	cout << "* Enter the beta value [0-100]: ";
//	cin >> beta;
//
//	/// 执行运算 new_image(i,j) = alpha*image(i,j) + beta
//
//	new_image = contrastBrightnessByOfficial(image_0, alpha, beta);
//	
//	new_image_0 = contrastBrightnessByC(image_0, alpha, beta);//使用opencv自带的函数
//
//	
//	imshow("Original Image", image);
//	imshow("New Image", new_image);
//	imshow("New Image2", new_image_0);
//
//	bool Match = matIsEqual(new_image, new_image_0);
//
//	cout << "the matrix is :"<< Match << endl;
//
//
//	waitKey(0);
//	return 0;
//}





/////////////////////////////////////byMyself///////////////

//int main()
//{
//	Mat I, J, K;
//	
//	double a = 2.2;
//	double b = 50;
//	
//	I = imread("E:\\openCV_Pictures\\fig5_classical.jpg", 1);
//	Mat L = Mat::zeros(I.size(), I.type());
//
//	Mat I_clone = I.clone();
//	Mat I_clone2 = I.clone();
//
//	J = contrastBrightnessByC(I_clone, a, b);
//	K = contrastBrightnessByRA(I_clone, a, b);
//
//	I_clone2.convertTo(L, -1, 1.5, 3);//-1 表示和源矩阵数据类型一致；
//
//
//	imshow("1", I);
//	imshow("2", J);
//	imshow("3", K);
//	imshow("4", L);
//
//	waitKey(0);
//	
//	return 0;
//}




///////////////////////////伽马校正///////////////////////

int main()
{
	Mat img_src, img_rsd;

	double gamma = 0.4;

	img_src = imread("E:\\openCV_Pictures\\fig8_buildings.jpg");

	img_rsd = Mat::zeros(img_src.size(), img_src.type());

	img_rsd = gammaResived(img_src, gamma);

	namedWindow("input", 0);
	namedWindow("output", 0);
	imshow("input", img_src);
	imshow("output", img_rsd);
	waitKey(0);

}




//////////////////////////DFT/////////////////////
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>

using namespace cv;
using namespace std;

static void help(void)
{
	cout << endl
		<< "This program demonstrated the use of the discrete Fourier transform (DFT). " << endl
		<< "The dft of an image is taken and it's power spectrum is displayed." << endl
		<< "Usage:" << endl
		<< "./discrete_fourier_transform [image_name -- default ../data/lena.jpg]" << endl;
}
int main(int argc, char ** argv)
{
	help();
	const char* filename = argc >= 2 ? argv[1] : "../data/lena.jpg";
	Mat I = imread(filename, IMREAD_GRAYSCALE);
	if (I.empty()) {
		cout << "Error opening image" << endl;
		return -1;
	}
	Mat padded;                            //expand input image to optimal size
	int m = getOptimalDFTSize(I.rows);
	int n = getOptimalDFTSize(I.cols); // on the border add zero values
	copyMakeBorder(I, padded, 0, m - I.rows, 0, n - I.cols, BORDER_CONSTANT, Scalar::all(0));
	Mat planes[] = { Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F) };
	Mat complexI;
	merge(planes, 2, complexI);         // Add to the expanded another plane with zeros
	dft(complexI, complexI);            // this way the result may fit in the source matrix
	// compute the magnitude and switch to logarithmic scale
	// => log(1 + sqrt(Re(DFT(I))^2 + Im(DFT(I))^2))
	split(complexI, planes);                   // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
	magnitude(planes[0], planes[1], planes[0]);// planes[0] = magnitude
	Mat magI = planes[0];
	magI += Scalar::all(1);                    // switch to logarithmic scale
	log(magI, magI);
	// crop the spectrum, if it has an odd number of rows or columns
	magI = magI(Rect(0, 0, magI.cols & -2, magI.rows & -2));
	// rearrange the quadrants of Fourier image  so that the origin is at the image center
	int cx = magI.cols / 2;
	int cy = magI.rows / 2;
	Mat q0(magI, Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
	Mat q1(magI, Rect(cx, 0, cx, cy));  // Top-Right
	Mat q2(magI, Rect(0, cy, cx, cy));  // Bottom-Left
	Mat q3(magI, Rect(cx, cy, cx, cy)); // Bottom-Right
	Mat tmp;                           // swap quadrants (Top-Left with Bottom-Right)
	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);
	q1.copyTo(tmp);                    // swap quadrant (Top-Right with Bottom-Left)
	q2.copyTo(q1);
	tmp.copyTo(q2);
	normalize(magI, magI, 0, 1, NORM_MINMAX); // Transform the matrix with float values into a
											// viewable image form (float between values 0 and 1).
	imshow("Input Image", I);    // Show the result
	imshow("spectrum magnitude", magI);
	waitKey();
	return 0;
}