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
//			////先赋值后计算*output++,赋值完成后每次指向下一个地址；
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




/////////////////////////////伽马校正///////////////////////
//
//int main()
//{
//	Mat img_src, img_rsd;
//
//	double gamma = 0.4;
//
//	img_src = imread("E:\\openCV_Pictures\\fig8_buildings.jpg");
//
//	img_rsd = Mat::zeros(img_src.size(), img_src.type());
//
//	img_rsd = gammaResived(img_src, gamma);
//
//	namedWindow("input", 0);
//	namedWindow("output", 0);
//	imshow("input", img_src);
//	imshow("output", img_rsd);
//	waitKey(0);
//
//}


///////////////////////////////////////////////////////////////////////////////////////////////
                                              /*DFT*/
///////////////////////////////////////////////////////////////////////////////////////////////
//#include "opencv2/core.hpp"
//#include "opencv2/imgproc.hpp"
//#include "opencv2/imgcodecs.hpp"
//#include "opencv2/highgui.hpp"
//#include <iostream>
//
//using namespace cv;
//using namespace std;
//
//static void help(void)
//{
//	cout << endl
//		<< "This program demonstrated the use of the discrete Fourier transform (DFT). " << endl
//		<< "The dft of an image is taken and it's power spectrum is displayed." << endl
//		<< "Usage:" << endl
//		<< "./discrete_fourier_transform [image_name -- default ../data/lena.jpg]" << endl;
//}
//int main()
//{
//
//	Mat I = imread("E:\\openCV_Pictures\\fig7_txt_H.jpg", IMREAD_GRAYSCALE);
//
//	/*if (I.empty()) {
//		cout << "Error opening image" << endl;
//		return -1;
//	}*/
//
//
//	Mat padded;                            //expand input image to optimal size
//	int m = getOptimalDFTSize(I.rows); //转换为2,3,5倍数相乘的形式；
//	int n = getOptimalDFTSize(I.cols); 
//
//	//把灰度图像放在左上角,在右边和下边扩展图像,扩展部分填充为0
//	copyMakeBorder(I, padded, 0, m - I.rows, 0, n - I.cols, BORDER_CONSTANT, Scalar::all(0));// on the border add zero values
//
//	//这里是获取了两个Mat,一个用于存放dft变换的实部，一个用于存放虚部,初始的时候,实部就是图像本身,虚部全为零
//	Mat planes[] = { Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F) };
//	Mat complexI;
//
//	//将几个单通道的mat融合成一个多通道的mat,这里融合的complexImg既有实部又有虚部
//	merge(planes, 2, complexI);         // Add to the expanded another plane with zeros
//
//	//对上边合成的mat进行傅里叶变换,支持原地操作,傅里叶变换结果为复数.通道1存的是实部,通道二存的是虚部
//	dft(complexI, complexI);            // this way the result may fit in the source matrix
//
//
//	/*这一部分是为了计算dft变换后的幅值，傅立叶变换的幅度值范围大到不适合在屏幕上显示。
//	高值在屏幕上显示为白点，而低值为黑点，高低值的变化无法有效分辨。
//	为了在屏幕上凸显出高低变化的连续性，我们可以用对数尺度来替换线性尺度, 
//	以便于显示幅值, 计算公式如下*/
//	// compute the magnitude and switch to logarithmic scale
//	// => log(1 + sqrt(Re(DFT(I))^2 + Im(DFT(I))^2))
//	split(complexI, planes);                   // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
//	magnitude(planes[0], planes[1], planes[0]);// planes[0] = magnitude
//	Mat magI = planes[0];
//	magI += Scalar::all(1);                    // switch to logarithmic scale
//	log(magI, magI);
//
//	//修剪频谱,如果图像的行或者列是奇数的话,那其频谱是不对称的,因此要修剪
//	// crop the spectrum, if it has an odd number of rows or columns
//	magI = magI(Rect(0, 0, magI.cols & -2, magI.rows & -2));
//	
//
//	//重新分配象限，使（0,0）移动到图像中心，  
//	//在《数字图像处理》中，傅里叶变换之前要对源图像乘以（-1）^(x+y)进行中心化。  
//	//这是是对傅里叶变换结果进行中心化
//	// rearrange the quadrants of Fourier image  so that the origin is at the image center
//	int cx = magI.cols / 2;
//	int cy = magI.rows / 2;
//	Mat q0(magI, Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
//	Mat q1(magI, Rect(cx, 0, cx, cy));  // Top-Right
//	Mat q2(magI, Rect(0, cy, cx, cy));  // Bottom-Left
//	Mat q3(magI, Rect(cx, cy, cx, cy)); // Bottom-Right
//	Mat tmp;                           // swap quadrants (Top-Left with Bottom-Right)
//	q0.copyTo(tmp);
//	q3.copyTo(q0);
//	tmp.copyTo(q3);
//	q1.copyTo(tmp);                    // swap quadrant (Top-Right with Bottom-Left)
//	q2.copyTo(q1);
//	tmp.copyTo(q2);
//
//	//这一步的目的仍然是为了显示,但是幅度值仍然超过可显示范围[0,1],我们使用 normalize() 函数将幅度归一化到可显示范围
//	normalize(magI, magI, 0, 1, NORM_MINMAX); // Transform the matrix with float values into a
//											// viewable image form (float between values 0 and 1).
//	imshow("Input Image", I);    // Show the result
//	imshow("spectrum magnitude", magI);
//	waitKey();
//	return 0;
//}







//int main()
//{
//	Mat I, J;
//
//	I = imread("E:\\openCV_Pictures\\fig8_median.jpg");
//
//
//	for (int i = 1; i < 31; i = i + 2)
//	{
//		medianBlur(I, J, i);//中值滤波
//		
//		imshow("1", J);
//		waitKey();
//	}
//		
//	return 0;
//
//}


//
//void dealMedianGary(const cv::Mat *src, cv::Mat *dst, int n, int m)
//void MedianGary(cv::Mat *dst, int n, int m)
//
//int main(void)
//{
//	// [1] src读入图片
//	cv::Mat src = cv::imread("pic3.jpg");
//	// [2] dst目标图片
//	cv::Mat dst;
//	// [3] 中值滤波 RGB （5*5）的核大小
//	dealMedianRGB(src, dst, 5, 5);
//	// [4] 窗体显示
//	imshow("src", src);
//	imshow("dst", dst);
//	waitKey(0);
//	destroyAllWindows();
//	return 0;
//}
//
//
//void dealMedianGary(const cv::Mat *src, cv::Mat *dst, int n, int m)
//{
//	// [1] 初始化
//	*dst = (*src).clone();
//	// [2] 彩色图片通道分离
//	std::vector<cv::Mat> channels;
//	cv::split(*src, channels);
//	// [3] 滤波
//	for (int i = 0; i < 3; i++) {
//		MedianGary(&channels[i], n, m);
//	}
//	// [4] 合并返回
//	cv::merge(channels, *dst);
//	return;
//}
//
//void MedianGary(cv::Mat *dst, int n, int m)
//{
//	unsigned char *_medianArray = NULL;
//
//	// [2] 初始化
//	_medianArray = (unsigned char *)malloc(sizeof(unsigned char)*(n*m));
//
//	printf(" 5 * 5 start ... ... \n");
//	// [2-1] 扫描
//	for (int i = 0; i < dst->height; i++) {
//		for (int j = 0; j < dst->width; j++) {
//			// [2-2] 忽略边缘
//			if ((i > 1) && (j > 1) && (i < dst->height - 2)
//				&& (j < dst->width - 2))
//			{
//				// [2-3] 保存数组
//				int _count = 2;
//				for (int num = 0; num < (n*m); num += 5, _count--)
//				{
//					_medianArray[num] = *((unsigned char*)(dst->imageData + (i - _count)*dst->widthStep + (j - 2)));
//					_medianArray[num + 1] = *((unsigned char*)(dst->imageData + (i - _count)*dst->widthStep + (j - 1)));
//					_medianArray[num + 2] = *((unsigned char*)(dst->imageData + (i - _count)*dst->widthStep + (j)));
//					_medianArray[num + 3] = *((unsigned char*)(dst->imageData + (i - _count)*dst->widthStep + (j + 1)));
//					_medianArray[num + 4] = *((unsigned char*)(dst->imageData + (i - _count)*dst->widthStep + (j + 2)));
//				}
//				// [2-5] 求中值并保存
//				*((unsigned char*)(dst->imageData + i * dst->widthStep + j)) = medianValue(_medianArray, (n*m));
//			}//for[2-2]
//		}
//	}//for[2-1]
//}









////////////////////////////////////////////////////////////////////////////////////////
								   /*平滑图像*/
////////////////////////////////////////////////////////////////////////////////////////

///官方例子
//#include <iostream>
//#include "opencv2/imgproc.hpp"
//#include "opencv2/imgcodecs.hpp"
//#include "opencv2/highgui.hpp"
//using namespace std;
//using namespace cv;
//int DELAY_CAPTION = 1500;
//int DELAY_BLUR = 100;
//int MAX_KERNEL_LENGTH = 31;
//
//Mat src; Mat dst;
//char window_name[] = "Smoothing Demo";
//int display_caption(const char* caption);
//int display_dst(int delay);
//
//int main()
//{
//	namedWindow(window_name, WINDOW_AUTOSIZE);
//	src = imread("E:\\openCV_Pictures\\fig5_classical.jpg", IMREAD_COLOR);
//	
//	if (src.empty())
//	{
//		printf(" Error opening image\n");
//		printf(" Usage: ./Smoothing [image_name -- default ../data/lena.jpg] \n");
//		return -1;
//	}
//
//	if (display_caption("Original Image") != 0)//窗口显示"Original Image"
//	{
//		return 0;
//	}
//
//	dst = src.clone();//复制图像
//
//	if (display_dst(DELAY_CAPTION) != 0)
//	{
//		return 0;
//	}
//
//	if (display_caption("Homogeneous Blur") != 0)
//	{
//		return 0;
//	}
//
//	for (int i = 1; i < MAX_KERNEL_LENGTH; i = i + 2)
//	{
//		blur(src, dst, Size(i, i), Point(-1, -1));//归一化滤波
//		if (display_dst(DELAY_BLUR) != 0)
//		{
//			return 0;
//		}
//	}
//
//	if (display_caption("Gaussian Blur") != 0)
//	{
//		return 0;
//	}
//
//	for (int i = 1; i < MAX_KERNEL_LENGTH; i = i + 2)
//	{
//		GaussianBlur(src, dst, Size(i, i), 0, 0);//高斯滤波
//		if (display_dst(DELAY_BLUR) != 0)
//		{
//			return 0;
//		}
//	}
//
//	if (display_caption("Median Blur") != 0)
//	{
//		return 0;
//	}
//
//	for (int i = 1; i < MAX_KERNEL_LENGTH; i = i + 2)
//	{
//		medianBlur(src, dst, i);//中值滤波
//		if (display_dst(DELAY_BLUR) != 0)
//		{
//			return 0;
//		}
//	}
//
//	if (display_caption("Bilateral Blur") != 0)
//	{
//		return 0;
//	}
//
//	for (int i = 1; i < MAX_KERNEL_LENGTH; i = i + 2)
//	{
//		bilateralFilter(src, dst, i, i * 2, i / 2);//双边滤波
//		if (display_dst(DELAY_BLUR) != 0)
//		{
//			return 0;
//		}
//	}
//
//	display_caption("Done!");
//	return 0;
//}
//int display_caption(const char* caption)
//{
//	dst = Mat::zeros(src.size(), src.type());
//	putText(dst, caption,
//		Point(src.cols / 4, src.rows / 2),
//		FONT_HERSHEY_COMPLEX, 1, Scalar(255, 255, 255));
//	return display_dst(DELAY_CAPTION);
//}
//int display_dst(int delay)
//{
//	imshow(window_name, dst);
//	int c = waitKey(delay);
//	if (c >= 0) { return -1; }
//	return 0;
//}


///练习版--（注：边缘图像不处理，核为3*3）

#include "tools.h"

//int main()
//{
//	Mat img_src, img_dst, img_dst1;
//	img_src = imread("E:\\openCV_Pictures\\fig8_median.jpg");
//
//	MedianBlur(img_src, img_dst);
//	medianBlur(img_src, img_dst1, 3);//中值滤波
//
//	int match = matIsEqual(img_dst, img_dst1);
//
//	if (match == 1)
//		printf("The matrix is equal！");
//	else
//		printf("The matrix is not equal！");
//
//	imshow("input", img_src);
//	imshow("output", img_dst);
//	imshow("output1", img_dst1);
//	waitKey(0);
//
//	return 0;
//}




/////////////////////////////////////膨胀和腐蚀//////////////////////
//int main()
//{
//	int para = 0;
//	Mat img_src, img_dst, img_dst1;
//	img_src = imread("E:\\openCV_Pictures\\fig9_font.jpg");
//
//	//ErodingAndDilating(img_src, img_dst,para);//para ,1为膨胀，0为腐蚀；
//
//
//	/*形态学操作*/
//	/*Opening: MORPH_OPEN: 2
//	Closing : MORPH_CLOSE : 3
//	Gradient : MORPH_GRADIENT : 4
//	Top Hat : MORPH_TOPHAT: 5
//	Black Hat : MORPH_BLACKHAT: 6*/
//	int arr[5] = { 2,3,4,5,6 };
//	
//	Mat element = getStructuringElement(0, Size(3,3), Point(0, 0));
//
//	for (int i = 0; i < 5; ++i) 
//	{
//		morphologyEx(img_src, img_dst, arr[i], element);
//
//		imshow("output", img_dst);
//		waitKey(0);
//	}
//
//	return 0;
//}




/////////////////////////////////图像金字塔////////////////////////////////////


//#include "iostream"
//#include "opencv2/imgproc.hpp"
//#include "opencv2/imgcodecs.hpp"
//#include "opencv2/highgui.hpp"
//
//using namespace std;
//using namespace cv;

//const char* window_name = "Pyramids Demo";
//int main()
//{
//	
//	const char* filename = "E:\\openCV_Pictures\\fig5_classical.jpg";
//	
//
//	Mat src = imread(filename);
//
//	// Check if image is loaded fine
//	if (src.empty()) {
//		printf(" Error opening image, Please reload!\n");
//		return -1;
//	}
//
//	for (;;)
//	{
//		imshow(window_name, src);
//		char c = (char)waitKey(0);
//		if (c == "p")
//		{
//			break;
//		}
//		else if (c == 'i') //上采样
//		{
//			pyrUp(src, src, Size(src.cols * 2, src.rows * 2));
//			printf("** Zoom In: Image x 2 \n");
//		}
//		else if (c == 'o')//下采样
//		{
//			pyrDown(src, src, Size(src.cols / 2, src.rows / 2));
//			printf("** Zoom Out: Image / 2 \n");
//		}
//	}
//
//	system("pause");
//	return 0;
//}


//int main()
//{
//	Mat src, dst;
//
//	src = imread("E:\\openCV_Pictures\\fig10_dog.jpg");
//
//
//	for (int i = 0; i < 8; i++)
//	{
//	/*	pyrDown(src, src, Size(src.cols / 2, src.rows / 2));*/
//		pyrUp(src, src, Size(src.cols * 2, src.rows * 2));
//
//
//		namedWindow("input", 0);
//		imshow("input", src);
//		waitKey(0);
//	}	
//
//}



///////////////////////////////////////阈值操作//////////////////////////////

//#include "iostream"
//#include "opencv2/imgproc.hpp"
//#include "opencv2/imgcodecs.hpp"
//#include "opencv2/highgui.hpp"
//#include "tools.h"
//
//using namespace std;
//using namespace cv;
//
//
//int main()
//{
//	Mat img_src, img_gray, img_clone, img_dst, img_dst_official;
//	int arr[5] = { 0,1,2,3,4 };
//	int len = sizeof(arr) / sizeof(arr[0]);//obtain the length of arr
//	int const  threshold_value = 50;
//	int const  max_binary_value = 255;
//
//	
//	img_src = imread("E:\\openCV_Pictures\\fig10_dog.jpg");
//
//
//
//	//当img_src.data=0，0代表false，！0就是true，即当图像为空时，程序结束返回-1；
//	if (!img_src.data)
//	{
//		cout << "The image is empty!" << endl;
//		return -1;
//	}
//
//	cvtColor(img_src, img_gray, COLOR_RGB2GRAY);
//
//	imshow("input", img_gray);
//	
//
//	for (int i=0; i<len; ++i)
//	{
//		img_clone = img_gray.clone();
//		thresholdOperation(img_clone, arr[i], threshold_value);
//
//		//输入图像；输出图像；阈值；最大值；阈值算法的类型（供计5种）
//		threshold(img_clone, img_dst_official, threshold_value, max_binary_value, arr[i]);
//		
//		namedWindow("myMethod", 1);
//		namedWindow("official", 1);
//		
//		imshow("myMethod", img_clone);
//		imshow("official", img_dst_official);
//
//		waitKey(0);
//	}
//		
//	return 0;
//
//}



//////////////////////////////////////线性滤波///////////////////
//#include "opencv2/imgproc/imgproc.hpp"
//#include "opencv2/highgui/highgui.hpp"
//#include <stdlib.h>
//#include <stdio.h>
//
//using namespace cv;


//int main()
//{
//	/// 声明变量
//	Mat src, dst;
//
//	Mat kernel;
//	Point anchor;//锚点
//	double delta;
//	int ddepth;
//	int kernel_size;
//	char window_name[] = "filter2D Demo";
//
//	int c;
//
//	/// 载入图像
//	src = imread("E:\\openCV_Pictures\\dragon.jpg");
//
//	if (!src.data)
//	{
//		return -1;
//	}
//
//	/// 创建窗口
//	namedWindow(window_name, 0);
//
//	/// 初始化滤波器参数
//	anchor = Point(-1, -1);
//	delta = 0;
//	ddepth = -1;
//
//	/// 循环 - 每隔0.5秒，用一个不同的核来对图像进行滤波
//	int ind = 0;
//	while (true)
//	{
//		c = waitKey(500);
//		/// 按'ESC'可退出程序
//		if ((char)c == 27)
//		{
//			break;
//		}
//
//		/// 更新归一化块滤波器的核大小
//		kernel_size = 3 + 2 * (ind % 5);
//		kernel = Mat::ones(kernel_size, kernel_size, CV_32F) / (float)(kernel_size*kernel_size);
//
//		/// 使用滤波器
//		filter2D(src, dst, ddepth, kernel, anchor, delta, BORDER_DEFAULT);
//		imshow(window_name, dst);
//		ind++;
//	}
//
//	return 0;
//}


/*byMyself*/
//
//int main()
//{
//	Mat img_src, img_dst, kernel;
//	int ksize;
//	Point anchor = Point(-1, -1);//表示核的中心点
//	double delta = 0;
//	int ddepth = -1;//输入图像和输出图像的类型相同
//
//	img_src = imread("E:\\openCV_Pictures\\dragon.jpg");
//
//	if (!img_src.data)
//	{
//		return -1;
//	}
//
//	for (int index = 0; index<5; index++)
//	{
//		ksize = 3 + 2 * (index % 5);
//
//		//建立核矩阵
//		kernel = Mat::ones(ksize, ksize, CV_32F) / (float)(ksize*ksize);//(float)转换类型
//		filter2D(img_src, img_dst, ddepth, kernel, anchor, delta, BORDER_DEFAULT);
//
//		imshow("img_dst", img_dst);
//		waitKey(0);
//	}
//
//}



///////////////////////////////////给图像添加边界//////////////////////////////////

#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
//
//
//using namespace cv;
//// Declare the variables
//Mat src, dst;
//int top, bottom;
//
//int borderType = BORDER_CONSTANT;
//const char* window_name = "copyMakeBorder Demo";
//RNG rng(12345);//rng ，这是一个随机数生成器， 用来产生随机边界色彩。12345是随机种子编号
//
//int main(int argc, char** argv)
//{
//	
//	src = imread("E:\\openCV_Pictures\\fig5_classical.jpg", IMREAD_COLOR); // Load an image
//
//
//	// Check if image is loaded fine
//	if (src.empty()) {
//		printf(" Error opening image\n");
//		printf(" Program Arguments: [image_name -- default ../data/lena.jpg] \n");
//		return -1;
//	}
//
//	/*if (!src.data)
//	{
//		printf("The image is not exist！")；
//		return -1；
//	}*/
//
//
//	// Brief how-to for this program
//	printf("\n \t copyMakeBorder Demo: \n");
//	printf("\t -------------------- \n");
//	printf(" ** Press 'c' to set the border to a random constant value \n");
//	printf(" ** Press 'r' to set the border to be replicated \n");
//	printf(" ** Press 'ESC' to exit the program \n");
//	namedWindow(window_name, WINDOW_AUTOSIZE);
//	
//	// Initialize arguments for the filter，边框的宽度
//	top = (int)(0.05*src.rows); 
//	bottom = top;
//	int left = (int)(0.05*src.cols); 
//	int right = left;
//	
//	for (;;)
//	{
//		Scalar value(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));//边界的像素值范围
//
//		copyMakeBorder(src, dst, top, bottom, left, right, borderType, value);
//		imshow(window_name, dst);
//		
//		char c = (char)waitKey(500);
//		if (c == 27)
//		{
//			break;
//		}
//		else if (c == 'c')
//		{
//			borderType = BORDER_CONSTANT;
//		}
//		else if (c == 'r')
//		{
//			borderType = BORDER_REPLICATE;
//		}
//	}
//	return 0;
//}


/*byMyself*/
//int main()
//{
//
//	Mat img_src, img_dst;
//	RNG rng(1234);
//	int top, bottom, left, right;
//	int borderType = BORDER_CONSTANT;
//
//	img_src = imread("E:\\openCV_Pictures\\fig5_classical.jpg");
//
//	if (!img_src.data)
//	{
//		return -1;
//	}
//
//	top = (int)img_src.rows*0.05;
//	top = bottom;
//	right = (int)img_src.cols*0.05;
//	left = right;
//
//	for (;;)
//	{
//		Scalar value = (rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
//		copyMakeBorder(img_src, img_dst, top, bottom, left, right, borderType, value);
//
//		char c = (char)waitKey(500);
//		if (c == 27)
//		{
//			break;
//		}
//		else if (c == 'c')
//		{
//			borderType = BORDER_CONSTANT;
//		}
//		else if (c == 'r')
//		{
//			borderType = BORDER_REPLICATE;
//		}
//
//	}
//
//	return 0;
//}


/////////////////////////////CV::Mat.compare（），逐像素比较//////////////////////////




//int main()
//{
//
//	Mat img_src1,  img_src2, img_dst;
//	
//
//	img_src1 = imread("E:\\openCV_Pictures\\fig5_classical.jpg", 0);//单通道图
//	img_src2 = imread("E:\\openCV_Pictures\\fig5_classical2.jpg", 0);
//
//	imshow("input1", img_src1);
//	imshow("input2", img_src2);
//
//	if (img_src1.empty() || img_src2.empty())
//	{
//		return -1;
//	}
//
//	for (int i = 0; i < 6; i++)
//	{
//		compare(img_src1, img_src2, img_dst, i);
//		imshow("output", img_dst);
//		waitKey(0);
//
//	}
//
//}



////////////////////////////////////////////////////////////////
                        /*Sobel 导数*/ 
////////////////////////////////////////////////////////////////

#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>
using namespace cv;
using namespace std;
//
int main()
{

	
	Mat image, src, src_gray;
	Mat grad;

	int ksize = 3;
	int scale = 3;
	int delta =0;
	int ddepth = CV_16S;

	image = imread("E:\\openCV_Pictures\\house.png", IMREAD_COLOR); // Load an image
	

	//判断图像是否为空
	if (image.empty())
	{
		return -1;
	}

	
	/*for (;;)
	{*/
		 //Remove noise by blurring with a Gaussian filter ( kernel size = 3 )
		/*GaussianBlur(image, src, Size(3, 3), 0, 0, BORDER_DEFAULT);*/
		
		cvtColor(image, src_gray, COLOR_BGR2GRAY);

		Mat grad_x, grad_y;
		Mat abs_grad_x, abs_grad_y;
		Sobel(src_gray, grad_x, ddepth, 1, 0, ksize, scale, delta, BORDER_DEFAULT);
		Sobel(src_gray, grad_y, ddepth, 0, 1, ksize, scale, delta, BORDER_DEFAULT);
		
	/*	double maxVal_x, maxVal_y, minVal_x, minVal_y;
		int maxVal_x1[2] = {}, maxVal_y1[2] = {}, minVal_x1[2] = {}, minVal_y1[2] = {};
		minMaxIdx(abs(grad_x), &minVal_x, &maxVal_x, minVal_x1, maxVal_x1);
		minMaxIdx(abs(grad_y), &minVal_y, &maxVal_y, minVal_y1, maxVal_y1);*/



		double maxVal_x, maxVal_y, minVal_x, minVal_y;
		cv::Point maxVal_x1 , maxVal_y1, minVal_x1, minVal_y1;
		cv::minMaxLoc(grad_x, &minVal_x, &maxVal_x, &minVal_x1, &maxVal_x1);
		cv::minMaxLoc(grad_y, &minVal_y, &maxVal_y, &minVal_y1, &maxVal_y1);


		//converting back to CV_8U
		convertScaleAbs(grad_x, abs_grad_x);//负数变整数，超过255，变255
		convertScaleAbs(grad_y, abs_grad_y);
		addWeighted(abs_grad_x, 1, abs_grad_y, 1, 0, grad);

		
		imshow("grad_x_NoAbs", grad_x);
		imshow("grad_y_NoAbs", grad_y);
		imshow("grad_x_Abs", abs_grad_x);
		imshow("grad_y_Abs", abs_grad_y);
		imshow("grad_xy_weight", grad);
	

		waitKey(0);
		
		



	//	char key = (char)waitKey(0);
	//	if (key == 27)
	//	{
	//		return 0;
	//	}
	//	if (key == 'k' || key == 'K')
	//	{
	//		ksize = ksize < 30 ? ksize + 2 : -1;
	//	}
	//	if (key == 's' || key == 'S')
	//	{
	//		scale++;
	//	}
	//	if (key == 'd' || key == 'D')
	//	{
	//		delta++;
	//	}
	//	if (key == 'r' || key == 'R')
	//	{
	//		scale = 1;
	//		ksize = -1;
	//		delta = 0;
	//	}
	//}
	return 0;
}





////////////////////////////////////////////////sobel by myself//////////////////////////////////
//#include "opencv2/imgproc.hpp"
//#include "opencv2/imgcodecs.hpp"
//#include "opencv2/highgui.hpp"
//#include <iostream>
//#include "tools.h"
//using namespace cv;
//using namespace std;
//
//int main()
//{
//
//	Mat img_src, img_dst;
//
//	img_src = imread("E:\\openCV_Pictures\\fig5_classical.jpg", 0);
//
//	if (img_src.empty())
//	{
//		cout << "The image is empty!" << endl;
//		return -1;
//	}
//
//	sobelEdgeDetection(img_src, img_dst);
//
//	imshow("input", img_src);
//	imshow("output", img_dst);
//	waitKey(0);
//	destroyAllWindows();
//}




/////////////////////////////////测试uchar运算//////////////////////////////////////////
//#include <iostream>
//using namespace std;
//int main()
//{
//	
//
//	//Mat n = Mat::zeros(3, 3, CV_8U);
//
//	//Mat_<uchar> img = n;
//
//	//img(0,0) = -1;
//	//img(1, 1) = 300;
//
//	
//	uchar m = 1;
//	uchar n = 255;
//	uchar z1,z2;
//
//	z1 = m - n;
//	z2 = m + n;
//
//	//unsigned char的范围是0~255,在用cout输出的时候要显示数字的话记得进行int的强制转化才可以，否则都是输出的字符
//	cout << int(z1) << endl << int(z2);
//
//	system("pause");
//
//	return 0;
//	
//}


///////////////////生成高斯核/////////////////////////////////

/*长的表达式中注意不同数据的类型的计算时的转换，最后把他们统一为同一类型，
如果无法统一，低级的数据类型会向高级的数据类型转换，注意整体之间的运算出现
如果小于0时，会认为是0，如果整数有带小数则默认去掉，向下取整，因为如果出现
负数和小数就用float和double*/
#include "tools.h"


//int main()
//{
//	int size = 3;
//	double sigma = 0.8;
//
//	Mat kernel = Mat::zeros(size, size, CV_32F);
//
//	getGaussKernel(size, sigma, kernel);
//
//	cout << kernel << endl;
//
//	system("pause");
//
//}

////////////////////////////////laplace//////////////////////////////////

//int main()
//{
//	Mat img_src, img_float, img_v1, img_v2;
//
//	img_src = imread("E:\\openCV_Pictures\\moon.jpg", 0);
//
//	if (img_src.empty())
//		return -1;
//
//	int rows = img_src.rows;
//	int cols = img_src.cols;
//
//	img_src.convertTo(img_float, CV_32F);
//
//	Mat_<float> img_laplace = Mat::zeros(rows, cols, CV_32F);
//	
//	Mat_<float> img = img_float;
//
//	for (int i = 1; i < rows - 1; i++)
//		for (int j = 1; j < cols - 1; j++)
//			img_laplace(i, j) = 0 * img(i - 1, j - 1) - 1 * img(i - 1, j) + 0 * img(i - 1, j + 1)\
//			- 1 * img(i, j - 1) + 4 * img(i, j) - 1 * img(i, j + 1)\
//			+ 0 * img(i + 1, j - 1) - 1 * img(i + 1, j) + 0 * img(i + 1, j + 1);
//	
//	/*for (int i = 1; i < cols - 1; i++)
//		for (int j = 1; j < rows - 1; j++)
//		{
//			float pix = img_laplace(i, j);
//			if (pix >= 1.0f)
//				img_laplace(i, j) = 1.0f;
//			else if (pix < 0)
//				img_laplace(i, j) = 0.0f;
//			else
//				img_laplace(i, j) = pix;
//		}*/
//	
//	img_v1 = img_float - img_laplace;
//	img_v2 = img_float + img_laplace;
//
//	//转换的时候准备一个新的Mat，不要使用原Mat作为输出的结果
//	img_v1.convertTo(img_v1, CV_8U);
//	img_v2.convertTo(img_v2, CV_8U);
//
//	imshow("input", img_src);
//	imshow("output", img_laplace);
//	imshow("output1", img_v2);
//	waitKey(0);
//
//	return 0;
//
//}
