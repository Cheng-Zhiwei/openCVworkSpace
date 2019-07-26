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
//int main(int argc, char** argv) {//int argc, char** argv������ȥ����ʹ��
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





/////////////////////����Mat����/////////////////////////////
//#include <opencv2/opencv.hpp>
//#include <opencv2/imgproc.hpp>
//#include <opencv2/highgui.hpp>
//#include <opencv2/highgui/highgui_c.h>
//#include <opencv2/imgcodecs.hpp>
//#include <iostream> // ����cout endl ��ӡ
//#include <string>
//#include <opencv2/imgproc/imgproc_c.h> //����IplImage


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



/*����һ��ͨ��Mat��������*/
//int main() {
//
//	Mat M(2, 2, CV_8UC3, Scalar(0, 0, 255));
//	std::cout << "M = " << std::endl << " "<< M << std::endl;//C++��ӡ���
//
//	system("pause");
//	return 0;
//}


/*��������ʹ��C���鲢ͨ�����캯����ʼ��*/
//int main() {
//
//	int sz[3] = { 2,2,2 };//����C����
//
//	Mat M(2, sz, CV_8UC3, Scalar::all(0));//Scalar����Ԫ������Ϊ0
//
//	system("pause");
//	return 0;
//}


/*��������Ϊ�Ѵ��ڵ�ipiimageָ�봴����Ϣͷ*/
//int main() {
//
//	IplImage* img = cvLoadImage("E:\\fig1.jpg",1)��//4.0�汾���޴˺�����IpiImageΪָ��
//
//	Mat mtx��iimg��;//������Ϣͷ
//
//	system("pause");
//	return 0;
//}




/*�����ģ�����create��������
����Mat���create������Ա��������Mat��ĳ�ʼ������
�˷�������Ϊ�������ֵ��ֻ���ڸı�ߴ�ʱΪ���󿪱ٿռ�
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



/*�����壬����matlabʽ��ʼ����ʽ
���������Matlab ��ʽ�ĳ�ʼ����ʽ�� zeros( ) , ones(), eyes�� �� ��ʹ�����·�
��ָ���ߴ���������ͣ�*/

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


/*����������matlabʽ��ʼ����ʽ*/
//int main() {
//
//	Mat C = (Mat_<double>(3, 3) << 0, 1, 2, 3, 4, 5, 6, 7, 8);
//	std::cout << C << std::endl;
//
//	system("pause");
//	return 0;
//}



/*�����ߣ�Ϊ�Ѵ��ڵĶ��󴴽���Ϣͷ*/

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


/////////////////////////02 openCV��ʽ���������/////////////////////////

//int main() {
//
//	/*��������randu��������һ��������ľ���*/
//	Mat r = Mat(10, 3, CV_8UC3);
//	randu(r, Scalar::all(0), Scalar::all(255));
//
//	//opencvĬ�Ϸ��
//	std::cout << "r(Ĭ�Ϸ��)=" << r << std::endl;
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
//	//��ά��Ķ�������
//	Point2f p(6, 2);
//	std::cout << "2D_p=" << p << std::endl;
//
//	//��ά��Ķ�������
//	Point3f p1(1, 2, 3);
//	std::cout << "3D_p1=" << p << std::endl;
//
//	//����Mat��std::vector
//	std::vector<float> v;
//	v.push_back(3);
//	v.push_back(5);
//	v.push_back(7);
//	std::cout << "3Dp1=" << Mat(v) << std::endl;//ע�⣺����V��ʾ��ΪMat��v����
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



///////////////////////////03_ɨ��ͼ��//////////////////////////////////////////////////


/*����֪ʶ----C++������*/
#include <iostream>

//01��������
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

////������Ϊ�������β�
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
////ֱ�Ӵ��ݲ�������
//void swap1(int a, int b) {
//	int temp = a;
//	a = b;
//	b = temp;
//}
////����ָ��
//void swap2(int *p1, int *p2) {
//	int temp = *p1;
//	*p1 = *p2;
//	*p2 = temp;
//}
////�����ô���
//void swap3(int &a, int &b) {
//	int temp = a;
//	a = b;
//	b = temp;
//}


////������Ϊ��������ֵ
/*���г����õ��Ľ��Ϊnu=20��num2=20���������£�
����num1=10,Ȼ���ʹ�ú�����int &n = num1,n��num1����ͬһ���ڴ��ֵ��
��n=n+10��n=20����num1=20��Ȼ��num2 = num1 = 20*/
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











#include <opencv2/core.hpp>  //����ģ�飬�����˻��������ݽṹ����������
#include <opencv2/core/utility.hpp>  //����getTickFrequency()�Ⱥ���
#include "opencv2/imgcodecs.hpp"  //ͼƬ����롢���ر���֮��ĺ���
#include <opencv2/highgui.hpp>  //��Ƶ��׽��ͼ�����Ƶ�ı�����롢ͼ�ν�������Ľӿ�
#include <iostream>  //C++���������ݵ���ʽ�����������ͷ�ļ�
#include <sstream>  //ʹ��stringsteam������Ҫ�õ���ͷ�ļ�

	using namespace std;
	using namespace cv;

	static void help() //��̬����������������������Ϣ�������в�����Ϣ
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



	/*������C������[]��ָ�룩������������ʱ��Ŀ��ַ�������ַ�������������
	���е�&�������ã��൱�ڸ���������������˵ڶ������֣����ó�ʼ��ĳ��������
	����ʹ�ø��������ƻ�ԭ��������ָ��ñ�������ָ����һ�������𣬾�����ο�C++����*/


	//Mat& ����Mat���ͷ���ֵ������
	Mat& ScanImageAndReduceC(Mat& I, const uchar* table);//ͨ��Cָ�뷽ʽɨ��
	Mat& ScanImageAndReduceIterator(Mat& I, const uchar* table);//ͨ��������
	Mat& ScanImageAndReduceRandomAccess(Mat& I, const uchar * table);//ͨ��at����

	// argc���������ܵĲ���������argv[]��argc�����������е�0�������ǳ����ȫ������������û�����Ĳ���
	int main(int argc, char* argv[])
	{
		help();
		if (argc < 3)//�ж���������Ƿ�Ϸ����������С��3�����������������
		{
			cout << "Not enough parameters" << endl;
			return -1;
		}

		Mat I, J; //IΪ�������JΪ�������

		//int strcmp(const char *s1, const char *s2);
	   //����ֵ����s1��s2�ַ�����ȣ��򷵻��㣻��s1����s2���򷵻ش���������������򷵻�С���������
		if (argc == 4 && !strcmp(argv[3], "G"))
			I = imread(argv[1], IMREAD_GRAYSCALE);
		else
			I = imread(argv[1], IMREAD_COLOR);

		if (I.empty())
		{
			cout << "The image" << argv[1] << " could not be loaded." << endl;
			return -1;
		}

	
		int divideWith = 0;  // ������������ַ���ת��Ϊ���� -C++���
		stringstream s; //�����õ���stringstream
		s << argv[2]; //���������������Ƹ��ַ���
		s >> divideWith;//���ַ���ת��Ϊ����





		if (!s || !divideWith)
		{
			cout << "Invalid number entered for dividing. " << endl;
			return -1;
		}

		uchar table[256];//����һ����ɫ�ռ������ı����ʵ�������飬�����߲��Ҹ�ֵ
		for (int i = 0; i < 256; ++i)
			table[i] = (uchar)(divideWith * (i / divideWith));
		//! [dividewith]



		const int times = 100;//���峣������ֵ����ı������������ֵΪ100��Ŀ���Ǽ���ִ��100�ε�ƽ��ʱ��
		double t;//ƽ��ִ��ʱ��


		////////////////////C��������ʱ��////////////////////
		t = (double)getTickCount();

		for (int i = 0; i < times; ++i)
		{
			cv::Mat clone_i = I.clone();
			J = ScanImageAndReduceC(clone_i, table);
		}

		//1000 * �ܴ��� / һ�����ظ��Ĵ��� = ʱ��(ms)
		t = 1000 * ((double)getTickCount() - t) / getTickFrequency();
		t /= times;
		cout << "Time of reducing with the C operator [] (averaged for "
			<< times << " runs): " << t << " milliseconds." << endl;


		////////////////////Iterator��������ʱ��////////////////////
		t = (double)getTickCount();

		for (int i = 0; i < times; ++i)
		{
			cv::Mat clone_i = I.clone();
			J = ScanImageAndReduceIterator(clone_i, table);
		}


	
		//1000 * �ܴ��� / һ�����ظ��Ĵ��� = ʱ��(ms)
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



		//////////////////////////LUT����ʱ��//////////////////////////////////////
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















