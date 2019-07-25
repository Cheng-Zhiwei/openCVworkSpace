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
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgcodecs.hpp>
#include <iostream> // ����cout endl ��ӡ
#include <string>
#include <opencv2/imgproc/imgproc_c.h> //����IplImage

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














