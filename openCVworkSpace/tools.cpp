#include "tools.h"

//////////////////////////////////////////////////////////////////////////
bool matIsEqual(const cv::Mat mat1, const cv::Mat mat2) {
	if (mat1.empty() && mat2.empty()) {
		return true;
	}
	if (mat1.cols != mat2.cols || mat1.rows != mat2.rows || mat1.dims != mat2.dims ||
		mat1.channels() != mat2.channels()) {
		return false;
	}
	if (mat1.size() != mat2.size() || mat1.channels() != mat2.channels() || mat1.type() != mat2.type()) {
		return false;
	}
	int nrOfElements1 = mat1.total()*mat1.elemSize();
	if (nrOfElements1 != mat2.total()*mat2.elemSize()) return false;

	//C �⺯�� int memcmp(const void *str1, const void *str2, size_t n)) 
	//�Ѵ洢�� str1 �ʹ洢�� str2 ��ǰ n ���ֽڽ��бȽϡ�
	bool lvRet = memcmp(mat1.data, mat2.data, nrOfElements1) == 0;
	return lvRet;
}




/////////////////////////////////////////////////////////////
Mat& contrastBrightness(Mat& I, double& a, double& b)
{

	int channels = I.channels();
	int nRows = I.rows;
	int nCols = I.cols;

	nCols = channels * nRows * nCols;

	uchar *p = I.ptr<uchar>(0);

	for (int i = 0; i < nCols; ++i)
	{
		p[i] = saturate_cast<uchar>(a * p[i] + b);
	}

	return I;
}




//////////////////////////////////////////////////////////////
Mat& contrastBrightnessByC(Mat& I, double& a, double& b)
{

	int channels = I.channels();
	int nRows = I.rows;
	int nCols = I.cols;

	nCols = channels * nRows * nCols;

	uchar *p = I.ptr<uchar>(0);

	for (int i = 0; i < nCols; ++i)
	{
		p[i] = saturate_cast<uchar>(a * p[i] + b);
	}

	return I;
}


//////////////////////////////////////////////////////////////////////
Mat& contrastBrightnessByRA(Mat& I, double& c, double& d)
{

	const int channels = I.channels();

	switch (channels)

	{
	case 1:
	{for (int i = 0; i < I.rows; ++i)
	{
		for (int j = 0; i < I.cols; ++j)
		{
			I.at<uchar>(i, j) = saturate_cast<uchar>(c * I.at<uchar>(i, j) + d);
		}
		break;
	}
	}

	case 3:
	{
		Mat_<Vec3b> _I = I;// ����_I ����ֵΪI

		for (int i = 0; i < I.rows; ++i)
		{
			for (int j = 0; j < I.cols; ++j)
			{
				_I(i, j)[0] = saturate_cast<uchar>(c*(_I.at<Vec3b>(i, j)[0]) + d);
				_I(i, j)[1] = saturate_cast<uchar>(c*(_I.at<Vec3b>(i, j)[1]) + d);
				_I(i, j)[2] = saturate_cast<uchar>(c*(_I.at<Vec3b>(i, j)[2]) + d);
			}

			I = _I;
			break;
		}
	}
	}

	return I;
}




Mat& contrastBrightnessByOfficial(Mat& I, double& c, double& d)
{
	const int channels = I.channels();

	Mat I_clone = I.clone();

	for (int y = 0; y < I.rows; y++)
	{
		for (int x = 0; x < I.cols; x++)
		{
			for (int c = 0; c < 3; c++)
			{
				I_clone.at<Vec3b>(y, x)[c] = saturate_cast<uchar>(c*(I.at<Vec3b>(y, x)[c]) + d);
			}
		}
	}

	return I_clone;
}



/////////////////////////٤��У��////////////////////////
Mat gammaResived(Mat& img, double& gamma) //����û����Mat& ��Ϊ�����ķ���ֵ������ԭ���Ǿ���LUT�������صı���������
{

	Mat lookUpTable(1, 256, CV_8U);
	uchar* p = lookUpTable.ptr();
	for (int i = 0; i < 256; ++i)
		p[i] = saturate_cast<uchar>(pow(i / 255.0, gamma) * 255.0);
	Mat res;
	LUT(img, lookUpTable, res);//����

	return res;
}




//////////////////////////////////��ֵ����
uchar& getMedian(Mat& zone)
{
	uchar median;
	uchar *p;
	cv::sort(zone, zone, SORT_EVERY_ROW + SORT_ASCENDING);

	int len = zone.cols;
	int index = len / 2 + 1;

	p = zone.ptr<uchar>(0);

	median = p[index];

	return median;
}



////////////////////////��ֵ�˲���3*3��/////////////////////
void MedianBlur(const Mat& src, Mat& img)
{
	img.create(src.size(), src.type());
	int const nChanels = src.channels();

	for (int j = 1; j < src.rows - 1; ++j)
	{
		const uchar* previous = src.ptr<uchar>(j - 1);
		const uchar* current = src.ptr<uchar>(j);
		const uchar* next = src.ptr<uchar>(j + 1);

		uchar* output = img.ptr<uchar>(j);//��ȡoutput���׵�ַ

		for (int i = nChanels; i < nChanels*src.cols; ++i)//�ӵ�3������ֵ��ʼ�����߿����ز�����
		{

			Mat C_rank = (Mat_<uchar>(1, 9) << current[i], previous[i], next[i],
				current[i - nChanels], previous[i - nChanels], next[i - nChanels],
				current[i + nChanels], previous[i + nChanels], next[i + nChanels]);

			*output++ = getMedian(C_rank);
		}
	}

	img.row(0).setTo(Scalar(0));//0��
	img.row(img.rows - 1).setTo(Scalar(0));//rows-1�м����һ��
	img.col(0).setTo(Scalar(0));//0��
	img.col(img.cols - 1).setTo(Scalar(0));//cols-1�У������һ��
}




////////////////////////////////ɨ��ͼ��////////////////////

/*����cָ��*/
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
			p[j] = table[p[j]];//�����Ƕ�������صĲ���
		}
	}
	return I;
}


/*���õ�����*/
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




/*����randomAccess*/
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



////////////////////////�����븯ʴ/////////////////////
void ErodingAndDilating(const Mat& src, Mat& img, int& para)
{
	img.create(src.size(), src.type());

	int const nChanels = src.channels();

	double  maxValue, minValue;
 
	for (int j = 1; j < src.rows - 1; ++j)
	{
		const uchar* previous = src.ptr<uchar>(j - 1);
		const uchar* current = src.ptr<uchar>(j);
		const uchar* next = src.ptr<uchar>(j + 1);

		uchar* output = img.ptr<uchar>(j);//��ȡoutput���׵�ַ
		

		for (int i = nChanels; i < nChanels*src.cols; ++i)//�ӵ�3������ֵ��ʼ�����߿����ز�����
		{

			Mat C_rank = (Mat_<uchar>(1, 9) << current[i], previous[i], next[i],
				current[i - nChanels], previous[i - nChanels], next[i - nChanels],
				current[i + nChanels], previous[i + nChanels], next[i + nChanels]);

			minMaxIdx(C_rank, &minValue, &maxValue);//�������Ϊ��ַ

			switch (para)
			{
			case 1:
				*output++ = maxValue;


			case 0:
				*output++ = minValue;
			}
			
		}
	}

	img.row(0).setTo(Scalar(0));//0��
	img.row(img.rows - 1).setTo(Scalar(0));//rows-1�м����һ��
	img.col(0).setTo(Scalar(0));//0��
	img.col(img.cols - 1).setTo(Scalar(0));//cols-1�У������һ��

}




////////////////////////��ֵ����/////////////////////
Mat& thresholdOperation(Mat& I, const int para, const uchar thres)
{	
	
	CV_Assert(I.depth() == CV_8U);
	
	for (int i = 0; i < I.rows; ++i)
	{
		for (int j = 0; j < I.cols; ++j)
		{
			if (para == 0)//��������ֵ��
			{
				if (I.at<uchar>(i, j) > thres)
					I.at<uchar>(i, j) = 255;
				else 
					I.at<uchar>(i, j) = 0;
			}
				 
		
			else if (para == 1)//����������ֵ��
			{
				{
					if (I.at<uchar>(i, j) > thres)
						I.at<uchar>(i, j) = 0;
					else
						I.at<uchar>(i, j) = 255;
				}
			}

			else if (para == 2)//�ض���ֵ��
			{
				{
					if (I.at<uchar>(i, j) > thres)
						I.at<uchar>(i, j) = thres;
					else
						I.at<uchar>(i, j) = I.at<uchar>(i, j);
				}
			}

			else if (para == 3)//��ֵ��Ϊ0
			{
				{
					if (I.at<uchar>(i, j) > thres)
						I.at<uchar>(i, j) = I.at<uchar>(i, j);
					else
						I.at<uchar>(i, j) = 0;
				}
			}

			else if (para == 4)//����ֵ��Ϊ0
			{
				{
					if (I.at<uchar>(i, j) > thres)
						I.at<uchar>(i, j) = 0;
					else
						I.at<uchar>(i, j) = I.at<uchar>(i, j);
				}
			}

		}
	}	
		
	return I;
}





////////////////////////sobel���ӱ�Ե���////////////////////////////
void  sobelEdgeDetection(Mat& img_src, Mat& img_edge)
{

	/*��uint8��uchar������Matͼת��Ϊ����������Ϊ32F��32λ��������
	ת��ʱ����convertTo()�����߶���������Ϊ1/255.0��imshow��floatͼ���ر�������[0-1]֮��
	Ҫת��8λ�Ҷȣ��߶�����Ϊ255.0,���������ֻ��Ϊ��imshow������ʾͼƬ��ʵ��ת��ʱ����Ҫ���߶�
	����*/
	img_src.convertTo(img_src, CV_32F, 1 / 255.0);

	int rows = img_src.rows;
	int cols = img_src.cols;
	
	Mat_<float> img_x = Mat::zeros(rows, cols, CV_32F);
	Mat_<float> img_y = Mat::zeros(rows, cols, CV_32F);
	Mat_<float> img = img_src;

	for (int i = 0; i < rows - 1; ++i)
	{
		for (int j = 0; j < cols - 1; ++j)
		{
			img_x(i, j) = img(i + 1, j) - img(i, j);
			img_y(i, j) = img(i, j + 1) - img(i, j);
		}
	}

	add(abs(img_x), abs(img_y), img_edge);

	img_edge.convertTo(img_edge, CV_8U, 255.0);//ת��Ϊ8λͼ

	//add(img_src, img_edge, img_dst);//ԭͼ������Եͼ��õ�img_dst����ǿͼ��

}



/*һ��Mat& function������void function()�����
��1�������з��صĽ����һ������main������ֱ����صı�����
��2��������ص��Ǻ����ı������޷����ݣ�
��3����������Ҫ���ص�Kernel��main�еı��������������kernel�е���ֵ��Ȼ�����˱仯��������ָ�����ͬһ�ڴ��ַ
��4����Ҫ��ͼ���غ��������еı���*/
void getGaussKernel(int& size, double& sigma, Mat& Kernel)
{

	const float pi = 3.141592;
	int center = size / 2; //����ȡ��
	float sum = 0;

	Mat_<float> kernel = Mat::ones(size, size, CV_32F);

	for (int i = 0; i < size; i++)
		for (int j = 0; j < size; j++)
			sum += kernel(i, j) = exp(-((i - center) * (i - center) + (j - center) * (j - center)) / 2.0f * sigma * sigma) 
			/ (2.0f * pi * sigma * sigma);

	for (int m = 0; m < size; m++)
		for (int n = 0; n < size; n++)
			kernel(m, n) /= sum;
	
	Kernel = kernel;

}


