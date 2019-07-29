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