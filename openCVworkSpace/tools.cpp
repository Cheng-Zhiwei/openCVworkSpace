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

	//C 库函数 int memcmp(const void *str1, const void *str2, size_t n)) 
	//把存储区 str1 和存储区 str2 的前 n 个字节进行比较。
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
		Mat_<Vec3b> _I = I;// 定义_I 并赋值为I

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



/////////////////////////伽马校正////////////////////////
Mat gammaResived(Mat& img, double& gamma) //这里没有用Mat& 作为函数的返回值，可能原因是经过LUT函数返回的变量的问题
{

	Mat lookUpTable(1, 256, CV_8U);
	uchar* p = lookUpTable.ptr();
	for (int i = 0; i < 256; ++i)
		p[i] = saturate_cast<uchar>(pow(i / 255.0, gamma) * 255.0);
	Mat res;
	LUT(img, lookUpTable, res);//这里

	return res;
}




//////////////////////////////////中值计算
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



////////////////////////中值滤波（3*3）/////////////////////
void MedianBlur(const Mat& src, Mat& img)
{
	img.create(src.size(), src.type());
	int const nChanels = src.channels();

	for (int j = 1; j < src.rows - 1; ++j)
	{
		const uchar* previous = src.ptr<uchar>(j - 1);
		const uchar* current = src.ptr<uchar>(j);
		const uchar* next = src.ptr<uchar>(j + 1);

		uchar* output = img.ptr<uchar>(j);//获取output的首地址

		for (int i = nChanels; i < nChanels*src.cols; ++i)//从第3个像素值开始，即边框像素不计算
		{

			Mat C_rank = (Mat_<uchar>(1, 9) << current[i], previous[i], next[i],
				current[i - nChanels], previous[i - nChanels], next[i - nChanels],
				current[i + nChanels], previous[i + nChanels], next[i + nChanels]);

			*output++ = getMedian(C_rank);
		}
	}

	img.row(0).setTo(Scalar(0));//0行
	img.row(img.rows - 1).setTo(Scalar(0));//rows-1行即最后一行
	img.col(0).setTo(Scalar(0));//0列
	img.col(img.cols - 1).setTo(Scalar(0));//cols-1列，即最后一列
}




////////////////////////////////扫描图像////////////////////

/*利用c指针*/
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
			p[j] = table[p[j]];//这里是对逐个像素的操作
		}
	}
	return I;
}


/*利用迭代器*/
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




/*利用randomAccess*/
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



////////////////////////膨胀与腐蚀/////////////////////
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

		uchar* output = img.ptr<uchar>(j);//获取output的首地址
		

		for (int i = nChanels; i < nChanels*src.cols; ++i)//从第3个像素值开始，即边框像素不计算
		{

			Mat C_rank = (Mat_<uchar>(1, 9) << current[i], previous[i], next[i],
				current[i - nChanels], previous[i - nChanels], next[i - nChanels],
				current[i + nChanels], previous[i + nChanels], next[i + nChanels]);

			minMaxIdx(C_rank, &minValue, &maxValue);//这里参数为地址

			switch (para)
			{
			case 1:
				*output++ = maxValue;


			case 0:
				*output++ = minValue;
			}
			
		}
	}

	img.row(0).setTo(Scalar(0));//0行
	img.row(img.rows - 1).setTo(Scalar(0));//rows-1行即最后一行
	img.col(0).setTo(Scalar(0));//0列
	img.col(img.cols - 1).setTo(Scalar(0));//cols-1列，即最后一列

}




////////////////////////阈值操作/////////////////////
Mat& thresholdOperation(Mat& I, const int para, const uchar thres)
{	
	
	CV_Assert(I.depth() == CV_8U);
	
	for (int i = 0; i < I.rows; ++i)
	{
		for (int j = 0; j < I.cols; ++j)
		{
			if (para == 0)//二进制阈值化
			{
				if (I.at<uchar>(i, j) > thres)
					I.at<uchar>(i, j) = 255;
				else 
					I.at<uchar>(i, j) = 0;
			}
				 
		
			else if (para == 1)//反二进制阈值化
			{
				{
					if (I.at<uchar>(i, j) > thres)
						I.at<uchar>(i, j) = 0;
					else
						I.at<uchar>(i, j) = 255;
				}
			}

			else if (para == 2)//截断阈值化
			{
				{
					if (I.at<uchar>(i, j) > thres)
						I.at<uchar>(i, j) = thres;
					else
						I.at<uchar>(i, j) = I.at<uchar>(i, j);
				}
			}

			else if (para == 3)//阈值化为0
			{
				{
					if (I.at<uchar>(i, j) > thres)
						I.at<uchar>(i, j) = I.at<uchar>(i, j);
					else
						I.at<uchar>(i, j) = 0;
				}
			}

			else if (para == 4)//反阈值化为0
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





////////////////////////sobel算子边缘检测////////////////////////////
void  sobelEdgeDetection(Mat& img_src, Mat& img_edge)
{

	/*将uint8（uchar）类型Mat图转换为数据内类型为32F（32位浮点数）
	转换时利用convertTo()函数尺度因子设置为1/255.0，imshow的float图像素被控制在[0-1]之间
	要转换8位灰度，尺度因子为255.0,这里的设置只是为了imshow可以显示图片，实际转换时不需要做尺度
	因子*/
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

	img_edge.convertTo(img_edge, CV_8U, 255.0);//转换为8位图

	//add(img_src, img_edge, img_dst);//原图像加入边缘图像得到img_dst的增强图像

}



/*一点Mat& function（）和void function()的体会
（1）函数中返回的结果，一定是与main函数中直接相关的变量；
（2）如果返回的是函数的变量则无法传递；
（3）例如我们要返回的Kernel是main中的变量，经过计算后，kernel中的数值虽然发生了变化但是他们指向的是同一内存地址
（4）不要试图返回函数方法中的变量*/
void getGaussKernel(int& size, double& sigma, Mat& Kernel)
{

	const float pi = 3.141592;
	int center = size / 2; //向下取整
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


