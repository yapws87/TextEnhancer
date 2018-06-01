#include "TextExtractor.h"



TextExtractor::TextExtractor()
{
}


TextExtractor::~TextExtractor()
{
}


void TextExtractor::slidingTextExtractor(const cv::Mat &matSrc, cv::Mat &matDst, cv::Size targetSize)
{
	int nWidth = matSrc.cols;
	int nHeight = matSrc.rows;
	int nStride = 3;

	// Preprocess 
	cv::Size gaussianSize = targetSize;

	// Avoid even number for gaussian mask
	if (targetSize.width % 2 == 0)
		gaussianSize.width++;
	if (targetSize.height % 2 == 0)
		gaussianSize.height++;
	
	cv::Mat matBlurred;
	
	cv::GaussianBlur(matSrc, matBlurred, gaussianSize, -1);


	cv::Rect targetRect;

	cv::Mat rectTemplate = cv::Mat::ones(targetSize.height, targetSize.width, CV_8UC1);
	cv::Mat rectMask = cv::Mat::zeros(targetSize.height, targetSize.width, CV_8UC1);
	matDst = cv::Mat::zeros(matSrc.rows, matSrc.cols, CV_8UC1);

	rectMask.col(0).setTo(255);
	rectMask.row(0).setTo(255);
	rectMask.col(targetSize.width - 1).setTo(255);
	rectMask.row(targetSize.height - 1).setTo(255);

	for (int y = 0; y < nHeight - targetSize.height; y++)
	{
		for (int x = 0; x < nWidth - targetSize.width; x++)
		{
			int y_off = y + targetSize.height / 2;
			int x_off = x + targetSize.width / 2;
			targetRect = cv::Rect(x, y, targetSize.width, targetSize.height);
			cv::Mat matRoi = matBlurred(targetRect);

			cv::Scalar summed_roi = cv::sum(matRoi);
			if (summed_roi[0] == 0)
				continue;
			
			cv::Mat matDiff = cv::Mat::zeros(targetSize.height, targetSize.width, CV_32F);
			int nCenter = matBlurred.at<uchar>(y_off, x_off) - 2;
			
			cv::subtract(matRoi, rectTemplate * nCenter, matDiff, rectMask,CV_32F);
			
			cv::Mat matThresh;
			cv::threshold(matDiff, matThresh, 0, 1, cv::THRESH_BINARY);

			cv::Scalar summed =  cv::sum(matThresh);

			if (summed[0] == 0)
				matDst.at<uchar>(y_off, x_off) += 1;
		
		}
	}
}
void TextExtractor::blobDetection(cv::Mat matSrc)
{
	cv::Mat matSmallSrc;
	cv::Mat matBinary;
	float fScale = 0.25f;
	cv::resize(matSrc, matSmallSrc, cv::Size(0,0), fScale, fScale);
	

	cv::Mat matDst[4],matFinalDst;
	//slidingTextExtractor(matSmallSrc, matDst, cv::Size(matSmallSrc.cols / 12, matSmallSrc.rows / 22));
	slidingTextExtractor(matSmallSrc, matDst[0], cv::Size(10, 10));
	slidingTextExtractor(matSmallSrc, matDst[1], cv::Size(15, 15));
	slidingTextExtractor(matSmallSrc, matDst[2], cv::Size(20, 20));
	slidingTextExtractor(matSmallSrc, matDst[3], cv::Size(25, 25));

	cv::add(matDst[0], matDst[1], matFinalDst);
	cv::add(matFinalDst, matDst[2], matFinalDst);
	cv::add(matFinalDst, matDst[3], matFinalDst);
	//cv::morphologyEx(matSmallSrc, matClose, cv::MORPH_CLOSE, cv::Mat(), cv::Point(-1, -1), 2);
	
	std::vector<std::vector<cv::Point>> matContours;
	cv::findContours(matFinalDst, matContours, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);
	
	cv::Mat matDisplay, matDisplay2;
	
	cv::cvtColor(matSrc, matDisplay, cv::COLOR_GRAY2BGR);

	cv::normalize(matFinalDst, matFinalDst, 0, 255, cv::NORM_MINMAX);
	cv::cvtColor(matFinalDst, matDisplay2, cv::COLOR_GRAY2BGR);
	cv::resize(matDisplay2, matDisplay2, cv::Size(0, 0), 1/ fScale, 1/ fScale);
	
	std::vector<cv::Rect> blobRects;
	// Draw a square on all the blobs
	for (int i = 0; i < matContours.size(); i++)
	{
		cv::Point minPoint, maxPoint;
		cv::Rect blobRect;
		// For each contours sort for x
		std::sort(matContours[i].begin(), matContours[i].end(), [](cv::Point p1, cv::Point p2) {
			return p1.x < p2.x;
		});
		minPoint.x = matContours[i][0].x;
		maxPoint.x = matContours[i][matContours[i].size() -1].x;

		// For each contours sort for y
		std::sort(matContours[i].begin(), matContours[i].end(), [](cv::Point p1, cv::Point p2) {
			return p1.y < p2.y;
		});
		minPoint.y = matContours[i][0].y;
		maxPoint.y = matContours[i][matContours[i].size() - 1].y;

		blobRect = cv::Rect(minPoint * 1 / fScale, maxPoint * 1 / fScale);
		
		// Remove small regions
		if (blobRect.area() < 1)
			continue;

		// Remove overly big text
		//if (blobRect.width > matSrc.cols / 10)
		//	continue;
		
		// Remove abnormaly sized text
		//float fRatio = blobRect.width > blobRect.height ? blobRect.width / blobRect.height : blobRect.height / blobRect.width;
		//if (fRatio > 3)
		//	continue;


		cv::Size size_up(15, 15);
		blobRect += size_up;
		blobRect -= cv::Point(size_up.width / 2, size_up.height / 2);

		// Check for intersection, and keep the larger ones
		bool bIntersect = false;
		for (int n = 0; n < blobRects.size(); n++)
		{
			if ((blobRects[n] & blobRect).area() > 0)
			{
				bIntersect = true;
				cv::Rect combinedRect = blobRects[n] | blobRect;
				blobRects[n] = combinedRect;
				blobRect = combinedRect;
						
			}
		}
		if (bIntersect)
			continue;
			
		blobRects.push_back(blobRect);
		
	}


	// Draw all out
	for (int n = 0; n < blobRects.size(); n++)
	{
		cv::rectangle(matDisplay, blobRects[n], cv::Scalar(0, 0, 255), 2);
		cv::rectangle(matDisplay2, blobRects[n], cv::Scalar(0, 0, 255), 2);
		
	}
}