#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/opencv_modules.hpp>

class TextExtractor
{
public:
	TextExtractor();
	~TextExtractor();

	void blobDetection(cv::Mat matSrc);
	void slidingTextExtractor(const cv::Mat &matSrc, cv::Mat &matDst, cv::Size targetRect);
};

