#include <stdio.h>
#include <iostream>
#include <vector>
#include <string>

#include <opencv2\opencv.hpp>
#include <opencv2\opencv_modules.hpp>

#include "DeepSparse.h"
#include "TextExtractor.h"


int m_nDictionarySize = 0;
CDeepSparse sc;
TextExtractor tx;
int m_nStride = 0;
int m_nTrainLoop = 1;
int m_nMaxSparse = 1;
float m_fSizeUp = 3;
int m_nFeaturePatch = 12;

cv::Mat image_enhancer(cv::Mat matSrc)
{
	cv::Mat matInverse;
	cv::Mat matSrc_binary;
	cv::Mat matSrc_binary_2x;

	double dSizeUp = 8;
	int nSmoothLevel = 2;

	

	cv::bitwise_not(matSrc, matInverse);
	
	cv::threshold(matInverse, matSrc_binary, 25, 255, CV_THRESH_BINARY);
	//cv::medianBlur(matSrc_binary, matSrc_binary, 3);

	cv::resize(matSrc_binary, matSrc_binary_2x, cv::Size(0, 0), dSizeUp, dSizeUp, cv::INTER_LINEAR);

	

	cv::Mat matCross =  cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
	cv::medianBlur(matSrc_binary_2x, matSrc_binary_2x, 3);
	for (int i = 0; i < nSmoothLevel; i++)
	{
		cv::dilate(matSrc_binary_2x, matSrc_binary_2x, matCross, cv::Point(-1, -1), 2 );
		cv::erode(matSrc_binary_2x, matSrc_binary_2x, matCross, cv::Point(-1, -1), 2 );
	}
	cv::GaussianBlur(matSrc_binary_2x, matSrc_binary_2x, cv::Size(5, 5), -1);
	cv::resize(matSrc_binary_2x, matSrc_binary, cv::Size(0, 0), 1 / m_fSizeUp, 1 / m_fSizeUp, cv::INTER_LINEAR);
	cv::bitwise_not(matSrc_binary, matSrc_binary);

	return matSrc_binary;
}

void initScreen()
{
	std::cout << "***********************************\n";
	std::cout << "       Sparse Learning\n";
	std::cout << "***********************************\n";
	std::cout << std::endl;

}


//Main Console
void main(int argc, const char** argv)
{
	int nArgIdx = 0;
	std::string dst_folder = argv[++nArgIdx];
	std::string image_path = argv[++nArgIdx];
	std::string dic_path = argv[++nArgIdx];
	std::string result_path = argv[++nArgIdx];

	bool bTrain = std::string(argv[++nArgIdx]).compare("TRUE") ? false : true;
	m_nDictionarySize = atoi(argv[++nArgIdx]);
	m_nStride = atoi(argv[++nArgIdx]);
	m_nTrainLoop = atoi(argv[++nArgIdx]);
	m_nMaxSparse = atoi(argv[++nArgIdx]);
	m_nFeaturePatch = atoi(argv[++nArgIdx]);
	m_fSizeUp = atof(argv[++nArgIdx]);


	// Show parameters
	std::cout << "********* PARAMETERS ********" << std::endl;
	std::cout << "\t" << "dst_folder" << "\t: "<< dst_folder << std::endl;
	std::cout << "\t" << "image_path" << "\t: " << image_path << std::endl;
	std::cout << "\t" << "dic_path" << "\t: " << dic_path << std::endl;
	std::cout << "\t" << "result_path" << "\t: " << result_path << std::endl;
	std::cout << "\t" << "bTrain" << "\t\t: " << bTrain << std::endl;
	std::cout << "\t" << "m_nDictionarySize" << "\t: " << m_nDictionarySize << std::endl;
	std::cout << "\t" << "m_nStride" << "\t: " << m_nStride << std::endl;
	std::cout << "\t" << "m_nTrainLoop" << "\t: " << m_nTrainLoop << std::endl;
	std::cout << "\t" << "m_nMaxSparse" << "\t: " << m_nMaxSparse << std::endl;
	std::cout << "\t" << "m_nFeaturePatch" << "\t: " << m_nFeaturePatch << std::endl;
	std::cout << "\t" << "m_fSizeUp" << "\t: " << m_fSizeUp << std::endl;
	std::cout << std::endl;


	initScreen();

	std::cout << "\t Load Image\t : \t" << image_path << "\t";
	cv::Mat matSrc = cv::imread(image_path,0);
	std::cout << "\t[DONE]" << std::endl;
	

	// Load Data
	cv::Mat matInverse;
	cv::bitwise_not(matSrc, matInverse);

	//tx.blobDetection(matInverse);


	std::cout << "\t Extracting data\t : \t";
	cv::Mat matInverseLarge;
	cv::resize(matInverse, matInverseLarge, cv::Size(0, 0), m_fSizeUp, m_fSizeUp);

	
	sc.SetParam(matInverseLarge, m_nDictionarySize, m_nStride, m_nFeaturePatch);


	if (bTrain)	{

		sc.ExtractTrainData();
		std::cout << "[DONE]" << std::endl;

		std::cout << "\t Set random data\t : \t";
		sc.SetRandomDictionary(m_nDictionarySize);
		std::cout << "[DONE]" << std::endl;

		std::cout << "\t Start Training\t : ";
		sc.Train(m_nTrainLoop, m_nMaxSparse);
		std::cout << "[DONE]";

		std::cout << "\t Save dictionary\t : " << dic_path << "\t";
		if (sc.saveDictionary(dic_path)) {
			std::cout << "[DONE]";
		}
		std::cout << std::endl;
	}
	else{
		std::cout << "\t Load dictionary\t : " << dic_path << "\t";
		if (sc.loadDictionary(dic_path)){
			std::cout << "[DONE]";
		}
		std::cout << std::endl;
	}

	std::cout << "\t Reconstructing..\t : \t";
	cv::Mat matResult = sc.reconstruct_full(m_nMaxSparse);
	cv::Mat matResultOriSize;
	cv::resize(matResult, matResultOriSize, cv::Size(0, 0), 1 / m_fSizeUp, 1 / m_fSizeUp);
	//cv::threshold(matResultOriSize, matResultOriSize, 200, 255, cv::THRESH_BINARY);
	cv::imwrite(result_path, matResultOriSize);
	std::cout << "[DONE]";
	std::cout << std::endl;
	
	std::cout << "[ALL DONE]" << std::endl;
	//std::cout << "\t Press any key to end\t : \t" << std::endl;
	//getchar();
	return;
}