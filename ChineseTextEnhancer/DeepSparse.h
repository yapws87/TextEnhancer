#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/opencv_modules.hpp>
#include "PiCommon.h"
//#include "C:\Users\yapws87\Documents\Visual Studio 2015\Projects\SwallowSpeed2\SwallowSpeed2\PiCommon.h"

#include "tbb/tbb.h"
#include "tbb/parallel_for.h"
#include "tbb/blocked_range.h"

class CDeepSparse
{
private:
	cv::Mat m_trainData;
	cv::Mat m_dictionary;
	cv::Mat m_sparseData;
	cv::Mat m_inputDataNorm;
	cv::Mat m_inputData8Bit;

	cv::Size m_featurePatchSize;
	int m_nDictionarySize = 0;
	int m_nStride = 1;
	int m_nSparseCount = 0;

	PiCommon m_picom;

	//void SetRandomDictionary(int nDicLength);

	// Random function creation
	std::vector<int> m_randIdx;
	int m_nRandMin = 0;
	int m_nRandMax = 0;
	void ClearRandomStack(void);
	void SetRandomRange(int nMin, int nMax);
	int GetRandomIndex_nonRepeat();
	unsigned long long llrand();

	//Matching pursuit
	void MatchingPursuit(int nMaxSparseCount);
	cv::Mat deconstruction(cv::Mat matSrc, int nMaxSparseCount, int &nSparseIndex);
	bool findBestAtom(cv::Mat matPatch, int &_nBestDic, float &_fBestDicCorr);

	//Reconstruction
	void reconstruct(cv::Mat matSrc, cv::Mat &matReconstructed, int nMaxSparseCount);

	// Distance
	float PatchDistance(const cv::Mat &trainPatch, const cv::Mat &dicPatch);
	float PatchCorrelation(const cv::Mat &trainPatch, const cv::Mat &dicPatch);
	float PatchCorrelationCoef(const cv::Mat &trainPatch, const cv::Mat &dicPatch);
	void K_SVD();

	void display_dictionary(int nWaitKey, bool bSave = false);

	cv::Mat getBestFeatures(int nFeature);
	// Analysis
	cv::Mat sort_sparsemat(cv::Mat matSparsemat);

	std::vector<int> m_sparse_combi_histo;

public:
	CDeepSparse();
	~CDeepSparse();

	void SetParam(const cv::Mat matSrc, int nDictionarySize, int nStride, int nFeaturePatchDim)
	{
		m_nDictionarySize = nDictionarySize;
		m_nStride = nStride;

		m_featurePatchSize = cv::Size(nFeaturePatchDim, nFeaturePatchDim);

		matSrc.convertTo(m_inputDataNorm, CV_32F, 1 / 255.f);
		matSrc.copyTo(m_inputData8Bit);
		
	}
	
	void ExtractTrainData();
	cv::Mat GetTrainData(void) { return m_trainData; }

	void SetRandomDictionary(int nDicLength);
	void SetFromPathDictionary(std::string dic_path, int nDicLength);
	void SetDictionary(cv::Mat matDic);
	cv::Mat GetDictionary(void) { return m_dictionary; }

	void Train(int nTrainLoop,int nMaxSparse);
	cv::Mat Predict(void);
	cv::Mat reconstruct_full( int nMaxSparse);
	bool loadDictionary(std::string dic_string);
	bool saveDictionary(std::string dic_string);
	bool saveHistogram(std::string histo_string);

};
