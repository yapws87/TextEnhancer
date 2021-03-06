#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/opencv_modules.hpp>
#include "PiCommon.h"
//#include "C:\Users\yapws87\Documents\Visual Studio 2015\Projects\SwallowSpeed2\SwallowSpeed2\PiCommon.h"

#include "tbb/tbb.h"
#include "tbb/parallel_for.h"
#include "tbb/blocked_range.h"

#ifdef _OPENMP
# include <omp.h>
#endif

struct SparseHisto {
	cv::Mat data;
	std::vector<cv::Mat> sample_imgs;
	SparseHisto() {

	}
	SparseHisto(int id, cv::Mat &matSample)
	{
		sample_imgs.clear();
		cv::Mat localData = (cv::Mat_<int>(1, 2) << id, 0);


		if (data.empty()) {
			localData.copyTo(data);
		}
		else {
			data.push_back(localData);
		}
		sample_imgs.push_back(matSample);

	}
	~SparseHisto() {}

	void insertData(int id, cv::Mat &matSample)
	{
		bool bFound = false;
		for (int i = 0; i < data.rows; i++)
		{
			if (id == data.at<int>(i, 0))
			{
				data.at<int>(i, 1) = data.at<int>(i, 1) + 1;
				bFound = true;
				cv::add(sample_imgs[i] * 0.5, matSample*(0.5), sample_imgs[i]);
				break;
			}
		}

		if (!bFound) {
			cv::Mat tempMat = (cv::Mat_<int>(1, 2) << (int)id, 0);
			data.push_back(tempMat);
			sample_imgs.push_back(matSample);
		}

	}

	cv::Mat getData() { return data; }
	std::vector<cv::Mat> getSamplesImages() { return sample_imgs; }
	
	void clearData() {
		if(!data.empty())
			data.release();
	}

	void sortData() {
		// Sort to get the sorted index
		cv::Mat sorted_idx;
		cv::sortIdx(data.col(1), sorted_idx, cv::SORT_EVERY_COLUMN + cv::SORT_DESCENDING);

		cv::Mat sortedData(data.size(),data.type());
		std::vector<cv::Mat> sortedSamples;
		sortedSamples.assign(sample_imgs.size(), cv::Mat());

		// Real moving of objects
		for (int i = 0; i < sorted_idx.total(); i++)
		{
			int idx = sorted_idx.at<int>(i);
			data.row(idx).copyTo(sortedData.row(i));
			sample_imgs[idx].copyTo(sortedSamples[i]);
		}		

	}

};


#ifdef _OPENMP
struct MutexType
{
	MutexType() { omp_init_lock(&omplock); }
	~MutexType() { omp_destroy_lock(&omplock); }
	void Lock() { omp_set_lock(&omplock); }
	void Unlock() { omp_unset_lock(&omplock); }

	MutexType(const MutexType&) { omp_init_lock(&omplock); }
	MutexType& operator= (const MutexType&) { return *this; }
public:
	omp_lock_t omplock;
};
#else
/* A dummy mutex that doesn't actually exclude anything,
* but as there is no parallelism either, no worries. */
struct MutexType
{
	void Lock() {}
	void Unlock() {}
};
#endif

class CDeepSparse
{
private:
	cv::Mat m_trainData;
	cv::Mat m_dictionary;
	cv::Mat m_sparseData;
	cv::Mat m_inputDataNorm;
	cv::Mat m_testDataNorm;
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
	bool findBestAtom(cv::InputArray matPatch, int &_nBestDic, float &_fBestDicCorr);

	//Reconstruction
	void reconstruct(cv::Mat matSrc, cv::Mat &matReconstructed, int nMaxSparseCount);

	// Distance
	float PatchDistance(const cv::Mat &trainPatch, const cv::Mat &dicPatch);
	float PatchCorrelation(cv::InputArray trainPatch, cv::InputArray dicPatch);
	float PatchCorrelationCoef(const cv::Mat &trainPatch, const cv::Mat &dicPatch);
	void K_SVD(cv::OutputArray matDictionary);

	void display_dictionary(int nWaitKey, bool bSave = false);

	cv::Mat getBestFeatures(int nFeature);
	// Analysis
	cv::Mat sort_sparsemat(cv::Mat matSparsemat);

	SparseHisto m_sparse_combi_histo;

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
	void SetTestImage(const cv::Mat matSrc) {
		matSrc.convertTo(m_testDataNorm, CV_32F, 1 / 255.f);
	}
	
	void ExtractTrainData();
	void initHisto();
	void insertHisto(int id, cv::Mat matSample);
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
	bool saveHistogram(std::string folderPath, std::string histo_string);



	MutexType omp_lock;


};



/* An exception-safe scoped lock-keeper. */
//struct ScopedLock
//{
//	explicit ScopedLock(MutexType& m) : mut(m), locked(true) { mut.Lock(); }
//	~ScopedLock() { Unlock(); }
//	void Unlock() { if (!locked) return; locked = false; mut.Unlock(); }
//	void LockAgain() { if (locked) return; mut.Lock(); locked = true; }
//private:
//	MutexType& mut;
//	bool locked;
//private: // prevent copying the scoped lock.
//	void operator=(const ScopedLock&);
//	ScopedLock(const ScopedLock&);
//};
