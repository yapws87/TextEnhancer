#include "DeepSparse.h"
#include <time.h>
#include <direct.h>





void CDeepSparse::SetRandomDictionary(int nDicLength)
{
	if (m_trainData.empty()) {
		return;
	}

	if (m_trainData.rows < nDicLength) {
		nDicLength = m_trainData.rows;
	}

	int nTotalElement = m_trainData.rows;


	int nIdx;
	SetRandomRange(0, nTotalElement);
	for (int rin = 0; rin < nDicLength; rin++)
	{
		// Get non-repeat random value
		nIdx = GetRandomIndex_nonRepeat();

		// Push back random dictionary data
		cv::Mat patchFlat = m_trainData.row(nIdx).reshape(1, 1);
		m_dictionary.push_back(patchFlat);
	}
}

void CDeepSparse::SetFromPathDictionary(std::string dic_path, int nDicLength)
{
	if (m_trainData.empty()) {
		return;
	}

	if (m_trainData.rows < nDicLength) {
		nDicLength = m_trainData.rows;
	}

	int nTotalElement = m_trainData.rows;

	loadDictionary(dic_path);

	if (m_dictionary.cols != m_featurePatchSize.area())
		m_dictionary = cv::Mat();
	
	if (m_dictionary.rows < nDicLength)
	{
		int nIdx;
		SetRandomRange(0, nTotalElement);
		for (int rin = m_dictionary.rows; rin < nDicLength; rin++)
		{
			// Get non-repeat random value
			nIdx = GetRandomIndex_nonRepeat();

			// Push back random dictionary data
			cv::Mat patchFlat = m_trainData.row(nIdx).reshape(1, 1);
			m_dictionary.push_back(patchFlat);
		}
	}
	else if(m_dictionary.rows > nDicLength) 
	{
		m_dictionary(cv::Rect(0, 0, m_dictionary.cols, nDicLength)).copyTo(m_dictionary);
	}
	
}

bool CDeepSparse::findBestAtom(cv::InputArray matPatch, int &_nBestDic, float &_fBestDicCorr )
{
	int nTotalDic = m_dictionary.rows;
	float fMaxCorr = 0;
	int nBestDic = -1;
	// Find the most similar patch
	for (int din = 0; din < nTotalDic; din++)
	{
		//float fCorr = PatchDistance(trainPatch,dicPatch)
		float fCorr = PatchCorrelation(matPatch, m_dictionary.row(din));

		//float fCorr = PatchCorrelationCoef(matResidue, m_dictionary.row(din));

		if (std::abs(fCorr) >= std::abs(fMaxCorr)) {
			fMaxCorr = fCorr;
			nBestDic = din;
		}
	}// for(int din

	 // Calculate Squared Magnitude of Dictionary vec
	//cv::Mat dicMag;
	//cv::multiply(m_dictionary.row(nBestDic), m_dictionary.row(nBestDic), dicMag);
	//cv::Scalar sSumDic = cv::sum(dicMag);
	//fMaxCorr = fMaxCorr / (float)sSumDic.val[0];
	// Set the sparse matrix

	// Update Coeff the residue
	_nBestDic = nBestDic;
	_fBestDicCorr = fMaxCorr;

	return true;
}


cv::Mat CDeepSparse::deconstruction(cv::Mat matSrc, int nMaxSparseCount, int &nSparseIndex)
{
	cv::Mat matResidue;
	
	int nSparseCount = 0;
	int nBestDic = -1;
	float fBestCorr = 0;

	// Inialize the residuw with original image
	matSrc.convertTo(matResidue,CV_32F);

	// Initialize sparse Matrix
	int nTotalWords = m_dictionary.rows;
	cv::Mat sparse_vec = cv::Mat::zeros(1, nTotalWords, CV_32FC1);

	std::vector<int> sparse_index;
	cv::Mat matBestDics;

	// Search loop
	float bestError = FLT_MAX;
	while (nSparseCount < nMaxSparseCount)
	{

		findBestAtom(matResidue, nBestDic, fBestCorr);
		
		
		//m_sparseData.at<float>(pin, nBestDic) += fBestCorr;
		
		// Reconstruct Local Patch
		cv::Mat local_reconstruct;
		cv::multiply(m_dictionary.row(nBestDic), fBestCorr, local_reconstruct);

		
		cv::Mat matLocalResidue;
		cv::subtract(matResidue, local_reconstruct, matLocalResidue);

		
		//Check Error
		cv::Mat matResidue_2;
		cv::multiply(matLocalResidue, matLocalResidue, matResidue_2);
		cv::Scalar sResiSum = cv::sum(matResidue_2);
		
		// Break Conditions
		double err = sqrt(sResiSum.val[0]);

		//if (err < bestError)
		{
			bestError = err;
			sparse_vec.at<float>(nBestDic) += fBestCorr;
			matLocalResidue.copyTo(matResidue);

			
			
			// Check for repeated feature
			bool bNewDic = true;
			for (int i = 0; i < sparse_index.size() && nSparseCount > 0; i++)
			{
				if (nBestDic  == sparse_index[i])
					bNewDic = false;
			}

			// if new add to vector
			if (bNewDic) {
				matBestDics.push_back(m_dictionary.row(nBestDic));
				sparse_index.push_back(nBestDic);
			}
			//cv::normalize(matLocalResidue, matResidue, 1.0, 0.0, cv::NORM_L2);
			cv::Mat m_residue = matResidue.reshape(1, m_featurePatchSize.width);
			cv::Mat m_reconst = local_reconstruct.reshape(1, m_featurePatchSize.width);

			// Min max norm will remove negative values, so avoid using it.
			//cv::normalize(matLocalResidue, matResidue, 0.0, 1.0, cv::NORM_MINMAX);
		}
		//else
		//	break;
		
		//Break if image is perfect
		if (err < 1e-10)
			break;
		else
			nSparseCount++;


	}//while ()

	matSrc.convertTo(matResidue, CV_32F);
	// Calculate inverse to obtain better weights
	cv::SVD svd_result(matBestDics.t());

	cv::Mat newWeights;
	svd_result.backSubst(matResidue.t(), newWeights);

	// REplace the values in sparse matrix
	for(int i = 0; i < sparse_index.size(); i++)
	{
		sparse_vec.at<float> (sparse_index[i]) = newWeights.at<float>(i);
	}

	//cv::Mat recons;
	//recons = matBestDics.t() * newWeights;


	// Calculate sparse index
	long lSections = 0;
	for (int i = 0; i < sparse_index.size(); i++)
	{
		nSparseIndex += sparse_index[i] + lSections;
		
		if (i == 0)
			lSections = 1;

		lSections = lSections * nTotalWords;
	}
	
	//cv::normalize(sparse_vec, sparse_vec, 1.f, 0.f, cv::NORM_L2);
	//cv::normalize(sparse_vec, sparse_vec, -1, 1, cv::NORM_MINMAX);
	return sparse_vec;
}

// Fill in Sparse Matrix
void CDeepSparse::MatchingPursuit(int nMaxSparseCount)
{
	int nTotalDic = m_dictionary.rows;
	int nTotalPatch = m_trainData.rows;

	m_sparseData = cv::Mat::zeros(nTotalPatch, nTotalDic, CV_32F);

	tbb::mutex tbb_mutex;

	// Loop through training patch
#ifndef _DEBUG 	
	tbb::parallel_for(tbb::blocked_range<size_t>(0, nTotalPatch), [&](const tbb::blocked_range<size_t> &r){
		for (int pin = (int)r.begin(); pin != (int)r.end(); pin++)
//#pragma omp parallel for
//			for (int pin = 0; pin < nTotalPatch; pin++)
#else
		for (int pin = 0; pin < nTotalPatch; pin++)
#endif
		{
			int nSparseIdx = 0;
			cv::Mat local_sparse = deconstruction(m_trainData.row(pin), nMaxSparseCount, nSparseIdx);
			
			local_sparse.copyTo(m_sparseData.row(pin));

		}

#ifndef _DEBUG	
	});
#endif


	// Reconstruct
	
}

float CDeepSparse::PatchDistance(const cv::Mat & trainPatch, const cv::Mat & dicPatch)
{
	cv::Mat train_row;
	cv::Mat dic_row;

	trainPatch.copyTo(train_row);
	dicPatch.copyTo(dic_row);

	train_row = trainPatch.reshape(1, 1);
	dic_row = dicPatch.reshape(1, 1);

	//exceptions
	if (trainPatch.cols != dicPatch.cols)
		return 0.0f;

	train_row.convertTo(train_row, CV_32F);
	dic_row.convertTo(dic_row, CV_32F);

	cv::Mat diff_row;
	cv::subtract(train_row, dic_row, diff_row);
	cv::multiply(diff_row, diff_row, diff_row);

	cv::Scalar  sSum = cv::sum(diff_row);

	return (float)sqrt(sSum.val[0]);
}

float CDeepSparse::PatchCorrelation(cv::InputArray trainPatch, cv::InputArray dicPatch)
{
	cv::Mat train_row;
	cv::Mat dic_row;

	//trainPatch.copyTo(train_row);
	//dicPatch.copyTo(dic_row);

	train_row = trainPatch.getMat().reshape(1, 1);
	dic_row = dicPatch.getMat().reshape(1, 1);

	//exceptions
	if (trainPatch.cols() != dicPatch.cols())
		return 0.0f;

	cv::Mat matCor;
	cv::matchTemplate(train_row, dic_row, matCor, cv::TM_CCORR);

	//float temp = (matCor.at<float>(0) / (float)dic_row.cols);
	float temp = (matCor.at<float>(0));// / (float)dic_row.cols);

	return temp;

	//cv::Mat matCor;
	//cv::multiply(train_row, dic_row, matCor);
	//cv::Scalar sSumCor = cv::sum(matCor);
	//return (float)sSumCor[0];


}

float CDeepSparse::PatchCorrelationCoef(const cv::Mat & trainPatch, const cv::Mat & dicPatch)
{
	cv::Mat train_row;
	cv::Mat dic_row;

	trainPatch.copyTo(train_row);
	dicPatch.copyTo(dic_row);

	train_row.convertTo(train_row, CV_64F);
	dic_row.convertTo(dic_row, CV_64F);

	//exceptions
	if (trainPatch.cols != dicPatch.cols)
		return 0.0f;

	cv::Scalar train_mean, train_std;
	cv::Scalar dic_mean, dic_std;

	cv::meanStdDev(train_row, train_mean, train_std);
	cv::meanStdDev(dic_row, dic_mean, dic_std);

	cv::subtract(train_row, train_mean, train_row);
	cv::subtract(dic_row, dic_mean, dic_row);

	cv::Mat matCor;
	cv::multiply(train_row, dic_row, matCor);

	cv::Scalar sSumCor = cv::sum(matCor);

	double dDivisor = train_std.val[0] * dic_std.val[0] * (trainPatch.cols - 1);

	return (float)(sSumCor.val[0] / dDivisor);
}

void CDeepSparse::K_SVD(cv::OutputArray _matDictionary)
{
	int nTotalTrain = m_trainData.rows;
	int nTotalDic = m_dictionary.rows;
	_matDictionary.create(m_dictionary.size(), m_dictionary.type());
	cv::Mat matDictionary = _matDictionary.getMat();
	//m_trainData
	//m_sparseData;
	tbb::parallel_for(tbb::blocked_range<size_t>(0, nTotalDic), [&](const tbb::blocked_range<size_t> &r)	{
		// For every atom of dictionary
		for (int din = (int)r.begin(); din != (int)r.end(); din++)
//#pragma omp parallel for
	for (int din = 0; din < nTotalDic; din++)
		{
			// Obtain related train data
			cv::Mat relatedTrain;
			cv::Mat relatedSparse;
		
			// Scan through every train patch tat is related to dictionary
			for (int tin = 0; tin < nTotalTrain; tin++)
			{
				cv::Mat matCleanTrain;
				if (m_sparseData.at<float>(tin, din) != 0) {
					relatedSparse.push_back(m_sparseData.at<float>(tin, din));
				
					m_trainData.row(tin).copyTo(matCleanTrain);
					
					// Scan through dictionary related to the patch
					for (int sin = 0; sin < nTotalDic; sin++)
					{
						// Skip if same
						if (sin == din)
							continue;

						float fSparse_weight = m_sparseData.at<float>(tin, sin);
						if (fSparse_weight != 0)
						{
							cv::Mat local_reconstruct;
							cv::multiply(m_dictionary.row(sin), fSparse_weight, local_reconstruct);

							cv::subtract(matCleanTrain, local_reconstruct, matCleanTrain);
						}
					}
					cv::divide(matCleanTrain, m_sparseData.at<float>(tin, din), matCleanTrain);
					cv::normalize(matCleanTrain, matCleanTrain, 1.f, 0.f, cv::NORM_L2);
					relatedTrain.push_back(matCleanTrain);
				}
			}

			if (relatedTrain.empty())
				continue;
			// Solve SVD
		
			cv::SVD svd_result = cv::SVD(relatedTrain);
			

			// Updates DIC
			cv::Mat newDic = svd_result.vt.row(0);

			for (int i = 0; i < newDic.cols; i++)
			{
				float ftemp = newDic.at<float>(i);
			/*	if (abs(ftemp) < 0.0001)
					newDic.at<float>(i) = 0.0f;*/
			}
			//cv::normalize(newDic, newDic, 0, 1, cv::NORM_MINMAX);
			cv::normalize(newDic, newDic, 1.f, 0.f, cv::NORM_L2);
			


			newDic.copyTo(matDictionary.row(din));
		}
	});
}

void CDeepSparse::ClearRandomStack()
{
	m_randIdx.clear();
	std::srand((unsigned int)time(NULL));
}

void CDeepSparse::SetRandomRange(int nMin, int nMax)
{
	if (nMax < nMin)
		return;

	m_randIdx.clear();

	m_nRandMin = nMin;
	m_nRandMax = nMax;
	m_randIdx.assign(nMax - nMin, -1);
}

unsigned long long CDeepSparse::llrand() {
	unsigned long long r = 0;

	for (int i = 0; i < 5; ++i) {
		r = (r << 15) | (rand() & 0x7FFF);
	}

	return r & 0xFFFFFFFFFFFFFFFFULL;
}

int CDeepSparse::GetRandomIndex_nonRepeat()
{
	int nTotalElement = (int)m_randIdx.size();
	bool bCheckRepeat = true;
	int nCurrentIdx = 0;
	int nRandomIdx = 0;

	while (bCheckRepeat)
	{
		// Create random value
		bCheckRepeat = false;
		nRandomIdx = llrand() % (nTotalElement);
		nRandomIdx += m_nRandMin;

		// Check for repeat
		for (int i = 0; i < nTotalElement; i++)
		{
			// Break if non-data found
			if (m_randIdx[i] < 0) {
				nCurrentIdx = i;
				break;
			}

			// Break if repeat found
			if (m_randIdx[i] == nRandomIdx) {
				bCheckRepeat = true;
				break;
			}
		}
	}

	m_randIdx[nCurrentIdx] = nRandomIdx;
	return nRandomIdx;
}

CDeepSparse::CDeepSparse()
{
	// Init random seed
	ClearRandomStack();
}

CDeepSparse::~CDeepSparse()
{
}

void CDeepSparse::ExtractTrainData()
{
	cv::Mat matPatches;
	cv::Mat normTrainImage;
	cv::Size patchSize = m_featurePatchSize;
	int nStride = m_nStride;
	m_inputDataNorm.copyTo(normTrainImage);

	int nCount = 0;
	tbb::mutex tbb_mutex;
	int nThresh = (m_featurePatchSize.area()) * 0.05;

	int nCountProgress = 0;
	// Loop through training patch
	float fSizeMultiplier[] = { 0.5, 1.0, 1.5 };
//#pragma omp parallel for
	for (int nSize = 0; nSize < 3; nSize++)
	{
		cv::resize(normTrainImage, normTrainImage, cv::Size(0, 0), fSizeMultiplier[nSize], fSizeMultiplier[nSize]);
		int nTotalY = normTrainImage.rows - patchSize.height;
		int nTotalX = normTrainImage.cols - patchSize.width;
	
		tbb::parallel_for(tbb::blocked_range<size_t>(0, nTotalY), [&](const tbb::blocked_range<size_t> &r){
			for (int y = (int)r.begin(); y != (int)r.end(); y++)
//#pragma omp parallel for
//		for (int y = 0; y <nTotalY; y+= nStride)
			{
				if (y % nStride != 0)
					continue;

				for (int x = 0; x < nTotalX; x += nStride)
				{
					cv::Rect matROI(x, y, patchSize.width, patchSize.height);


					cv::Mat matPatch;
					normTrainImage(matROI).copyTo(matPatch);


					if (cv::mean(matPatch)[0] > 0.1) {
						tbb_mutex.lock();
						//omp_lock.Lock();
						cv::normalize(matPatch, matPatch, 1.f, 0.f, cv::NORM_L2);
						//cv::normalize(matPatch, matPatch, 0, 1, cv::NORM_MINMAX);
						matPatches.push_back(matPatch.reshape(1, 1));
						nCount++;
						if (nCountProgress % 10 == 0)
						{
							std::cout << "\r" << "\t Extracting data 1\t : \t" << nCountProgress << " / " << (nTotalX * nTotalY) / (nStride *2);
						}
						nCountProgress++;
						tbb_mutex.unlock();
						//omp_lock.Unlock();
						
					}

				}

			}
		});
	}

	//std::cout << "\r" << "\t Extracting data\t : \t" << "100%" << " [" << nCount << "] " << "\t";
	std::cout << std::endl;
	//Randomize data
	m_trainData = cv::Mat();

	// Shuffle train data without replacement
	int nTotalPatches = matPatches.rows;
	for (int i = 0; i < matPatches.rows; i++)
	{
		int nShuffleIndex = llrand() % (nTotalPatches - i);

		if (i % 10 == 0)
		{
			std::cout << "\r" << "\t Extracting data 2\t : \t" << std::setprecision(3) << (i) / (float)(matPatches.rows) * 100 << "%";

		}
	
		// Swap locations
		cv::Mat matTemp;
		matPatches.row(i).copyTo(matTemp);
		matPatches.row(nShuffleIndex).copyTo(matPatches.row(i));
		matTemp.copyTo(matPatches.row(nShuffleIndex));
		
		//m_trainData.push_back(matPatches.row(nRandIdx));
		
		
	}
	// Transfer data
	matPatches.copyTo(m_trainData);
	std::cout << "\r" << "\t Extracting data\t : \t" << std::setprecision(3) << "100 %";
}

void CDeepSparse::initHisto()
{
	m_sparse_combi_histo.clearData();
}
void CDeepSparse::insertHisto(int id, cv::Mat matSample)
{
	m_sparse_combi_histo.insertData(id, matSample);
}

void CDeepSparse::reconstruct(cv::Mat _matSrc, cv::Mat &matReconstructed, int nMaxSparseCount)
{
	cv::Mat matSrc;
	_matSrc.copyTo(matSrc);

	initHisto();


	int nWidth = matSrc.cols;
	int nHeight = matSrc.rows;
	int nFeatureSize = std::sqrt(m_dictionary.cols);
	int nStride = 1;

	cv::Mat matDst = cv::Mat::zeros(matSrc.rows, matSrc.cols, CV_32F);
	cv::Mat matDivisor = cv::Mat::zeros(matSrc.rows, matSrc.cols, CV_32F);

	//matSrc.copyTo(matDst);


	// Make hole mask for inpainting
	cv::Mat matMask, matThreshHigh, matThreshLow, matDilate;
	cv::inRange(matSrc,  0.95, 1, matThreshHigh);
	cv::inRange(matSrc, 0.2, 1, matThreshLow);
	cv::dilate(matThreshLow, matDilate, cv::Mat(), cv::Point(-1, -1), 2);

	cv::subtract(matDilate, matThreshHigh, matMask);


	tbb::mutex tbb_mutex;
	cv::Mat matOnes = cv::Mat::ones(nFeatureSize, nFeatureSize, CV_32F);
	// Loop through training patch
	
	tbb::parallel_for(tbb::blocked_range<size_t>(0, nHeight - nFeatureSize), [&](const tbb::blocked_range<size_t> &r){
		for (int y = (int)r.begin(); y != (int)r.end(); y ++)
//#pragma omp parallel for
//		for (int y = 0; y < nHeight - nFeatureSize; y++)
		{
			if (y % nStride != 0)
				continue;

			for (int x = 0; x < nWidth - nFeatureSize; x++)
			{
				if (x % nStride != 0)
					continue;


				if (matDilate.at<uchar>(y, x) == 0)
					continue;

				// Extract Patchs
				cv::Rect roi(x, y, nFeatureSize, nFeatureSize);

				// Get Sparse Matrix
				cv::Mat matPatch;
				matSrc(roi).copyTo(matPatch);

				//if (matPatch.at<float>(matPatch.cols * matPatch.rows / 2) > 0.5)
				//	continue;
				//if (cv::sum(matPatch)[0] == 0)
				//	continue;

				int nSparseIdx = 0;
				cv::Mat matSparse = deconstruction(matPatch.reshape(1, 1), nMaxSparseCount, nSparseIdx);

				// Reconstruct
				cv::Mat matReconstruct = matSparse * m_dictionary;
				matReconstruct = matReconstruct.reshape(1, nFeatureSize);

				// Build Mask
				cv::Mat matMask;
				cv::multiply(matReconstruct, matReconstruct, matMask);
				matMask.convertTo(matMask, CV_8UC1, 255);
				//cv::normalize(matMask, matMask, 0, 255, cv::NORM_MINMAX, CV_8UC1);

				tbb_mutex.lock();
				//omp_lock.Lock(); 
				insertHisto(nSparseIdx, matPatch);
				
				matDst(roi) = matDst(roi) + matReconstruct;			
				cv::add(matDivisor(roi), matOnes,matDivisor(roi), matMask);
			
				//cv::multiply(matReconstruct.reshape(1, nFeatureSize), matDst(roi), matDst(roi));
				tbb_mutex.unlock();
				//omp_lock.Unlock(); 

			}
		}
	});

	



	cv::divide(matDst, matDivisor, matDst);
	
	cv::Mat matMaskInv;
	cv::bitwise_not(matMask, matMaskInv);
	matDst.setTo(0, matMaskInv);

	cv::Mat matSureMask, matUnsureMask;
	cv::subtract(matSrc, matDst, matSureMask);
	cv::inRange(matSureMask, 0.000001, 1, matSureMask);
	matSureMask.convertTo(matSureMask, CV_8UC1,255);

	cv::bitwise_not(matSureMask, matUnsureMask);

	cv::Mat matCombineMask;
	cv::multiply(matMask, matSureMask,matCombineMask);
	matReconstructed = cv::Mat::zeros(matSrc.size(), matSrc.type());
	cv::add(matDst, matReconstructed, matReconstructed, matUnsureMask);
	cv::add(matSrc , matReconstructed, matReconstructed, matSureMask);
	//cv::add(matSrc*0.5, matDst*0.5, matReconstructed, matCombineMask);

	matReconstructed.convertTo(matReconstructed, CV_8UC1, 255);
	cv::normalize(matReconstructed, matReconstructed, 0, 255, cv::NORM_MINMAX,CV_8UC1);
	
	//cv::normalize(matDst, matReconstructed, 255, 0, cv::NORM_L2, CV_8UC1);
	cv::Mat matReconstSmall;
	cv::resize(matReconstructed, matReconstSmall, cv::Size(0, 0), 1, 1);
	cv::imshow("reconstructed", matReconstSmall);
	cv::imshow("matMask_", matMask);
	cv::waitKey(1);
	matReconstSmall.copyTo(matReconstructed);

}

void CDeepSparse::SetDictionary(cv::Mat matDic)
{
	matDic.copyTo(m_dictionary);
}

void CDeepSparse::display_dictionary(int nWaitKey,bool bSave)
{
	int nTotalData = m_dictionary.rows;
	int nFeatureSize = std::sqrt(m_dictionary.cols);

	int nCanvasSize = std::ceil(std::sqrt(nTotalData));

	cv::Mat matCanvas = cv::Mat::zeros(nCanvasSize*nFeatureSize, nCanvasSize*nFeatureSize, CV_8UC1);

	cv::Mat feature_ranking = sort_sparsemat(m_sparseData);


	int x = 0, y = 0;
	int dx = 1, dy = -1;
	int nRankID;
	for (int nID = 0; nID < nTotalData; nID++)
	{
		// If Ranking is unavailable, print all
		if (!feature_ranking.empty())
			nRankID = feature_ranking.at<int>(nID);
		else
			nRankID = nID;

		// In case of overflow
		if (nRankID >= nTotalData)
			break;
		cv::Rect roiRect(x*nFeatureSize, y*nFeatureSize, nFeatureSize, nFeatureSize);
		cv::Mat matDicRow = m_dictionary.row(nRankID).reshape(1, nFeatureSize);
		cv::Mat matDicNorm;
		cv::normalize(matDicRow, matDicNorm, 0, 255, cv::NORM_MINMAX);
		matDicNorm.copyTo(matCanvas(roiRect));

		// Zig Zag Motion
		bool bInRange = false;
		while (!bInRange && nID < nTotalData -1)
		{
			x += dx;
			y += dy;

			if (x < 0) 
			{
				x = 0;
				dx *= -1; dy *= -1;
			}
			else if (y < 0)
			{
				y = 0;
				dx *= -1; dy *= -1;
			}
			else if (x >= nCanvasSize)
			{
				x = nCanvasSize - 1;
				y += 2;
				dx *= -1; dy *= -1;
			}
			else if (y >= nCanvasSize)
			{
				y = nCanvasSize - 1;
				x += 2;
				dx *= -1; dy *= -1;
			}
			
			// Repeat loop if location not within canvas
			if (x >= 0 && x < nCanvasSize
				&& y >= 0 && y < nCanvasSize)
				bInRange = true;
					
		}

	
	}
	cv::Mat matBigCanvas;
	cv::Mat matSparseBig;
	cv::resize(matCanvas, matBigCanvas, cv::Size(0, 0), 4, 4, cv::INTER_NEAREST);
	cv::imshow("Features", matBigCanvas);
	cv::waitKey(1);

	if (!m_sparseData.empty())
	{
		cv::resize(m_sparseData, matSparseBig, cv::Size(nTotalData,1024 ), 0, 0, cv::INTER_LINEAR);
		cv::normalize(matSparseBig, matSparseBig, 0,255, cv::NORM_MINMAX,CV_8UC1);

		cv::imshow("Sparse", matSparseBig);
		cv::waitKey(1);
	}

	static int m_nSaveCount = 0;
	if (bSave)
	{
		std::string filename = "Feature_" + std::to_string(m_nSaveCount) + ".jpg";
		std::string filename2 = "SparseMat_" + std::to_string(m_nSaveCount) + ".jpg";
		cv::imwrite(filename, matCanvas);
		if(!m_sparseData.empty())
			cv::imwrite(filename2, m_sparseData);
		m_nSaveCount++;
	}
	cv::waitKey(nWaitKey);
}

cv::Mat CDeepSparse::sort_sparsemat(cv::Mat matSparsemat)
{
	if (matSparsemat.empty())
		return cv::Mat();

	cv::Mat matRedeucesSparse;
	cv::Mat matRedeucesSparse_sorted;
	cv::Mat matAbsSparse;

	matAbsSparse = cv::abs(matSparsemat);

	cv::reduce(matAbsSparse, matRedeucesSparse, 0, CV_REDUCE_SUM);

	int nTotalValue = cv::sum(matRedeucesSparse)[0];
	cv::sortIdx(matRedeucesSparse, matRedeucesSparse_sorted, cv::SORT_EVERY_ROW | cv::SORT_DESCENDING);

	// Print best 10
	for (int i = 0; i < 10; i++)
	{
		int nIndex = matRedeucesSparse_sorted.at<int>(i);
		m_picom.printStdLog("[" + std::to_string(i) + "]\t Coverage Percentage : " + std::to_string(matRedeucesSparse.at<float>(nIndex) / nTotalValue * 100) + "%");
		//std::cout << "[" << i << "]" << "\t" << matRedeucesSparse.at<float>(nIndex) / nTotalValue *100 << "%" << std::endl;
	}

	return matRedeucesSparse_sorted;
}

cv::Mat CDeepSparse::getBestFeatures(int nFeature)
{
	cv::Mat feature_ranking = sort_sparsemat(m_sparseData);

	int nPatchSize = std::sqrt(m_dictionary.cols);
	cv::Mat matCanvas(nPatchSize * nFeature, nPatchSize, CV_8UC1);
	cv::Mat bestRow;
	cv::Mat bestPatch;
	for (int i = 0; i < nFeature; i++)
	{
		int nIdx = feature_ranking.at<int>(i);
		m_dictionary.row(nIdx).copyTo(bestRow);
		bestPatch = bestRow.reshape(1, nPatchSize);
		cv::Mat matPatch_Norm;
		cv::normalize(bestPatch, matPatch_Norm, 0,255, cv::NORM_MINMAX, CV_8UC1);
		matPatch_Norm.copyTo(matCanvas(cv::Rect(0, nPatchSize * i, nPatchSize, nPatchSize)));
	
	}

	cv::Mat matCanvas_big;
	cv::resize(matCanvas, matCanvas_big, cv::Size(0, 0), 3, 3);
	cv::imshow("Best Features", matCanvas_big);
	cv::waitKey(1);

	return matCanvas;
}

void CDeepSparse::Train(int nTotalLoop, int nMaxSparse)
{

	// Initial features
	int nWaitTime = 200;
	display_dictionary(nWaitTime);
	cv::Mat matReconstructed;
	for (int i = 0; i < nTotalLoop; i++)
	{
		m_picom.printStdLog("\n\t Start training\t : \t" + std::to_string(i) + " / " + std::to_string(nTotalLoop));

		double dTime = cv::getTickCount();
		
		MatchingPursuit(nMaxSparse);
		dTime = (cv::getTickCount() - dTime) / cv::getTickFrequency();
		m_picom.printStdLog("\t MatchingPursuit Time :  \t" + std::to_string(dTime) + " sec");
		
		dTime = cv::getTickCount();
		cv::Mat matNewDictionary;
		K_SVD(matNewDictionary);

		// Calculates error of dic
		cv::Mat dicDiff;
		cv::absdiff(matNewDictionary, m_dictionary, dicDiff);
		double dic_error = cv::mean(dicDiff)[0];

		// Update new dictionary
		matNewDictionary.copyTo(m_dictionary);

		dTime = (cv::getTickCount() - dTime) / cv::getTickFrequency();
		m_picom.printStdLog("\t K_SVD Time :  \t" + std::to_string(dTime) + " sec");
		
		//if (i % 25 == 0)
		//	display_dictionary(0);
		//else
		display_dictionary(200);

		int nFeatures = std::sqrt(m_dictionary.cols);
		int nSaveFeatureCount = nFeatures <= 25 ? nFeatures - 1 : 25;
		//cv::Mat best_feat = getBestFeatures(nSaveFeatureCount);
		//imwrite("best_feat" + std::to_string(i) + ".jpg", best_feat);

		cv::Rect teaserRect(200, 200, 400, 300);
		teaserRect = teaserRect & cv::Rect(0, 0, m_inputDataNorm.cols, m_inputDataNorm.rows);
		dTime = cv::getTickCount();
		reconstruct(m_testDataNorm(teaserRect), matReconstructed,nMaxSparse);
		dTime = (cv::getTickCount() - dTime) / cv::getTickFrequency();
		m_picom.printStdLog("\t Reconstruct Time :  \t" + std::to_string(dTime) + " sec");

		// Find Error
		cv::Mat matTest8U;
		cv::normalize(m_testDataNorm, matTest8U, 0, 255, cv::NORM_MINMAX);
		matTest8U.convertTo(matTest8U, CV_8UC1);
		cv::Mat matAbsDiff, matAbsDiffShow;
		cv::absdiff(matTest8U(teaserRect), matReconstructed, matAbsDiff);
		int nNonZero = cv::countNonZero(matTest8U(teaserRect));
		
		m_picom.printStdLog("\t Reconstruction Error :  \t" + std::to_string(cv::sum(matAbsDiff)[0] / (float)(nNonZero)) + " %");
		cv::normalize(matAbsDiff, matAbsDiffShow, 0, 255, cv::NORM_MINMAX, CV_8UC1);
		cv::imshow("Error Img", matAbsDiffShow);
		cv::waitKey(1);

		//if (i % 5 == 0)
		//	display_dictionary(200,true);

		saveDictionary("temp_dic.xml");

		// no Changes in dictionary
		if (dic_error < 0.0001)
		{
			std::cout << "Inter-dic Error : " << dic_error << std::endl;
			//break;
		}
		
		//std::cout << "\r" <<"\t Start training\t : \t" << i <<  " / " <<nTotalLoop  << std::endl;
	}

}

cv::Mat CDeepSparse::Predict()
{
	return cv::Mat();
}

cv::Mat CDeepSparse::reconstruct_full( int nMaxSparse)
{
	cv::Mat matReconstructed;

	

	reconstruct(m_inputDataNorm, matReconstructed, nMaxSparse);
	
	cv::Mat matReconst_8U;
	cv::normalize(matReconstructed, matReconst_8U, 0, 255, cv::NORM_MINMAX, CV_8UC1);
	cv::bitwise_not(matReconst_8U, matReconst_8U);
	//cv::imwrite(result_path, matReconst_8U);

	return matReconst_8U;
}

bool CDeepSparse::loadDictionary(std::string dic_string)
{
	cv::FileStorage fs(dic_string,cv::FileStorage::READ);
	if (!fs.isOpened())
	{
		std::cout << "XML file path not correct." << std::endl;
		return false;
	}

	fs["m_dictionary"] >> m_dictionary;


	return true;
}

bool CDeepSparse::saveDictionary(std::string dic_string)
{
	// Saves the dictionary
	cv::FileStorage fs(dic_string, cv::FileStorage::WRITE);
	if (fs.isOpened())
	{
		fs << "m_dictionary" << GetDictionary();
		std::cout << "Saved dictionary" << std::endl;
		fs.release();
		return true;
	}
		
	fs.release();
	return false;

}

bool CDeepSparse::saveHistogram(std::string folderPath, std::string histo_string)
{
	// Saves the Histogram
	m_sparse_combi_histo.sortData();
	

	cv::Mat matHisto = m_sparse_combi_histo.getData();
	std::string histoName = folderPath + "\\" +histo_string + "_D" + std::to_string(m_nDictionarySize) + "_F" + std::to_string(m_featurePatchSize.width) + "_hist.xml";
	cv::FileStorage fs(histoName, cv::FileStorage::WRITE);
	
	bool bPass_flag = false;

	//Saves histogram

	if (fs.isOpened())
	{
		std::cout << "Saving histogram" << std::endl;
		fs << "matHisto" << matHisto;
		std::cout << "Saved Histo" << std::endl;
		fs.release();

		bPass_flag = true;
	}


	std::vector<cv::Mat> sampleImgs = m_sparse_combi_histo.getSamplesImages();
	// Saves sample images
	std::string image_folder = folderPath + "\\sample_images" + histo_string;
	_mkdir(image_folder.c_str());

	std::cout << "Writing images" << std::endl;
	for(int i = 0; i < sampleImgs.size(); i++)
	{
		std::string imgName = image_folder + "\\img_" + std::to_string(matHisto.at<int>(i, 0)) + ".jpg";

		cv::Mat matSave;
		cv::normalize(sampleImgs[i], matSave, 0, 255, cv::NORM_MINMAX);
		matSave.convertTo(matSave, CV_8UC1);
		cv::resize(matSave, matSave, cv::Size(0, 0), 3, 3, cv::INTER_NEAREST);
		cv::imwrite(imgName, matSave);
	}
	std::cout << "Image writing complete" << std::endl;


	return bPass_flag;
	
}
