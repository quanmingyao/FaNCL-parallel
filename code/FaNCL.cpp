#include "mkl.h"
#include "matrix.h"
#include "Prox.h"
#include "barrier.h"
#include "tools.h"

#include <fstream>
#include <iostream>
#include <chrono>
#include <assert.h>
#include <vector>
#include <atomic>
#include <thread>
#include <algorithm>

using namespace std;

/******************** global parameter for parallel *********************/
const long long g_maxIter = 500;
ofstream g_LogFile("2.txt");

const long long g_rowSpt = 3;
const long long g_colSpt = g_rowSpt;

const long long g_maxPwIter = 2;

// partially observed matrix
vector<CooSparse> g_data = vector<CooSparse>(g_rowSpt*g_colSpt);
// matrix for multiplication
vector<CsrSparse> g_csrSpa = vector<CsrSparse>(g_rowSpt*g_colSpt);

vector<double*> g_partXY1 = vector<double*>(g_rowSpt*g_colSpt);
vector<double*> g_partXY0 = vector<double*>(g_rowSpt*g_colSpt);

CooSparse g_tsData;

vector<CMatrix> g_U1(g_rowSpt);
vector<CMatrix> g_V1(g_colSpt);

vector<CMatrix> g_U0(g_rowSpt);
vector<CMatrix> g_V0(g_colSpt);

vector<CMatrix> g_Ut(g_rowSpt);
vector<CMatrix> g_Vt(g_colSpt);

// for reduce matrix size
vector<CMatrix> g_Sml(max(g_rowSpt, g_colSpt));

CMatrix g_rU;
CMatrix g_rV;
CVector g_sv;

CMatrix g_Uqr;
CMatrix g_Vqr;

// acceleration
bool g_isAcc = false;

double g_alpha0 = 1;
double g_alpha1 = 1;

vector<double> g_blkVal(max(g_rowSpt, g_colSpt));

// barrier
Barrier *g_barrier = NULL;

/************************************************************************/

// hyper-parameter
double g_lambda;
double g_theta;
RegType g_rType;

double g_objVal[g_maxIter] = { 0 };
double g_runTme[g_maxIter] = { 0 };

auto g_startTme = chrono::steady_clock::now();

/************************************************************************/

// U = U * Diag(S)
void MergeSvToU(vector<CMatrix>& Ug, const CVector& sv)
{
	for (unsigned long long i = 0; i < Ug.size(); i++)
	{
		DenDiag(Ug.at(i), sv);
	}
}

// oU = iU*rU, oV = iV*rV
void UpdateFactUV(const vector<CMatrix>& iU, vector<CMatrix>& oU, const CMatrix& rU,
	const vector<CMatrix>& iV, vector<CMatrix>& oV, const CMatrix& rV)
{
	for (unsigned long long i = 0; i < iU.size(); i++)
	{
		DenDen(iU.at(i), false, rU, false, 1, oU.at(i), 0);
	}

	for (unsigned long long i = 0; i < iV.size(); i++)
	{
		DenDen(iV.at(i), false, rV, false, 1, oV.at(i), 0);
	}
}

// get object value
/* double GetObjectVal(const vector<CsrSparse>& csrSpa, 
	const CVector& sv,
	const double lambda, const double theta, RegType regType)
{
	double loss = 0;
	for (unsigned long long i = 0; i < csrSpa.size(); i++)
	{
		double lossi = cblas_dnrm2(csrSpa.at(i).nnz, csrSpa.at(i).pData, 1);
		loss = loss + 0.5*(lossi*lossi);
	}

	double reg = 0;
	for (long long i = 0; i < sv.length; i++)
	{
		sv.pData[i] = abs(sv.pData[i]);
	}

	RegObj(&reg, sv.pData, sv.length, lambda, theta, regType);

	return loss + reg;
} */

double GetObjectVal(const vector<double*>& pred,
	const vector<CooSparse>& data,
	const CVector& sv,
	const double lambda, const double theta, RegType regType)
{
	double loss = 0;
	for (unsigned long long i = 0; i < pred.size(); i++)
	{
		const long long nnzi = data[i].nnz;
		const double* predi = pred[i];
		const double* pDati = data[i].pData;
		
		double lossi = 0;
		for (long long p = 0; p < nnzi; p++)
		{
			lossi = *predi - *pDati;
			lossi = 0.5*lossi*lossi;

			loss = loss + lossi;
			
			predi++;
			pDati++;
		}
	}

	double reg = 0;
	for (long long i = 0; i < sv.length; i++)
	{
		sv.pData[i] = abs(sv.pData[i]);
	}

	RegObj(&reg, sv.pData, sv.length, lambda, theta, regType);

	return loss + reg;
}

double GetObjectVal(const vector<double>& blkLoss, const CVector& sv, const double lambda, const double theta, RegType regType)
{
	double loss = 0;
	for (unsigned long long i = 0; i < blkLoss.size(); i++)
	{
		loss = loss + blkLoss[i];
	}

	double reg = 0;
	for (long long i = 0; i < sv.length; i++)
	{
		sv.pData[i] = abs(sv.pData[i]);
	}

	RegObj(&reg, sv.pData, sv.length, lambda, theta, regType);

	return loss + reg;
}

double GetLossBlk_pl(const long long thdID, const vector<double*>& pred)
{
	double loss = 0;

	for (long long c = 0; c < g_colSpt; c++)
	{
		const long long blkID = thdID + c*g_rowSpt;

		const long long nnzi = g_data[blkID].nnz;
		const double* predc = pred.at(blkID);
		const double* datac = g_data.at(blkID).pData;

		for (long long p = 0; p < nnzi; p++)
		{
			double lossi = *predc - *datac;
			lossi = 0.5*lossi*lossi;

			loss = loss + lossi;

			predc++;
			datac++;
		}
	}

	return loss;
}

void ReducedSVD_pl(vector<CMatrix>& rS, CMatrix& oU, CVector& oS, CMatrix& oV, CMatrix& mTmp)
{
	// combine all reduced matrices
	long long length = mTmp.rows*mTmp.cols;
	long long incx = 1;
	double alpha = 1;

	memset(mTmp.pData, 0, sizeof(double)*length);
	for (unsigned long long i = 0; i < rS.size(); i++)
	{
		const CMatrix& rSi = rS.at(i);
		daxpy(&length, &alpha, rSi.pData, &incx, mTmp.pData, &incx);
		memset(rSi.pData, 0, sizeof(double)*(rSi.rows*rSi.cols));
	}

	// small size SVD
	ReducedSVD(mTmp, oU, oS, oV);
}

void ZeroGMat_pl(vector<CMatrix>& mat)
{
	for (unsigned long long i = 0; i < mat.size(); i++)
	{
		CMatrix& mati = mat.at(i);
		memset(mati.pData, 0, sizeof(double)*(mati.rows*mati.cols));
	}
}

void ReadLargeMat_pl(vector<CMatrix>& mat, const CMatrix& lrgMat)
{
	long long cpSta = 0;
	for (unsigned long long i = 0; i < mat.size(); i++)
	{
		CMatrix mati = mat.at(i);

		long long cpSiz = mati.rows;

		for (long long c = 0; c < mati.cols; c++)
		{
			memcpy(mati.pData + c*cpSiz, lrgMat.pData + c*lrgMat.rows + cpSta, sizeof(double)*cpSiz);
		}

		cpSta = cpSta + cpSiz;
	}
}

void WriteLargeMat_pl(const vector<CMatrix>& mat, CMatrix& lrgMat)
{
	long long cpSta = 0;
	for (unsigned long long i = 0; i < mat.size(); i++)
	{
		CMatrix mati = mat.at(i);

		long long cpSiz = mati.rows;

		for (long long c = 0; c < mati.cols; c++)
		{
			memcpy(lrgMat.pData + c*lrgMat.rows + cpSta, mati.pData + c*cpSiz, sizeof(double)*cpSiz);
		}

		cpSta = cpSta + cpSiz;
	}
}

double MakePrediction(const vector<CMatrix> & U, const vector<CMatrix> & V, CMatrix & gU, CMatrix & gV,
	const CooSparse & tsData)
{
	if (NULL == tsData.pData)
	{
		return 0;
	}
	WriteLargeMat_pl(U, gU);
	WriteLargeMat_pl(V, gV);

	double* pVal = new double[tsData.nnz];

	MakePartUV(gU, gV, tsData.pRowInd, tsData.pColInd, pVal, tsData.nnz);

	cblas_daxpy(tsData.nnz, -1.0, tsData.pData, 1, pVal, 1);

	double loss = cblas_dnrm2(tsData.nnz, pVal, 1);
	loss = loss / sqrt(tsData.nnz);

	delete pVal;

	return loss;
}

void QRFact_pl(vector<CMatrix>& partMat, CMatrix& qrMat)
{
	WriteLargeMat_pl(partMat, qrMat);

	QRfact(qrMat);

	ReadLargeMat_pl(partMat, qrMat);
}

// plain
void PowerMethod_splr_pl(const long long thdID,
	const vector<CMatrix>& gU1,
	const vector<CMatrix>& gV1,
	CMatrix& temp)
{
	for (long long pw = 0; pw < g_maxPwIter; pw++)
	{
		// update Ut
		if (0 == thdID) { ZeroGMat_pl(g_Ut); }

		g_barrier->Wait();

		// write to Ut(thdID)
		for (long long c = 0; c < g_colSpt; c++)
		{
			const CsrSparse& csrSpa = g_csrSpa.at(thdID + c*g_rowSpt);
			const CMatrix& U1 = gU1.at(thdID);
			const CMatrix& V1 = gV1.at(c);

			CMatrix& Ut = g_Ut.at(thdID);
			CMatrix& Vt = g_Vt.at(c);

			SpLrDen(csrSpa, U1, V1, false, Ut, Vt, temp);
		}

		g_barrier->Wait();

		if (0 == thdID) { QRFact_pl(g_Ut, g_Uqr); }

		// update Vt
		if (0 == thdID) { ZeroGMat_pl(g_Vt); }

		g_barrier->Wait();

		// write to Vt(thdID)
		for (long long r = 0; r < g_rowSpt; r++)
		{
			const CsrSparse& csrSpa = g_csrSpa.at(r + thdID*g_colSpt);
			const CMatrix& U1 = gU1.at(r);
			const CMatrix& V1 = gV1.at(thdID);

			CMatrix& Ut = g_Ut.at(r);
			CMatrix& Vt = g_Vt.at(thdID);

			SpLrDen(csrSpa, U1, V1, true, Ut, Vt, temp);
		}

		g_barrier->Wait();

		if (0 == thdID) { QRFact_pl(g_Vt, g_Vqr); }
	}

	g_barrier->Wait();
}

// for acceleration
void PowerMethod_splr_pl(const long long thdID, 
	const vector<CMatrix>& gU1,	const vector<CMatrix>& gV1, const double beta1, 
	const vector<CMatrix>& gU0, const vector<CMatrix>& gV0, const double beta0, CMatrix& temp)
{
	g_barrier->Wait();

	for (long long pw = 0; pw < g_maxPwIter; pw++)
	{
		// update Ut
		if (0 == thdID)
		{
			ZeroGMat_pl(g_Ut);
		}
		g_barrier->Wait();

		// write to Ut(thdID)
		for (long long c = 0; c < g_colSpt; c++)
		{
			const CsrSparse& csrSpa = g_csrSpa.at(thdID + c*g_rowSpt);
			const CMatrix& U0 = gU0.at(thdID);
			const CMatrix& V0 = gV0.at(c);
			const CMatrix& U1 = gU1.at(thdID);
			const CMatrix& V1 = gV1.at(c);

			CMatrix& Ut = g_Ut.at(thdID);
			CMatrix& Vt = g_Vt.at(c);

			SpLrDen(csrSpa, U1, V1, beta1, U0, V0, beta0, false, Ut, Vt, temp);
		}

		g_barrier->Wait();

		if (0 == thdID)
		{
			QRFact_pl(g_Ut, g_Uqr);
		}

		// update Vt
		if (0 == thdID)
		{
			ZeroGMat_pl(g_Vt);
		}
		g_barrier->Wait();

		// write to Vt(thdID)
		for (long long r = 0; r < g_rowSpt; r++)
		{
			const CsrSparse& csrSpa = g_csrSpa.at(r + thdID*g_colSpt);
			const CMatrix& U0 = gU0.at(r);
			const CMatrix& V0 = gV0.at(thdID);
			const CMatrix& U1 = gU1.at(r);
			const CMatrix& V1 = gV1.at(thdID);

			CMatrix& Ut = g_Ut.at(r);
			CMatrix& Vt = g_Vt.at(thdID);

			SpLrDen(csrSpa, U1, V1, beta1, U0, V0, beta0, true, Ut, Vt, temp);
		}

		g_barrier->Wait();

		if (0 == thdID)
		{
			QRFact_pl(g_Vt, g_Vqr);
		}
	}

	g_barrier->Wait();
}

void ReduceMat_pl(const long long thdID, 
	CMatrix& splrTemp, CMatrix& uTmp, CMatrix& vTmp)
{
	const long long redSize = (g_U1.at(0).cols)*(g_U1.at(0).cols);

	memset(g_Sml.at(thdID).pData, 0, sizeof(double)*redSize);

	for (long long c = 0; c < g_colSpt; c++)
	{
		CsrSparse& csrSpa = g_csrSpa.at(thdID + c*g_rowSpt);

		CMatrix& U1 = g_U1.at(thdID);
		CMatrix& V1 = g_V1.at(c);

		CMatrix& Ut = g_Ut.at(thdID);
		CMatrix& Vt = g_Vt.at(c);

		ReducedMat(csrSpa, U1, V1, Ut, Vt, splrTemp, uTmp, vTmp);

		cblas_daxpy(redSize, 1, splrTemp.pData, 1, g_Sml.at(thdID).pData, 1);
	}

	memset(splrTemp.pData, 0, sizeof(double)*redSize);

	memset(uTmp.pData, 0, sizeof(double)*redSize);
	memset(vTmp.pData, 0, sizeof(double)*redSize);
}

void ReduceMat_pl(const long long thdID, 
	const double beta1, const double beta0,
	CMatrix& splrTemp, CMatrix& uTmp, CMatrix& vTmp)
{
	const long long redSize = (g_U1.at(0).cols)*(g_U1.at(0).cols);

	memset(g_Sml.at(thdID).pData, 0, sizeof(double)*redSize);

	for (long long c = 0; c < g_colSpt; c++)
	{
		CsrSparse& csrSpa = g_csrSpa.at(thdID + c*g_rowSpt);

		CMatrix& U1 = g_U1.at(thdID);
		CMatrix& V1 = g_V1.at(c);

		CMatrix& U0 = g_U0.at(thdID);
		CMatrix& V0 = g_V0.at(c);

		CMatrix& Ut = g_Ut.at(thdID);
		CMatrix& Vt = g_Vt.at(c);

		ReducedMat(csrSpa, U1, V1, beta1, U0, V0, beta0, Ut, Vt, splrTemp, uTmp, vTmp);

		cblas_daxpy(redSize, 1, splrTemp.pData, 1, g_Sml.at(thdID).pData, 1);
	}

	memset(splrTemp.pData, 0, sizeof(double)*redSize);

	memset(uTmp.pData, 0, sizeof(double)*redSize);
	memset(vTmp.pData, 0, sizeof(double)*redSize);
}

void MakeCSR_pl(const long long thdID)
{
	for (long long c = 0; c < g_colSpt; c++)
	{
		// CMatrix& U1 = g_U1.at(thdID);
		// CMatrix& V1 = g_V1.at(c);

		const long long blkID = thdID + c*g_rowSpt;
		const CooSparse& data = g_data.at(blkID);
		CsrSparse& csrSpa = g_csrSpa.at(blkID);

		// MakePartUV(U1, V1, data.pRowInd, data.pColInd, csrSpa.pData, data.nnz);

		memcpy(csrSpa.pData, g_partXY1.at(blkID), sizeof(double)*data.nnz);
		UpdateToCSR(csrSpa.pData, data.pData, data.nnz);
	}
}

void MakePartXY_pl(const long long thdID)
{
	for (long long c = 0; c < g_colSpt; c++)
	{
		const CMatrix& Ut = g_Ut.at(thdID);
		const CMatrix& Vt = g_Vt.at(c);

		const long long blkID = thdID + c*g_rowSpt;
		const CooSparse& data = g_data.at(blkID);
		const CsrSparse& spar = g_csrSpa.at(blkID);

		memcpy(g_partXY0.at(blkID), g_partXY1.at(blkID), sizeof(double)*data.nnz);

		MakePartUV(Ut, Vt, data.pRowInd, data.pColInd, spar.pData, data.nnz);
	}
}

/*void CopyGMats()
{
	for (long long r = 0; r < g_rowSpt; r++)
	{
		CopyMat(g_U1.at(r), g_U0.at(r));
		CopyMat(g_Ut.at(r), g_U1.at(r));
	}

	for (long long c = 0; c < g_colSpt; c++)
	{
		CopyMat(g_V1.at(c), g_V0.at(c));
		CopyMat(g_Vt.at(c), g_V1.at(c));
	}

	for (unsigned long long i = 0; i < g_data.size(); i++)
	{
		const long long nnzi = g_data.at(i).nnz;

		memcpy(g_partXY0.at(i), g_partXY1.at(i), sizeof(double)*nnzi);
		memcpy(g_partXY1.at(i), g_csrSpa.at(i).pData, sizeof(double)*nnzi);
	}
}*/

void CopyGMats_pl(const long long thdID)
{
	CopyMat(g_Ut.at(thdID), g_U1.at(thdID));
	CopyMat(g_Vt.at(thdID), g_V1.at(thdID));

	for (long long c = 0; c < g_colSpt; c++)
	{
		const long long blkID = thdID + c*g_rowSpt;
		const long long nnzi = g_data.at(blkID).nnz;

		memcpy(g_partXY1.at(blkID), g_csrSpa.at(blkID).pData, sizeof(double)*nnzi);
	}
}

void CopyGMats_acc_pl(const long long thdID)
{
	CopyMat(g_U1.at(thdID), g_U0.at(thdID));
	CopyMat(g_V1.at(thdID), g_V0.at(thdID));

	CopyMat(g_Ut.at(thdID), g_U1.at(thdID));
	CopyMat(g_Vt.at(thdID), g_V1.at(thdID));

	for (long long c = 0; c < g_colSpt; c++)
	{
		const long long blkID = thdID + c*g_rowSpt;
		const long long nnzi = g_data.at(blkID).nnz;
		
		memcpy(g_partXY0.at(blkID), g_partXY1.at(blkID), sizeof(double)*nnzi);
		memcpy(g_partXY1.at(blkID), g_csrSpa.at(blkID).pData, sizeof(double)*nnzi);
	}
}

void RecordTime(const long long i)
{
	// record running time
	auto currtTime = chrono::steady_clock::now() - g_startTme;
	if (i == 0)
	{
		g_runTme[i] = (double)chrono::duration_cast<chrono::milliseconds>(currtTime).count();
	}
	else
	{
		g_runTme[i] = g_runTme[i - 1] + (double)chrono::duration_cast<chrono::milliseconds>(currtTime).count();
	}
	cout << i << "," << g_objVal[i] << "," << g_runTme[i] / 1000 << "," << g_isAcc;
	g_LogFile << i << "," << g_objVal[i] << "," << g_runTme[i] / 1000 << "," << g_isAcc;

	// make prediction
	if (NULL != g_tsData.pData)
	{
		double preLoss = MakePrediction(g_U1, g_V1, g_Uqr, g_Vqr, g_tsData);
		cout << "," << preLoss;
		g_LogFile << "," << preLoss;
	}
	cout << endl;
	g_LogFile << endl;

	g_startTme = chrono::steady_clock::now();
}

void FaNCL_pl(const long long thdID)
{
	const long long maxRank = g_U1.at(0).cols;

	CMatrix splrTemp = InitMat(maxRank, maxRank);
	CMatrix reduSTmp = InitMat(maxRank, maxRank);
	CMatrix redvSTmp = InitMat(maxRank, maxRank);

	for (long long i = 0; i < g_maxIter; i++)
	{
		g_blkVal[thdID] = GetLossBlk_pl(thdID, g_partXY1);

		g_barrier->Wait();

		if (0 == thdID) { g_objVal[i] = GetObjectVal(g_blkVal, g_sv, g_lambda, g_theta, g_rType); }

		// makeup CSR part
		MakeCSR_pl(thdID);

		g_barrier->Wait();

		// power method
		PowerMethod_splr_pl(thdID, g_U1, g_V1, splrTemp);

		g_barrier->Wait();

		// reduce matrix size
		ReduceMat_pl(thdID, splrTemp, reduSTmp, redvSTmp);

		g_barrier->Wait();

		if (thdID == 0)
		{
			ReducedSVD_pl(g_Sml, g_rU, g_sv, g_rV, splrTemp);
			ProxStep(g_sv.pData, g_sv.pData, g_sv.length, g_lambda, g_theta, g_rType);
			UpdateFactUV(g_Ut, g_Ut, g_rU, g_Vt, g_Vt, g_rV);
			MergeSvToU(g_Ut, g_sv);
		}

		g_barrier->Wait();

		MakePartXY_pl(thdID);

		g_barrier->Wait();

		CopyGMats_pl(thdID);

		g_barrier->Wait();

		if (0 == thdID) { RecordTime(i); }
	}

	FreeMat(splrTemp);
	FreeMat(reduSTmp);
	FreeMat(redvSTmp);
}

void FaNCLacc_pl(const long long thdID)
{
	const long long maxRank = g_U1.at(0).cols;

	CMatrix splrTemp = InitMat(maxRank, maxRank);
	CMatrix reduSTmp = InitMat(maxRank, maxRank);
	CMatrix redvSTmp = InitMat(maxRank, maxRank);

	for (long long i = 0; i < g_maxIter; i++)
	{
		double beta0 = -(g_alpha0 - 1) / g_alpha1;
		double beta1 = 1 - beta0;

		// double beta0 = 0; double beta1 = 1;

		g_blkVal[thdID] =  GetLossBlk_pl(thdID, g_partXY1);

		g_barrier->Wait();

		if (0 == thdID) { g_objVal[i] = GetObjectVal(g_blkVal, g_sv, g_lambda, g_theta, g_rType); }

		// makeup CSR part
		MakeCSR_pl(thdID);

		g_barrier->Wait();

		// power method
		PowerMethod_splr_pl(thdID, g_U1, g_V1, beta1, g_U0, g_V0, beta0, splrTemp);

		g_barrier->Wait();

		// reduce matrix size
		ReduceMat_pl(thdID, beta1, beta0, splrTemp, reduSTmp, redvSTmp);
		
		g_barrier->Wait();

		if (thdID == 0)
		{
			ReducedSVD_pl(g_Sml, g_rU, g_sv, g_rV, splrTemp);
			ProxStep(g_sv.pData, g_sv.pData, g_sv.length, g_lambda, g_theta, g_rType);
			UpdateFactUV(g_Ut, g_Ut, g_rU, g_Vt, g_Vt, g_rV);
			MergeSvToU(g_Ut, g_sv);
		}

		g_barrier->Wait();

		MakePartXY_pl(thdID);

		g_barrier->Wait();

		g_blkVal[thdID] = GetLossBlk_pl(thdID, getDataVec(g_csrSpa));

		if (0 == thdID)
		{
			double objAcc = GetObjectVal(g_blkVal, g_sv, g_lambda, g_theta, g_rType);

			g_isAcc = (objAcc < g_objVal[i]);
			if (g_isAcc)
			{
				g_alpha0 = g_alpha1;
				g_alpha1 = 0.5*(1 + sqrt(1.0 + 4 * g_alpha1*g_alpha1));
			}
			else
			{
				g_alpha0 = 1;
				g_alpha1 = 1;
			}
		}

		CopyGMats_acc_pl(thdID);

		g_barrier->Wait();

		if (0 == thdID) { RecordTime(i); }
	}

	FreeMat(splrTemp);
	FreeMat(reduSTmp);
	FreeMat(redvSTmp);
}

/*void InitSvs(const CooSparse& iSpa, const CMatrix& iU, const CMatrix& iV, CVector& oSv)
{
	// can not get singular values
	double* pUcol = iU.pData;
	double* pVcol = iV.pData;
	const long long inc = 1;
	double alpha = -1.0;

	CVector proV = InitVec(iU.rows);
	for (long long sv = 0; sv < oSv.length; sv++)
	{
		SpaVec(iSpa, pVcol, 1, proV.pData, 0, false);

		oSv.pData[sv] = ddot(&iU.rows, pUcol, &inc, proV.pData, &inc);

		pUcol = pUcol + iU.rows;
		pVcol = pVcol + iV.rows;
	}

	FreeVec(proV);
}*/

void genOneSplit(const CooSparse& data,
	pair<long long, long long>& rowRange, const pair<long long, long long>& colRange,
	CooSparse& scooSpa, CsrSparse& scsrSpa)
{
	long long rSta = 0;
	long long rEnd = 0;

	for (long long r = 0; r < data.nnz; r++)
	{
		if (rowRange.first == data.pRowInd[r])
		{
			rSta = r;
			break;
		}
	}

	for (long long r = rSta; r < data.nnz; r++)
	{
		if (rowRange.second == data.pRowInd[r])
		{
			rEnd = r;
		}
	}

	// count non-zero elements inside this block
	long long snnz = 0;
	for (long long c = rSta; c <= rEnd; c++)
	{
		if (data.pColInd[c] >= colRange.first && data.pColInd[c] <= colRange.second)
		{
			snnz = snnz + 1;
		}
	}
	const long long srows = rowRange.second - rowRange.first + 1;
	const long long scols = colRange.second - colRange.first + 1;

	scooSpa.nnz = snnz;
	scooSpa.pData = new double[snnz];
	scooSpa.pRowInd = new long long[snnz];
	scooSpa.pColInd = new long long[snnz];

	// initialize values: row, col & val (the index starts from 1 for each block)
	snnz = 0;
	for (long long c = rSta; c <= rEnd; c++)
	{
		if (data.pColInd[c] >= colRange.first && data.pColInd[c] <= colRange.second)
		{
			scooSpa.pData[snnz] = data.pData[c];
			scooSpa.pRowInd[snnz] = data.pRowInd[c] - rowRange.first + 1;
			scooSpa.pColInd[snnz] = data.pColInd[c] - colRange.first + 1;

			snnz = snnz + 1;
		}
	}

	scooSpa.rows = srows;
	scooSpa.cols = scols;
	scooSpa.staRow = rowRange.first;
	scooSpa.staCol = colRange.first;

	// initialize CSR matrix from COO format
	scsrSpa.nnz = snnz;
	scsrSpa.rows = srows;
	scsrSpa.cols = scols;
	scsrSpa.staRow = rowRange.first;
	scsrSpa.staCol = colRange.first;

	scsrSpa.rowIndex = new long long[scsrSpa.rows + 1];
	scsrSpa.pColInd = new long long[scsrSpa.nnz];
	scsrSpa.pData = new double[scsrSpa.nnz];

	GenCsrFCoo(scooSpa, scsrSpa);

	memset(scsrSpa.pData, 0, scsrSpa.nnz * sizeof(double));
}

void GenSplitSpaMat(const CooSparse& data, const long long rowSpt, const long long colSpt,
	vector<CooSparse>& cooSpt, vector<CsrSparse>& csrSpa,
	vector<double*>& gPart1, vector<double*>& gPart0)
{
	long long* key = new long long[data.nnz];

	// sort row index in increasing order
	SortIdx(data.pRowInd, data.nnz, key);
	SwapIdx(key, data.nnz, data.pColInd);
	SwapIdx(key, data.nnz, data.pData);

	vector<pair<long long, long long>> rsIdx = LinearSplit(data.rows, rowSpt);
	vector<pair<long long, long long>> csIdx = LinearSplit(data.cols, colSpt);

	for (long long c = 0; c < colSpt; c++)
	{
		for (long long r = 0; r < rowSpt; r++)
		{
			genOneSplit(data, rsIdx[r], csIdx[c], cooSpt[r + c*rowSpt], csrSpa[r + c*rowSpt]);
		}
	}

	long long nnz = 0;
	for (long long i = 0; i < rowSpt*colSpt; i++)
	{
		nnz = nnz + cooSpt[i].nnz;
	}

	assert(nnz == data.nnz);

	for (unsigned long long i = 0; i < cooSpt.size(); i++)
	{
		const long long nnzi = cooSpt.at(i).nnz;
		gPart1.at(i) = new double[nnzi];
		memset(gPart1.at(i), 0, sizeof(double)*nnzi);

		gPart0.at(i) = new double[nnzi];
		memset(gPart0.at(i), 0, sizeof(double)*nnzi);
	}
}

void genSplitOneFactor(const CMatrix& iU, vector<CMatrix>& oU)
{
	const long long rows = iU.rows;
	const long long cols = iU.cols;
	const long long spt = oU.size();

	vector<pair<long long, long long>> rsIdx = LinearSplit(rows, spt);

	for (unsigned long long i = 0; i < oU.size(); i++)
	{
		CMatrix& oUi = oU.at(i);

		oUi.rows = rsIdx[i].second - rsIdx[i].first + 1;
		oUi.cols = cols;
		oUi.pData = new double[oUi.rows * oUi.cols];

		for (long long c = 0; c < cols; c++)
		{
			long long offset = rsIdx[i].first + iU.rows*c - 1;
			long long cpysiz = oUi.rows;

			memcpy(oUi.pData + c*oUi.rows, (iU.pData + offset), sizeof(double)*cpysiz);
		}
	}
}

void GenSplitFactor(CooSparse& data, const long long maxRank,
	vector<CMatrix>& gU1, vector<CMatrix>& gU0, vector<CMatrix>& gUt,
	vector<CMatrix>& gV1, vector<CMatrix>& gV0, vector<CMatrix>& gVt,
	CVector& osv)
{
	// init big factors (needs splits)
	CMatrix Vg = GenGaussian(data.cols, maxRank);
	QRfact(Vg);
	CMatrix Ug = GenGaussian(data.rows, maxRank);
	QRfact(Ug);
	osv = InitVec(maxRank);

	// split big matrix into blocks
	genSplitOneFactor(Ug, gU1);
	genSplitOneFactor(Vg, gV1);
	genSplitOneFactor(Ug, gU0);
	genSplitOneFactor(Vg, gV0);
	genSplitOneFactor(Ug, gUt);
	genSplitOneFactor(Vg, gVt);

	FreeMat(Vg);
	FreeMat(Ug);
}

void InitRedMat(vector<CMatrix> & mat, CMatrix& rU, CMatrix& rV, CVector& rsv,
	const long long maxRank)
{
	for (unsigned long long i = 0; i < mat.size(); i++)
	{
		mat[i] = InitMat(maxRank, maxRank);
	}

	rU = InitMat(maxRank, maxRank);
	rV = InitMat(maxRank, maxRank);
	rsv = InitVec(maxRank);
}

void FreeGlbRes()
{
	for (unsigned long long i = 0; i < g_data.size(); i++)
	{
		FreeCooSparse(g_data.at(i));
		FreeCsrSparse(g_csrSpa.at(i));

		delete[] g_partXY0[i];
		delete[] g_partXY1[i];
	}

	for (unsigned long long i = 0; i < g_U1.size(); i++)
	{
		FreeMat(g_U1.at(i));
		FreeMat(g_U0.at(i));
		FreeMat(g_Ut.at(i));
	}

	for (unsigned long long i = 0; i < g_V1.size();i++)
	{
		FreeMat(g_V1.at(i));
		FreeMat(g_V0.at(i));
		FreeMat(g_Vt.at(i));
	}

	FreeVec(g_sv);

	for (unsigned long long i = 0; i < g_Sml.size(); i++)
	{
		FreeMat(g_Sml.at(i));
	}

	FreeMat(g_rU);
	FreeMat(g_rV);

	FreeMat(g_Uqr);
	FreeMat(g_Vqr);

	delete g_barrier;
}

// maxRank is the estimated rank
void FaNCL(CooSparse& data, double lambda, double theta, RegType rType, const long long maxRank, 
	const CooSparse& tsData, bool isAcc)
{
	// set global hyper-parameter
	g_lambda = lambda;
	g_theta  = theta;
	g_rType  = rType;

	g_tsData = tsData;
	// g_LogFile.open(g_LogName);

	// split the big sparse data into [row split x column split] parts
	GenSplitSpaMat(data, g_rowSpt, g_colSpt, g_data, g_csrSpa, g_partXY1, g_partXY0);
	
	// split factors into blocks
	GenSplitFactor(data, maxRank, g_U1, g_U0, g_Ut, g_V1, g_V0, g_Vt, g_sv);

	InitRedMat(g_Sml, g_rU, g_rV, g_sv, maxRank);

	g_Uqr = InitMat(data.rows, maxRank);
	g_Vqr = InitMat(data.cols, maxRank);

	// data is allocated to global sparse matrices
	FreeCooSparse(data);

	// set barrier
	g_barrier = new Barrier(g_rowSpt);

	// config information
	for (unsigned long long i = 0; i < g_csrSpa.size(); i++)
	{
		cout << "block:" << i << " nnz:" << g_csrSpa.at(i).nnz << " rows:" << g_csrSpa.at(i).rows
			<< " cols:" << g_csrSpa.at(i).cols << endl;
	}

	// start parallel
	g_startTme = chrono::steady_clock::now();

	vector<std::thread> thds;
	long long toThreads = g_rowSpt;
	for (long long i = 0; i < toThreads; i++)
	{
		if (false == isAcc)
		{
			thds.push_back(thread(FaNCL_pl, i));
		}
		else
		{
			thds.push_back(thread(FaNCLacc_pl, i));
		}
	}

	for (long long i = 0; i < toThreads; i++) {thds[i].join();}

	// free resource
	FreeGlbRes();
}