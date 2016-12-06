#include "mkl.h"
#include "matrix.h"
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <random>

using namespace std;

// one - based indexing: https://software.intel.com/zh-cn/node/520801#TBL2-7
const char matdescra[6] = { 'G', 0, 0, 'F', 0, 0 };

CVector InitVec(long long leg)
{
	CVector vec = {};
	vec.length = leg;
	vec.pData = new double [leg];

	memset(vec.pData, 0, sizeof(double)*leg);

	return vec;
}

void FreeVec(CVector& pVec)
{
	pVec.length = 0;
	delete[] pVec.pData;

	pVec.pData = NULL;
}

void PrintCVector(const CVector& vec, long long prec)
{
	for (long long i = 0; i < vec.length; i++)
	{
		cout << fixed << setprecision(prec);
		cout << vec.pData[i] << " ";
		cout << endl;
	}
}

CMatrix InitMat(long long rows, long long cols)
{
	CMatrix mat = {};

	mat.rows = rows;
	mat.cols = cols;
	mat.pData = new(double[rows*cols]);
	memset(mat.pData, 0, rows*cols*sizeof(double));

	return mat;
}

void FreeMat(CMatrix& oB)
{
	oB.rows = 0;
	oB.cols = 0;
	delete[] (oB.pData);
	oB.pData = NULL;

	return;
}

void CopyMat(const CMatrix& fM, CMatrix& tM)
{
	tM.rows = fM.rows;
	tM.cols = fM.cols;

	long long inc = 1;
	long long leg = tM.rows*tM.cols;
	
	dcopy(&leg, fM.pData, &inc, tM.pData, &inc);
}

void PrintCMatrix(const CMatrix& mat)
{
	for (long long r = 0; r < mat.rows; r++)
	{
		for (long long c = 0; c < mat.cols; c++)
		{
			cout << fixed << setprecision(2);
			cout << mat.pData[c*(mat.rows) + r] << " ";
		}
		cout << endl;
	}
}

void PrintCooSparse(const CooSparse& mat)
{

	cout << "row ";
	for (long long r = 0; r < mat.nnz; r++)
	{
		cout << mat.pRowInd[r] << " ";
	}
	cout << endl;

	cout << "col ";
	for (long long r = 0; r < mat.nnz; r++)
	{
		cout << mat.pColInd[r] << " ";
	}
	cout << endl;
	
	cout << "val ";
	for (long long r = 0; r < mat.nnz; r++)
	{
		cout << mat.pData[r] << " ";
	}
}

void TransCooSparse(CooSparse& mat)
{
	long long* pIndex = mat.pRowInd;

	mat.pRowInd = mat.pColInd;
	mat.pColInd = pIndex;
}

void FreeCooSparse(CooSparse& spa)
{
	delete[] spa.pData;
	spa.pData = NULL;

	delete[] spa.pColInd;
	spa.pColInd = NULL;

	delete[] spa.pRowInd;
	spa.pRowInd = NULL;

	spa.cols = 0;
	spa.rows = 0;
	spa.nnz = 0;
}

CsrSparse InitCsrSparse(long long nnz, long long rows, long long cols)
{
	CsrSparse csrSpa = {};

	csrSpa.cols = cols;
	csrSpa.rows = rows;
	csrSpa.nnz = nnz;
	csrSpa.pData = new double[nnz];
	csrSpa.pColInd = new long long[nnz];
	csrSpa.rowIndex = new long long[rows + 1];

	return csrSpa;
}

void PrintCsrSparse(const CsrSparse& mat)
{
	cout << "col ";
	for (long long r = 0; r < mat.nnz; r++)
	{
		cout << mat.pColInd[r] << " ";
	}
	cout << endl;

	cout << "val ";
	for (long long r = 0; r < mat.nnz; r++)
	{
		cout << mat.pData[r] << " ";
	}
}



void FreeCsrSparse(CsrSparse& spa)
{
	delete[] spa.pData;
	spa.pData = NULL;

	delete[] spa.pColInd;
	spa.pColInd = NULL;

	delete[] spa.rowIndex;;
	spa.rowIndex = NULL;

	spa.cols = 0;
	spa.rows = 0;
	spa.nnz = 0;
}

long long GenCsrFCoo(CooSparse& cooSpa, CsrSparse& csrSpa)
{
	const long long coo2csr[6] = { 2, 1, 1, 0, cooSpa.nnz, 0 };
	long long info = 0;

	mkl_dcsrcoo(coo2csr, &csrSpa.rows, csrSpa.pData, csrSpa.pColInd, csrSpa.rowIndex,
		&csrSpa.nnz, cooSpa.pData, cooSpa.pRowInd, cooSpa.pColInd, &info);

	const long long csr2coo[6] = { 0, 1, 1, 0, cooSpa.nnz, 3 };

	mkl_dcsrcoo(csr2coo, &csrSpa.rows, csrSpa.pData, csrSpa.pColInd, csrSpa.rowIndex,
		&csrSpa.nnz, cooSpa.pData, cooSpa.pRowInd, cooSpa.pColInd, &info);

	return info;
}

// power method for sparse matrix
void PowerMethod_spa(const CooSparse iSp, CMatrix& oU, CMatrix& ioV)
{
	for (long long t = 0; t < 3; t++)
	{
		SpaDen(iSp, ioV, 1, oU, 0, false);
		QRfact(oU);
		// PrintCMatrix(*oU);

		SpaDen(iSp, oU, 1, ioV, 0, true);
		QRfact(ioV);

		// PrintCMatrix(*ioV);
	}
}

// C := alpha*A*B + beta*C || C := alpha*A^T*B + beta*C,
void SpaDen(const CooSparse& iA, const CMatrix& iB, double alpha, 
	CMatrix& oC, double beta, bool trans)
{
	if (false == trans)
	{
		const char cTran = 'N';
		long long m = iA.rows;
		long long n = oC.cols;
		long long k = iA.cols;

		// PrintCSparse(iA);
		// cout << endl;
		// PrintCMatrix(iB);
		// cout << endl;

		mkl_dcoomm(&cTran, &m, &n, &k, &alpha, matdescra,
			iA.pData, iA.pRowInd, iA.pColInd, &iA.nnz, iB.pData, &k,
			&beta, oC.pData, &m);

		// PrintCMatrix(*oC);
		// cout << endl;
	}
	else
	{
		const char cTran = 'T';
		long long m = iA.rows;
		long long n = oC.cols;
		long long k = iA.cols;

		// PrintCSparse(iA);
		// cout << endl;
		// PrintCMatrix(iB);
		// cout << endl;

		mkl_dcoomm(&cTran, &m, &n, &k, &alpha, matdescra,
			iA.pData, iA.pRowInd, iA.pColInd, &iA.nnz, iB.pData, &m,
			&beta, oC.pData, &k);

		// PrintCMatrix(*oC);
		// cout << endl;
	}

	// PrintCMatrix(*oC);
}

void SpaDen(const CsrSparse& iA, const CMatrix& iB, double alpha,
	CMatrix& oC, double beta, bool trans)
{
	// PrintCSparse(iA);
	// PrintCMatrix(iB);

	if (false == trans)
	{
		const char cTran = 'N';
		long long m = iA.rows;
		long long n = oC.cols;
		long long k = iA.cols;

		mkl_dcsrmm(&cTran, &m, &n, &k, &alpha, matdescra, iA.pData, iA.pColInd, iA.rowIndex, 
			(iA.rowIndex + 1), iB.pData, &k,
			&beta, oC.pData, &m);
	}
	else
	{
		const char cTran = 'T';
		long long m = iA.rows;
		long long n = oC.cols;
		long long k = iA.cols;

		mkl_dcsrmm(&cTran, &m, &n, &k, &alpha, matdescra, iA.pData, iA.pColInd, iA.rowIndex,
			(iA.rowIndex + 1), iB.pData, &m,
			&beta, oC.pData, &k);
	}

	// PrintCMatrix(*oC);
}

// y := alpha*A*x + beta*y
void SpaVec(const CooSparse& iA, const CVector& iB, double alpha, CVector& oC, double beta, bool trans)
{
	const char cTran = 'N';

	if (false == trans)
	{
		mkl_dcoomv(&cTran, &iA.rows, &iA.cols, &alpha, matdescra, iA.pData,
			iA.pRowInd, iA.pColInd, &iA.nnz, iB.pData, &beta, oC.pData);
	}
	else
	{
	}
}

void SpaVec(const CooSparse& iA, double* iB, double alpha, double* oC, double beta, bool trans)
{
	const char cTran = 'N';

	if (false == trans)
	{
		mkl_dcoomv(&cTran, &iA.rows, &iA.cols, &alpha, matdescra, iA.pData,
			iA.pRowInd, iA.pColInd, &iA.nnz, iB, &beta, oC);
	}
	else
	{
	}
}

void SpaVec(const CsrSparse& iA, double* iB, double alpha, double* oC, double beta, bool trans)
{
	const char cTran = 'N';

	if (false == trans)
	{
		mkl_dcsrmv(&cTran, &iA.rows, &iA.cols, &alpha, matdescra, iA.pData, iA.pColInd, iA.rowIndex,
			iA.rowIndex + 1, iB, &beta, oC);
	}
	else
	{
	}
}

// (sparse + low-rank) x dense
void SpLrDen(const CsrSparse& iSpa, const CMatrix& iU, const CMatrix& iV, bool trans,
	CMatrix& oiU, CMatrix& oiV, CMatrix& dpTpVM)
{
	memset(dpTpVM.pData, 0, sizeof(double)*(dpTpVM.rows * dpTpVM.cols));
	
	if (false == trans)
	{
		// sparse part
		SpaDen(iSpa, oiV, 1, oiU, 1, false); // add in
		// low-rank part
		DenDen(iV, true, oiV, false, 1, dpTpVM, 0);
		DenDen(iU, false, dpTpVM, false, 1, oiU, 1);
	}
	else
	{
		SpaDen(iSpa, oiU, 1, oiV, 1, true); // add in
		DenDen(iU, true, oiU, false, 1, dpTpVM, 0);
		DenDen(iV, false, dpTpVM, false, 1, oiV, 1);
	}
}

/* void SpLrDen(const CooSparse& iSpa, const CMatrix& iU, const CMatrix& iV, bool trans,
	CMatrix& oiU, CMatrix& oiV)
{
	CMatrix dpTpVM = InitMat(iV.cols, oiV.cols);

	if (false == trans)
	{
		// sparse part
		SpaDen(iSpa, oiV, 1, oiU, 0, false);

		// low-rank part
		DenDen(iV, true, oiV, false, 1, dpTpVM, 0);
		DenDen(iU, false, dpTpVM, false, 1, oiU, 1);
	}
	else
	{
		SpaDen(iSpa, oiU, 1, oiV, 0, true);
		DenDen(iU, true, oiU, false, 1, dpTpVM, 0);
		DenDen(iV, false, dpTpVM, false, 1, oiV, 1);
	}

	FreeMat(dpTpVM);
} */

void SpLrDen(const CsrSparse& iSpa,
	const CMatrix& U1, const CMatrix& V1, const double beta1,
	const CMatrix& U0, const CMatrix& V0, const double beta0,
	bool trans,
	CMatrix& oiU, CMatrix& oiV,
	CMatrix& dpTpVM)
{
	memset(dpTpVM.pData, 0, sizeof(double)*(dpTpVM.rows * dpTpVM.cols));

	if (false == trans)
	{
		// sparse part
		SpaDen(iSpa, oiV, 1, oiU, 1, false);

		// low-rank part
		DenDen(V1, true, oiV, false, 1, dpTpVM, 0);
		DenDen(U1, false, dpTpVM, false, beta1, oiU, 1);

		DenDen(V0, true, oiV, false, 1, dpTpVM, 0);
		DenDen(U0, false, dpTpVM, false, beta0, oiU, 1);
	}
	else
	{
		SpaDen(iSpa, oiU, 1, oiV, 1, true);
		DenDen(U1, true, oiU, false, 1, dpTpVM, 0);
		DenDen(V1, false, dpTpVM, false, beta1, oiV, 1);

		DenDen(U0, true, oiU, false, 1, dpTpVM, 0);
		DenDen(V0, false, dpTpVM, false, beta0, oiV, 1);
	}
}

long long ReducedSVD(const CMatrix iMat, CMatrix& oU, CVector& oS, CMatrix& oV)
{
	// standard svd algorithm
	double* superb = new(double[oS.length]);
	long long info = LAPACKE_dgesvd(LAPACK_COL_MAJOR, 'A', 'A', iMat.rows, iMat.cols, iMat.pData, iMat.rows, oS.pData,
		oU.pData, oU.rows, oV.pData, oV.rows, superb);
	delete superb;

	// svd algorithm: divide and conquer method (faster)
	// https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/20050192421.pdf
	/* long long info = LAPACKE_dgesdd(LAPACK_COL_MAJOR, 'S',
	iMat.rows, iMat.cols, iMat.pData, iMat.rows,
	oS.pData,
	oU.pData, oU.rows,
	oV.pData, oV.rows); */

	return 0;
}

// A = QR(A)
void QRfact(CMatrix& iA)
{
	CMatrix AtA = InitMat(iA.cols, iA.cols);
	CMatrix U = InitMat(iA.cols, iA.cols);
	CMatrix V = InitMat(iA.cols, iA.cols);
	CVector s = InitVec(iA.cols);

	DenDen(iA, true, iA, false, 1, AtA, 0);
	ReducedSVD(AtA, U, s, V);
	DenDen(iA, false, U, false, 1, iA, 0);

	for (long long i = 0; i < s.length; i++)
	{
		double sci = 1.0 / sqrt(s.pData[i] + 0.0001);
		cblas_dscal(iA.rows, sci, iA.pData + i*iA.rows, 1);
	}

	FreeMat(AtA);
	FreeMat(U);
	FreeMat(V);
	FreeVec(s);
}

// make matrix have unit column: column major
/* void normCol(CMatrix& pMat)
{
	double *pData = pMat.pData;
	const long long iRows = pMat.rows;

	for (long long c = 0; c < pMat.cols; c++)
	{
		double cnm = cblas_dnrm2(iRows, pData, 1);
		cblas_dscal(iRows, 1 / cnm, pData, 1);
		pData = pData + iRows;
	}

	return;
} */

// C := alpha*op(A)*op(B) + beta*C,
void DenDen(const CMatrix& iA, bool tranAs, const CMatrix& iB, bool tranBs, 
	double alpha, CMatrix& oC, double beta)
{
	if (false == tranAs)
	{
		if (false == tranBs)
		{
			const long long m = iA.rows;
			const long long n = oC.cols;
			const long long k = iA.cols;
			const char cTraA = 'N';
			const char cTraB = 'N';

			dgemm(&cTraA, &cTraB, &m, &n, &k, &alpha,
				iA.pData, &m, iB.pData, &k, &beta, oC.pData, &m);
		}
		else
		{
		}
	}
	else
	{
		if (false == tranBs)
		{
			const long long m = iA.cols;
			const long long n = oC.cols;
			const long long k = iA.rows;
			const char cTraA = 'T';
			const char cTraB = 'N';

			dgemm(&cTraA, &cTraB, &m, &n, &k, &alpha,
				iA.pData, &k, iB.pData, &k, &beta, oC.pData, &m);
		}
		else
		{
		}
	}
}

void DenDiag(CMatrix& U, const CVector& d)
{
	double* pU = U.pData;
	for (long long r = 0; r < d.length; r++)
	{
		cblas_dscal(U.rows, d.pData[r], pU, 1);

		pU = pU + U.rows;
	}
}

// make up partially observed matrix from given positions 
void MakePartUV(const CMatrix& iU, const CMatrix& iV,
	const long long* pRowInd, const long long* pColInd, double* oVal, const long long nnz)
{
	const long long rnk = iU.cols;

	// column major storage
	for (long long p = 0; p < nnz; p++)
	{
		// index in txt file starts from 1
		long long iUpos = pRowInd[p] - 1;
		long long iVpos = pColInd[p] - 1;

		oVal[p] = cblas_ddot(rnk, iU.pData + iUpos, iU.rows, iV.pData + iVpos, iV.rows);
	}
}

// P(O - U V^T)
void UpdateToCSR(double* partUV, const double* pData, const long long length)
{
	const long long incx = 1;
	double alpha = -1.0;
	cblas_daxpy(length, alpha, pData, incx, partUV, incx);
	cblas_dscal(length, alpha, partUV, incx);
}

// memory inside this function should be optimize
void ReducedMat(const CsrSparse& spa, 
	const CMatrix& U1, const CMatrix& V1,
	const CMatrix& Ut, const CMatrix& Vt, CMatrix& oS,
	CMatrix& uStmp, CMatrix& vStmp)
{
	DenDen(V1, true, Vt, false, 1, vStmp, 0);
	DenDen(Ut, true, U1, false, 1, uStmp, 0);
	DenDen(uStmp, false, vStmp, false, 1, oS, 0);

	double* paUtemp = new double[Ut.rows];
	long long inx = 1;

	for (long long c = 0; c < oS.cols; c++)
	{
		double* pVa = Vt.pData + c*Vt.rows;
		
		SpaVec(spa, pVa, 1, paUtemp, 0, false);
		
		for (long long r = 0; r < oS.rows; r++)
		{
			double* pUa = Ut.pData + r*Ut.rows;
			double rst = ddot(&Ut.rows, pUa, &inx, paUtemp, &inx);

			oS.pData[r + c*oS.rows] = oS.pData[r + c*oS.rows] + rst;
		}
	}

	delete[] paUtemp;
}

void ReducedMat(const CsrSparse& spa,
	const CMatrix& U1, const CMatrix& V1, const double beta1,
	const CMatrix& U0, const CMatrix& V0, const double beta0,
	const CMatrix& Ut, const CMatrix& Vt,
	CMatrix& oS, CMatrix& uStmp, CMatrix& vStmp)
{
	DenDen(V1, true, Vt, false, 1, vStmp, 0);
	DenDen(Ut, true, U1, false, 1, uStmp, 0);
	DenDen(uStmp, false, vStmp, false, beta1, oS, 0);

	DenDen(V0, true, Vt, false, 1, vStmp, 0);
	DenDen(Ut, true, U0, false, 1, uStmp, 0);
	DenDen(uStmp, false, vStmp, false, beta0, oS, 1);

	/* CMatrix tempU = InitMat(U1.rows, U1.cols);

	SpaDen(spa, Vt, 1, tempU, 0, false);
	DenDen(Ut, true, tempU, false, 1, oS, 1);

	FreeMat(tempU);*/

	double* paUtemp = new double[Ut.rows];
	long long inx = 1;

	for (long long c = 0; c < oS.cols; c++)
	{
		double* pVa = Vt.pData + c*Vt.rows;

		SpaVec(spa, pVa, 1, paUtemp, 0, false);

		for (long long r = 0; r < oS.rows; r++)
		{
			double* pUa = Ut.pData + r*Ut.rows;
			double rst = ddot(&Ut.rows, pUa, &inx, paUtemp, &inx);

			oS.pData[r + c*oS.rows] = oS.pData[r + c*oS.rows] + rst;
		}
	}

	delete[] paUtemp;
}

CMatrix GenGaussian(long long iRows, long long iCols)
{
	std::default_random_engine generator(0);
	std::normal_distribution<double> distribution(0.0, 1.0);

	CMatrix cMat = {};
	cMat.rows = iRows;
	cMat.cols = iCols;
	cMat.pData = new double[iRows*iCols];

	// generate a Gaussian matrix
	double* iterpData = cMat.pData;
	for (long long i = 0; i < iRows*iCols; i++)
	{
		*iterpData = distribution(generator);
		iterpData++;
	}

	return cMat;
}

vector<double*> getDataVec(const vector<CsrSparse> &csr)
{
	vector<double*> pData(csr.size());

	for (unsigned long long i = 0; i < pData.size(); i++)
	{
		pData[i] = csr[i].pData;
	}

	return pData;
}