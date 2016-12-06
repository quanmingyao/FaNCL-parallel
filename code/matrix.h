/*
CblasColMajor:
  Indicates that the matrices are stored in column major order, 
  with the elements of each column of the matrix stored contiguously.
*/

#include <vector>

struct CVector
{
	double* pData;
	long long length;
};

CVector InitVec(long long leg);

void FreeVec(CVector& pVec);

void PrintCVector(const CVector& vec, long long prec = 2);

struct CMatrix
{
	double* pData;
	long long rows;
	long long cols;
};

CMatrix InitMat(long long rows, long long cols);

void CopyMat(const CMatrix& fM, CMatrix& tM);

void FreeMat(CMatrix& oB);

void PrintCMatrix(const CMatrix& mat);


struct CooSparse
{
	double* pData;
	long long* pRowInd;
	long long* pColInd;

	long long nnz;
	long long rows;
	long long cols;

	long long staRow;
	long long staCol;
};

void PrintCooSparse(const CooSparse& mat);

void TransCooSparse(CooSparse& mat);

void FreeCooSparse(CooSparse& spa);

struct CsrSparse
{
	double* pData;
	long long* pColInd;
	long long* rowIndex;

	long long nnz;
	long long rows;
	long long cols;

	long long staRow;
	long long staCol;
};

CsrSparse InitCsrSparse(long long nnz, long long rows, long long cols);

void PrintCsrSparse(const CsrSparse& mat);

void FreeCsrSparse(CsrSparse& spa);

long long GenCsrFCoo(CooSparse& cooSpa, CsrSparse& csrSpa);

/*----------------------------------------------*/

void PowerMethod_spa(const CooSparse iSp, CMatrix& oU, CMatrix& ioV);

void SpaDen(const CooSparse& iA, const CMatrix& iB, double alpha, CMatrix& oC, double beta, bool trans);

void SpaDen(const CsrSparse& iA, const CMatrix& iB, double alpha, CMatrix& oC, double beta, bool trans);

void SpaVec(const CooSparse& iA, const CVector& iB, double alpha, CVector& oC, double beta, bool trans);

void SpaVec(const CooSparse& iA, double* iB, double alpha, double* oC, double beta, bool trans);

void SpLrDen(const CsrSparse& iSpa, const CMatrix& iU, const CMatrix& iV, bool trans, CMatrix& oiU, CMatrix& oiV, CMatrix& dpTpVM);

void SpLrDen(const CsrSparse& iSpa,
	const CMatrix& U1, const CMatrix& V1, const double beta1,
	const CMatrix& U0, const CMatrix& V0, const double beta0,
	bool trans,
	CMatrix& oiU, CMatrix& oiV,
	CMatrix& dpTpVM);

void QRfact(CMatrix& iA);

void DenDen(const CMatrix& iA, bool tranAs, const CMatrix& iB, bool tranBs, double alpha, CMatrix& oC, double beta);

void DenDiag(CMatrix& U, const CVector& d);

void MakePartUV(const CMatrix& iU, const CMatrix& iV, const long long* pRowInd, const long long* pColInd, double* oVal, const long long nnz);

void UpdateToCSR(double* partUV, const double* pData, const long long length);

void ReducedMat(const CsrSparse& spa,
	const CMatrix& U1, const CMatrix& V1,
	const CMatrix& aU, const CMatrix& aV, CMatrix& oS,
	CMatrix& uStmp, CMatrix& vStmp);

void ReducedMat(const CsrSparse& spa,
	const CMatrix& U1, const CMatrix& V1, const double beta1,
	const CMatrix& U0, const CMatrix& V0, const double beta0,
	const CMatrix& Ut, const CMatrix& Vt,
	CMatrix& oS, CMatrix& uStmp, CMatrix& vStmp);

void ReducedMat(const CsrSparse& spa, const CMatrix& U1, const CMatrix& V1, const double beta1, const CMatrix& U0, const CMatrix& V0, const double beta0, const CMatrix& Ut, const CMatrix& Vt, CMatrix& oS, CMatrix& uStmp, CMatrix& vStmp);
long long ReducedSVD(const CMatrix iMat, CMatrix& oU, CVector& oS, CMatrix& oV);

CMatrix GenGaussian(long long iRows, long long iCols);

std::vector<double*> getDataVec(const std::vector<CsrSparse> &csr);