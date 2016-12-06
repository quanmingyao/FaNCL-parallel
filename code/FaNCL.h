#include "matrix.h"
#include "Prox.h"

void FaNCL(CooSparse& data, double lambda, double theta, RegType rType, const long long maxRank,
	const CooSparse& tsData, bool isAcc);