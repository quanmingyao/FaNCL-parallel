#include <algorithm>

#include "mkl.h"
#include "tools.h"

using namespace std;

// used for sorting
template <class type>  class comIdx
{
public:
	comIdx() :m_val(0), m_idx(0) {}
	comIdx(const type val, const long long idx):m_val (val), m_idx(idx){}
	
	bool operator < (const comIdx <type>& input)
	{
		return m_val < input.m_val;
	}
	
	type m_val;
	long long m_idx;
};

void SortIdx(long long* val, const long long length, long long *idx)
{
	vector<comIdx<long long>> data(length);
	for (long long i = 0; i < length; i++)
	{
		data[i].m_val = val[i];
		data[i].m_idx = i;
	}

	sort(data.begin(), data.end());

	for (long long i = 0; i < length; i++)
	{
		val[i] = data[i].m_val;
		idx[i] = data[i].m_idx;
	}
}

// rearrange elements in val based on positions given by idx
void SwapIdx(const long long* idx, const long long length, long long* val)
{
	long long* tpVal = new long long[length];
	
	for (long long i = 0; i < length; i++)
	{
		tpVal[i] = val[idx[i]];
	}

	for (long long i = 0; i < length; i++)
	{
		val[i] = tpVal[i];
	}

	delete[] tpVal;
}

void SwapIdx(const long long* idx, const long long length, double* val)
{
	double* tpVal = new double[length];

	for (long long i = 0; i < length; i++)
	{
		tpVal[i] = val[idx[i]];
	}

	for (long long i = 0; i < length; i++)
	{
		val[i] = tpVal[i];
	}

	delete[] tpVal;
}

vector<pair<long long, long long>> LinearSplit(const long long length, const long long splits)
{
	// one base indexing
	vector <pair<long long, long long>> rst(splits);

	const long long rowJmp = (long long) floor(length / splits);
	for (long long s = 0; s < splits; s++)
	{
		long long sta = s;
		long long end = s + 1;

		sta = sta * rowJmp + 1;
		if (end == splits)
		{
			end = length;
		}
		else
		{
			end = end * rowJmp;
		}

		rst[s] = pair<long long, long long>(sta, end);
	}

	return rst;
}

double ZScoreNorm(double* val, const long long length)
{
	long long incx = 1;

	double mean = 0;
	for (long long i = 0; i < length; i++)
	{
		mean = mean + val[i];
	}
	mean = mean / length;

	for (long long i = 0; i < length; i++)
	{
		val[i] = val[i] - mean;
	}

	double nm = dnrm2(&length, val, &incx) / sqrt(length);

	for (long long i = 0; i < length; i++)
	{
		val[i] = val[i] / nm;
	}

	nm = 0;
	for (long long i = 0; i < length; i++)
	{
		nm = nm + val[i];
	}

	return nm;
}