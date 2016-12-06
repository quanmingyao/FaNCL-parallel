#include <vector>

void SortIdx(long long* val, const long long length, long long* idx);

void SwapIdx(const long long* idx, const long long length, long long* val);

void SwapIdx(const long long* idx, const long long length, double* val);

std::vector<std::pair<long long, long long>> LinearSplit(const long long length, const long long splits);

double ZScoreNorm(double* val, const long long length);