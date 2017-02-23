// FaNCL.cpp : Defines the entry point for the console application.
//

#define _CRT_SECURE_NO_DEPRECATE

#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <algorithm>
#include <iostream>
#include <random>

#include "mmio.h"
#include "FaNCL.h"
#include "tools.h"

using namespace std;

CooSparse ReadData(const char* fileName)
{
	MM_typecode banner = { 0 };

	FILE *pfile = fopen(fileName, "r");

	long long iRst = mm_read_banner(pfile, &banner);

	CooSparse data = {0};

	long long iRowNum = 0;
	long long iColNum = 0;
	long long iNnzNum = 0;

	// get rows & cols & nnz
	iRst = mm_read_mtx_crd_size(pfile, &data.rows, &data.cols, &data.nnz);

	data.pRowInd = new long long [data.nnz];
	data.pColInd = new long long[data.nnz];
	data.pData = new double[data.nnz];

	// read data
	iRst = mm_read_mtx_crd_data(pfile, data.rows, data.cols, data.nnz, 
		data.pRowInd, data.pColInd, data.pData, banner);

	fclose(pfile);

	return data;
}

int main()
{
	std::string traFile = "D:/WebDisk/data/netflix.txt";
	CooSparse traData = ReadData(traFile.c_str());
	ZScoreNorm(traData.pData, traData.nnz);

	// std::string tstFile = "data/movielens10m-tst.txt";
	// CooSparse tstData = ReadData(tstFile.c_str());
	// ZScoreNorm(tstData.pData, tstData.nnz);

	CooSparse tstData = {};

	double lambda = 1;
	double theta = 100;
	FaNCL(traData, lambda, theta, CAP, 5, tstData, 1);

	FreeCooSparse(traData);
	FreeCooSparse(tstData);

	// std::cout << "pause" << std::endl;
	// double i;
	// std::cin >> i;

	return 0;
}
