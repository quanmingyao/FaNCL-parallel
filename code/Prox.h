enum RegType
{
	CAP = 1,
	LSP = 2,
	SCAD = 3,
	MCP = 4
};

void RegObj(double *regVal, const double *ix, long long lng,
	double lambda, double theta, RegType type);

void ProxStep(double *ox, const double *id, long long lng,
	double lambda, double theta, RegType type);