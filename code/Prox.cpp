#include <stdlib.h>
#include <stdio.h>
#include <algorithm>
#include "prox.h"

long long mymin(double *ox, long long n)
{
	double temp = ox[0];
	long long ind = 0;
	for (long long i = 1; i < n; i++)
	{
		double xi = ox[i];
		if (xi < temp)
		{
			ind = i;
			temp = xi;
		}
	}
	return ind;
}

void proxCapL1(double *ox, const double *id, long long n, double lambda, double theta)
{
	double u, x1, x2;
	for (long long i = 0; i < n; i++)
	{
		u = fabs(id[i]);
		x1 = std::max(u, theta);
		x2 = std::min(theta, std::max(0.0, u - lambda));
		if (0.5*(x1 + x2 - 2 * u)*(x1 - x2) + lambda*(theta - x2) < 0)
			ox[i] = x1;
		else
			ox[i] = x2;
		ox[i] = id[i] >= 0 ? ox[i] : -ox[i];

	}
	return;
}

void proxLSP(double *ox, const double *id, long long n, double lambda, double theta)
{
	double v, u, z, sqrtv;
	double xtemp[3], ytemp[3];
	xtemp[0] = 0.0;

	for (long long i = 0; i < n; i++)
	{
		const float di = (float)id[i];
		u = fabs(di);
		z = u - theta;
		v = z*z - 4.0*(lambda - u*theta);

		if (v < 0)
			ox[i] = 0;
		else
		{
			sqrtv = sqrt(v);
			xtemp[1] = std::max(0.0, 0.5*(z + sqrtv));
			xtemp[2] = std::max(0.0, 0.5*(z - sqrtv));

			ytemp[0] = 0.5*u*u;
			double tempi = xtemp[1] - u;

			ytemp[1] = 0.5*tempi*tempi + lambda*log(1.0 + xtemp[1] / theta);

			tempi = xtemp[2] - u;
			ytemp[2] = 0.5*tempi*tempi + lambda*log(1.0 + xtemp[2] / theta);

			tempi = xtemp[mymin(ytemp, 3)];
			ox[i] = (di >= 0) ? tempi : -tempi;
		}
	}
	return;
}

void proxSCAD(double *ox, const double *id, long long n, double lambda, double theta)
{
	double u, z, w;
	double xtemp[3], ytemp[3];
	z = theta*lambda;
	w = lambda*lambda;
	for (long long i = 0; i < n; i++)
	{
		u = fabs(id[i]);
		xtemp[0] = std::min(lambda, std::max(0.0, u - lambda));
		xtemp[1] = std::min(z, std::max(lambda, (u*(theta - 1.0) - z) / (theta - 2.0)));
		xtemp[2] = std::max(z, u);

		ytemp[0] = 0.5*(xtemp[0] - u)*(xtemp[0] - u) + lambda*xtemp[0];
		ytemp[1] = 0.5*(xtemp[1] - u)*(xtemp[1] - u) + 0.5*(xtemp[1] * (-xtemp[1] + 2 * z) - w) / (theta - 1.0);
		ytemp[2] = 0.5*(xtemp[2] - u)*(xtemp[2] - u) + 0.5*(theta + 1.0)*w;

		ox[i] = xtemp[mymin(ytemp, 3)];
		ox[i] = id[i] >= 0 ? ox[i] : -ox[i];

	}
	return;
}

void proxMCP(double *ox, const double *id, long long n, double lambda, double theta)
{
	double x1, x2, v, u, z;
	z = theta*lambda;
	if (theta > 1)
	{
		for (long long i = 0; i < n; i++)
		{
			u = fabs(id[i]);
			x1 = std::min(z, std::max(0.0, theta*(u - lambda) / (theta - 1.0)));
			x2 = std::max(z, u);
			if (0.5*(x1 + x2 - 2 * u)*(x1 - x2) + x1*(lambda - 0.5*x1 / theta) - 0.5*z*lambda < 0)
				ox[i] = x1;
			else
				ox[i] = x2;
			ox[i] = id[i] >= 0 ? ox[i] : -ox[i];
		}
	}
	else if (theta < 1)
	{
		for (long long i = 0; i<n; i++)
		{
			u = fabs(id[i]);
			v = theta*(u - lambda) / (theta - 1);
			x1 = fabs(v) > fabs(v - z) ? 0.0 : z;
			x2 = std::max(z, u);
			if (0.5*(x1 + x2 - 2 * u)*(x1 - x2) + x1*(lambda - 0.5*x1 / theta) - 0.5*z*lambda < 0)
				ox[i] = x1;
			else
				ox[i] = x2;
			ox[i] = id[i] >= 0 ? ox[i] : -ox[i];
		}
	}
	else
	{
		for (long long i = 0; i<n; i++)
		{
			u = fabs(id[i]);
			x1 = lambda > u ? 0.0 : z;
			x2 = std::max(z, u);
			if (0.5*(x1 + x2 - 2 * u)*(x1 - x2) + x1*(lambda - 0.5*x1 / theta) - 0.5*z*lambda < 0)
				ox[i] = x1;
			else
				ox[i] = x2;
			ox[i] = id[i] >= 0 ? ox[i] : -ox[i];
		}
	}
	return;
}

void ProxStep(double *ox, const double *id, long long lng,
	double lambda, double theta, RegType type)
{
	switch (type)
	{
	case 1:
		proxCapL1(ox, id, lng, lambda, theta);
		break;
	case 2:
		proxLSP(ox, id, lng, lambda, theta);
		break;
	case 3:
		proxSCAD(ox, id, lng, lambda, theta);
		break;
	case 4:
		proxMCP(ox, id, lng, lambda, theta);
		break;
	default:
		proxCapL1(ox, id, lng, lambda, theta);
	}
}

/* reg value */
void funCapL1(double *f, const double *x, long long n, double lambda, double theta)
{
	double u = 0.0;;
	for (long long i = 0; i < n; i++)
	{
		u += std::min(fabs(x[i]), theta);
	}
	*f = u*lambda;
	return;
}

void funLSP(double *f, const double *x, long long n, double lambda, double theta)
{
	double u = 0.0;

	for (long long i = 0; i < n; i++)
	{
		u += log(1.0 + fabs(x[i]) / theta);
	}
	*f = u*lambda;
	return;
}

void funSCAD(double *f, const double *x, long long n, double lambda, double theta)
{
	double u, v, y, z, w;
	y = theta*lambda;
	w = lambda*lambda;
	z = 0.5*(theta + 1.0)*w;

	u = 0.0;
	for (long long i = 0; i<n; i++)
	{
		v = fabs(x[i]);
		if (v <= lambda)
			u += lambda*v;
		else if (v > y)
			u += z;
		else
			u += 0.5*(v*(2 * y - v) - w) / (theta - 1.0);
	}

	*f = u;
	return;
}

void funMCP(double *f, const double *x, long long n, double lambda, double theta)
{
	double v, u, y;
	y = theta*lambda;
	u = 0.0;
	for (long long i = 0; i < n; i++)
	{
		v = fabs(x[i]);
		if (v <= y)
			u += v*(lambda - 0.5*v / theta);
		else
			u += 0.5*y*lambda;
	}
	*f = u;
	return;
}

void RegObj(double *regVal, const double *ix, long long lng,
	double lambda, double theta, RegType type)
{
	switch (type)
	{
	case 1:
		funCapL1(regVal, ix, lng, lambda, theta);
		break;
	case 2:
		funLSP(regVal, ix, lng, lambda, theta);
		break;
	case 3:
		funSCAD(regVal, ix, lng, lambda, theta);
		break;
	case 4:
		funMCP(regVal, ix, lng, lambda, theta);
		break;
	default:
		funCapL1(regVal, ix, lng, lambda, theta);
	}
}
