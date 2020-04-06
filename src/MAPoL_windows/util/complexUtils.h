/*
 * complexUtils.cuh
 *
 *  Created on: 28/09/2015
 *      Author: Igor M. Araújo
 */

#ifndef COMPLEXUTILS_CUH_
#define COMPLEXUTILS_CUH_

__host__ __device__ cuDoubleComplex cuCexp(cuDoubleComplex arg) {
	cuDoubleComplex res;
	double s, c;
	double e = exp(arg.x);
	sincos(arg.y, &s, &c);
	res.x = c * e;
	res.y = s * e;
	return res;
}

__host__ __device__ double cuCangle(cuDoubleComplex arg) {
	return atan(cuCimag(arg) / cuCreal(arg));
}

#endif /* COMPLEXUTILS_CUH_ */
