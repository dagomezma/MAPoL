/*
 * newtonpf.cuh
 *
 *  Created on: 23/09/2015
 *      Author: Igor M. Araújo
 */

#ifndef NEWTONPF_CUH_
#define NEWTONPF_CUH_

#include <Eigen/SparseLU>
#include "util/quicksort.h"
#include "util/timer.h"

using namespace std;
using namespace Eigen;

__host__ double mkl_checkConvergence(
		Bus* buses,
		unsigned int* pv,
		unsigned int* pq,
		int nnzYbus,
		int* csrRowPtrYbus,
		int* csrColIndYbus,
		cuDoubleComplex* csrValYbus,
		cuDoubleComplex *V,
		double *F)
{
	double err = 0.0;
	#pragma omp parallel for
	for (int id = 0; id < H_NPV + H_NPQ; id++)
	{
		int i, indice;
		if (id < H_NPV)
		{
			i = id;
			indice = pv[i];
		}
		else
		{
			i = id - H_NPV;
			indice = pq[i];
		}

		cuDoubleComplex c = make_cuDoubleComplex(0, 0);
		for (int k = csrRowPtrYbus[indice] - BASE_INDEX, endFor = csrRowPtrYbus[indice + 1] - BASE_INDEX; k < endFor; k++) {
			int j = csrColIndYbus[k] - BASE_INDEX;
			c = cuCadd(c, cuCmul(csrValYbus[k], V[j]));
		}
		Bus l_bus = buses[indice];
		cuDoubleComplex pot = make_cuDoubleComplex(l_bus.P, l_bus.Q);
		cuDoubleComplex miss = cuCmul(V[indice], cuConj(c));

		miss = cuCsub(miss, pot);

		if (l_bus.type == l_bus.PV) {
			F[i] = cuCreal(miss);
			#pragma omp critical
			err = max(err, abs(cuCreal(miss)));
		}
		if (l_bus.type == l_bus.PQ) {
			F[H_NPV+ i] = cuCreal(miss);
			#pragma omp critical
			err = max(err, abs(cuCreal(miss)));
			F[H_NPV + H_NPQ + i] = cuCimag(miss);
			#pragma omp critical
			err = max(err, abs(cuCimag(miss)));
		}
	}
	return err;
}

__host__ void mkl_computeDiagIbus(
		int nnzYbus,
		int* csrRowPtrYbus,
		int* csrColIndYbus,
		cuDoubleComplex* csrValYbus,
		cuDoubleComplex* V,
		cuDoubleComplex* diagIbus)
{
	#pragma omp parallel for
	for (int i = 0; i < H_NBUS; i++)
	{
		double real = 0.0;
		double imag = 0.0;
		for(int k = csrRowPtrYbus[i] - BASE_INDEX, endFor = csrRowPtrYbus[i + 1] - BASE_INDEX; k < endFor; k++){
			int j = csrColIndYbus[k] - BASE_INDEX;
			cuDoubleComplex matrixAdmittance = csrValYbus[k];
			cuDoubleComplex voltage = V[j];
			real += cuCreal(matrixAdmittance) * cuCreal(voltage) - cuCimag(matrixAdmittance) * cuCimag(voltage);
			imag += cuCreal(matrixAdmittance) * cuCimag(voltage) + cuCimag(matrixAdmittance) * cuCreal(voltage);
		}
		diagIbus[i] = make_cuDoubleComplex(real, imag);
	}
}

__host__ void mkl_compuateJacobianMatrix(
		int nnzJ,
		int* d_cooRowJ,
		int* csrRowPtrJ,
		int* csrColIndJ,
		double* csrValJ,
		unsigned int* device_pq,
		unsigned int* device_pv,
		int nnzYbus,
		int* csrRowPtrYbus,
		int* csrColIndYbus,
		cuDoubleComplex* csrValYbus,
		cuDoubleComplex* diagIbus,
		cuDoubleComplex* V)
{
	#pragma omp parallel for
	for (int id = 0; id < nnzJ; id++)
	{
		int length = (H_NPV + H_NPQ);
		int i = d_cooRowJ[id];
		int j = csrColIndJ[id];
		int ii, jj;

		if (i < length) {
			ii = (i < H_NPV) ? device_pv[i] : device_pq[i - H_NPV];
		} else {
			ii = device_pq[i - H_NPV - H_NPQ];
		}
		if (j < length) {
			jj = (j < H_NPV) ? device_pv[j] : device_pq[j - H_NPV];
		} else {
			jj = device_pq[j - H_NPV - H_NPQ];
		}

		cuDoubleComplex admittance = make_cuDoubleComplex(0,0);
		for(int k = csrRowPtrYbus[ii] - BASE_INDEX, endFor = csrRowPtrYbus[ii + 1] - BASE_INDEX; k < endFor; k++)
		{
			if(jj == csrColIndYbus[k] - BASE_INDEX){
				admittance = csrValYbus[k];
				break;
			}
		}
		double admittanceReal = cuCreal(admittance);
		double admittanceImag = cuCimag(admittance);
		double magnitude_j = cuCreal(V[jj]);
		double angle_j = cuCimag(V[jj]);
		double IbusReal = ((ii == jj) ? cuCreal(diagIbus[ii]) : 0.0);
		double IbusImag = ((ii == jj) ? cuCimag(diagIbus[ii]) : 0.0);
		double magnitude_i = cuCreal(V[ii]);
		double angle_i = cuCimag(V[ii]);

		if (i < length)
		{
			if (j < length) {

				double real = admittanceReal * magnitude_j - admittanceImag * angle_j;
				double imag = admittanceReal * angle_j + admittanceImag * magnitude_j;
				csrValJ[id] = -angle_i * (IbusReal - real) - magnitude_i * (-IbusImag + imag);

			}
			else // if (j < length)
			{
				double abs = sqrt(magnitude_j * magnitude_j + angle_j * angle_j);

				double real = admittanceReal * magnitude_j / abs - admittanceImag * angle_j / abs;
				double imag = admittanceReal * angle_j / abs + admittanceImag * magnitude_j / abs;
				csrValJ[id] = magnitude_i * real - angle_i * -imag + IbusReal * magnitude_j / abs + IbusImag * angle_j / abs;
			}
		}
		else // if (i < length)
		{
			if (j < length)
			{
				double real = admittanceReal * magnitude_j - admittanceImag * angle_j;
				double imag = admittanceReal * angle_j + admittanceImag * magnitude_j;

				csrValJ[id] = -angle_i * (-IbusImag + imag) + magnitude_i * (IbusReal - real);

			}
			else //if (j < length)
			{
				double abs = sqrt(magnitude_j * magnitude_j + angle_j * angle_j);
				double real = admittanceReal * magnitude_j / abs - admittanceImag * angle_j / abs;
				double imag = admittanceReal * angle_j / abs + admittanceImag * magnitude_j / abs;

				csrValJ[id] = magnitude_i * -imag + angle_i * real + IbusReal * angle_j / abs + -IbusImag * magnitude_j / abs;
			}
		}
	}
}

__host__ void mkl_updateVoltage(
		unsigned int *pv,
		unsigned int *pq,
		cuDoubleComplex *V,
		double *dx)
{

	#pragma omp parallel for
	for (int id = 0; id < H_NPV + H_NPQ; id++)
	{
		int i;
		if (id < H_NPV)
		{
			i = pv[id];
			cuDoubleComplex voltage = V[i];
			V[i] = cuCmul(make_cuDoubleComplex(cuCabs(voltage), 0), cuCexp(make_cuDoubleComplex(0, cuCangle(voltage) - dx[id])));
		}
		else
		{
			i = pq[id - H_NPV];
			cuDoubleComplex voltage = V[i];
			V[i] = cuCmul(make_cuDoubleComplex(cuCabs(voltage) - dx[H_NPQ + id], 0),cuCexp(make_cuDoubleComplex(0,cuCangle(voltage) - dx[id])));
		}
	}

}

__host__ void mkl_computeNnzJacobianMatrix()
{
	// #1 Predict nonzero numbers of Matrix J
	for(int i = 0; i < H_NBUS; i++)
	{
		for(int k = csrRowPtrYbus[i] - BASE_INDEX; k < csrRowPtrYbus[i + 1] - BASE_INDEX; k++)
		{
			int j = csrColIndYbus[k] - BASE_INDEX;
			if(buses[i].type == Bus::PV && buses[j].type == Bus::PV)
			{
				nnzJ++;
			}
			if(buses[i].type == Bus::PV && buses[j].type == Bus::PQ)
			{
				nnzJ += 2;
			}
			if(buses[i].type == Bus::PQ && buses[j].type == Bus::PV)
			{
				nnzJ += 2;
			}
			if(buses[i].type == Bus::PQ && buses[j].type == Bus::PQ)
			{
				nnzJ += 4;
			}
		}
	}

	// #2 Compute indexes of Matrix J with nonzero numbers
	int *cooColJ;
	cooRowJ = (int*) malloc(sizeof(int) * nnzJ);
	cooColJ = (int*) malloc(sizeof(int) * nnzJ);
	csrValJ = (double*) MKL_malloc(sizeof(double) * nnzJ, 64);
	csrColIndJ = (int*) MKL_malloc(sizeof(int) * nnzJ, 64);
	int ptr = 0;
	for(int i = 0; i < H_NBUS; i++)
	{
		for(int k = csrRowPtrYbus[i] - BASE_INDEX; k < csrRowPtrYbus[i + 1] - BASE_INDEX; k++)
		{
			int j = csrColIndYbus[k] - BASE_INDEX;
			if(buses[i].type == Bus::PV && buses[j].type == Bus::PV)
			{
				cooRowJ[ptr] = buses[i].indicePVPQ;
				cooColJ[ptr] = buses[j].indicePVPQ;
				ptr++;
			}
			if(buses[i].type == Bus::PV && buses[j].type == Bus::PQ)
			{
				cooRowJ[ptr] = buses[i].indicePVPQ;
				cooColJ[ptr] = buses[j].indicePVPQ;
				ptr++;
				cooRowJ[ptr] = buses[i].indicePVPQ;
				cooColJ[ptr] = buses[j].indicePVPQ + H_NPQ;
				ptr++;
			}
			if(buses[i].type == Bus::PQ && buses[j].type == Bus::PV)
			{
				cooRowJ[ptr] = buses[i].indicePVPQ;
				cooColJ[ptr] = buses[j].indicePVPQ;
				ptr++;
				cooRowJ[ptr] = buses[i].indicePVPQ + H_NPQ;
				cooColJ[ptr] = buses[j].indicePVPQ;
				ptr++;
			}
			if(buses[i].type == Bus::PQ && buses[j].type == Bus::PQ)
			{
				cooRowJ[ptr] = buses[i].indicePVPQ;
				cooColJ[ptr] = buses[j].indicePVPQ;
				ptr++;
				cooRowJ[ptr] = buses[i].indicePVPQ + H_NPQ;
				cooColJ[ptr] = buses[j].indicePVPQ;
				ptr++;
				cooRowJ[ptr] = buses[i].indicePVPQ;
				cooColJ[ptr] = buses[j].indicePVPQ + H_NPQ;
				ptr++;
				cooRowJ[ptr] = buses[i].indicePVPQ + H_NPQ;
				cooColJ[ptr] = buses[j].indicePVPQ + H_NPQ;
				ptr++;
			}
		}
	}

	// #3 Sort Matrix J by ROW
	int info;
	int length = H_NPV + 2 * H_NPQ;
	int job[6];
	job[0] = 2;
	job[1] = 0;
	job[2] = 0;
	job[4] = nnzJ;
	job[5] = 0;
	MKL_DCSRCOO((const int*) &job,(const int*) &length, csrValJ, csrColIndJ,csrRowPtrJ, &nnzJ,csrValJ, cooRowJ, cooColJ, &info);
	if(info) {printf("MKL Library error in %s at %d", __FILE__, __LINE__);exit(1);}
	quickSort(cooRowJ, 0, nnzJ - 1);
	// #5 Clear Memory
	free(cooColJ);
}

__host__ void mkl_solver_MKL_DSS()
{
	int length = H_NPV + 2 * H_NPQ;
	_MKL_DSS_HANDLE_t handle;
	MKL_INT opt;
	opt  = MKL_DSS_MSG_LVL_WARNING;
//	opt += MKL_DSS_TERM_LVL_ERROR;
	opt += MKL_DSS_ZERO_BASED_INDEXING;
	MKL_INT result;
	result = DSS_CREATE(handle, opt); if(result != MKL_DSS_SUCCESS){printf("MKL Library error in %s at %d", __FILE__, __LINE__);exit(1);}
	MKL_INT opt_define = MKL_DSS_NON_SYMMETRIC;
	result = DSS_DEFINE_STRUCTURE(handle, opt_define, csrRowPtrJ, length, length, csrColIndJ, nnzJ);if(result != MKL_DSS_SUCCESS){printf("MKL Library error in %s at %d", __FILE__, __LINE__);exit(1);}
	int *perm = (int*) MKL_malloc(sizeof(int) * length, 64);
	for(int i = 0; i < length; i++){
		perm[i] = i;
	}
	MKL_INT opt_REORDER = MKL_DSS_AUTO_ORDER;
	result = DSS_REORDER(handle, opt_REORDER,perm);if(result != MKL_DSS_SUCCESS){printf("MKL Library error in %s at %d", __FILE__, __LINE__);exit(1);}
//	MKL_INT opt_REORDER2 = MKL_DSS_GET_ORDER;
//	result = DSS_REORDER(handle, opt_REORDER2,perm);if(result != MKL_DSS_SUCCESS){printf("MKL Library error in %s at %d", __FILE__, __LINE__);exit(1);}
	MKL_INT opt_FACTOR = MKL_DSS_POSITIVE_DEFINITE;
	result = DSS_FACTOR_REAL(handle, opt_FACTOR, csrValJ);if(result != MKL_DSS_SUCCESS){printf("MKL Library error in %s at %d", __FILE__, __LINE__);exit(1);}
	MKL_INT opt_DEFAULT = MKL_DSS_DEFAULTS;
	MKL_INT nrhs = 1;
	result = DSS_SOLVE_REAL(handle, opt_DEFAULT, F, nrhs, dx);if(result != MKL_DSS_SUCCESS){printf("MKL Library error in %s at %d", __FILE__, __LINE__);exit(1);}
	result = DSS_DELETE(handle, opt_DEFAULT);if(result != MKL_DSS_SUCCESS){printf("MKL Library error in %s at %d", __FILE__, __LINE__);exit(1);}
	MKL_free(perm);
}

__host__ void eigen_sparseLU_solver(){
	int length = H_NPV + 2 * H_NPQ;

	SparseMatrix<double> A(length, length);
	for(int i = 0; i < length; i++){
		for(int k = csrRowPtrJ[i]; k < csrRowPtrJ[i+1]; k++){
			int j = csrColIndJ[k];
			A.insert(i, j) = csrValJ[k];
		}
	}
	A.makeCompressed();
	SparseLU<SparseMatrix<double>, COLAMDOrdering<int> > solverA;
	solverA.compute(A);

	VectorXd B(length);
	for(int i = 0; i < length; i++){
		B(i) = F[i];
	}
	VectorXd X = solverA.solve(B);
	for(int i = 0; i < length; i++){
		dx[i] = X(i);
	}
}

__host__ bool mkl_newtonpf()
{
	double start;
	start =GetTimer();
	double err = mkl_checkConvergence(
			buses,
			pv,
			pq,
			nnzYbus,
			csrRowPtrYbus,
			csrColIndYbus,
			csrValYbus,
			V,
			F);
	timeTable[TIME_CHECKCONVERGENCE] += GetTimer() - start;


#ifdef DEBUG
	int length = H_NPV + 2 * H_NPQ;
	printf("F = \n");
	for(int i = 0; i < length; i++){
		double value = F[i];
		printf("\t(%d)\t->\t%.4e\n", i+1, value);
	}
#endif

	int iter = 0;
	bool converged = false;

	if (err < EPS) {
		converged = true;
	}

	while (!converged && iter < MAX_IT_NR) {
		iter++;
		start =GetTimer();
		mkl_computeDiagIbus(
				nnzYbus,
				csrRowPtrYbus,
				csrColIndYbus,
				csrValYbus,
				V,
				diagIbus);
		timeTable[TIME_COMPUTEDIAGIBUS] += GetTimer() - start;

#ifdef DEBUG
	printf("diagIbus = \n");
	for(int i = 0; i < H_NBUS; i++){
		cuDoubleComplex value = diagIbus[i];
		printf("%.4e %c %.4ei\n", value.x,((value.y < 0.0) ? '-' : '+'),((value.y < 0.0) ? -value.y : value.y));
	}
#endif

		if(nnzJ == 0)
		{
			start =GetTimer();
			mkl_computeNnzJacobianMatrix();
			timeTable[TIME_COMPUTENNZJACOBIANMATRIX] += GetTimer() - start;
		}
		start =GetTimer();
		mkl_compuateJacobianMatrix(
			nnzJ,
			cooRowJ,
			csrRowPtrJ,
			csrColIndJ,
			csrValJ,
			pq,
			pv,
			nnzYbus,
			csrRowPtrYbus,
			csrColIndYbus,
			csrValYbus,
			diagIbus,
			V);
		timeTable[TIME_COMPUTEJACOBIANMATRIX] += GetTimer() - start;

#ifdef DEBUG
	printf("J = \n");
	printf("\tCompressed Sparse Column(rows = %d, cols = %d, nnz = %d [%.2lf])\n",length, length,nnzJ, nnzJ * 100.0f / (length * length));
	for(int j = 0; j < length; j++){
		for(int i = 0; i < length; i++){
			for(int k = csrRowPtrJ[i]; k < csrRowPtrJ[i + 1]; k++){
				if(j == csrColIndJ[k]){
					double value = csrValJ[k];
					printf("\t(%d, %d)\t->\t%.4e\n", i+1, j+1, value);
					break;
				}
			}
		}
	}
#endif

		// compute update step ------------------------------------------------

		switch(H_LinearSolver){
		case MKL_DSS:
			start =GetTimer();
			mkl_solver_MKL_DSS();
			timeTable[TIME_SOLVER_MKL_DSS] += GetTimer() - start;
			break;
		case Eigen_SparseLU:
			start =GetTimer();
			eigen_sparseLU_solver();
			timeTable[TIME_SOLVER_MKL_DSS] += GetTimer() - start;
			break;
		}

#ifdef DEBUG
	printf("dx = \n");
	for(int i = 0; i < length; i++){
		double value = dx[i];
		printf("\t(%d)\t->\t%.4e\n", i+1, -value);
	}
#endif

	start =GetTimer();
		mkl_updateVoltage(
				pv,
				pq,
				V,
				dx);
		timeTable[TIME_UPDATEVOLTAGE] += GetTimer() - start;

#ifdef DEBUG
	printf("V = \n");
	for(int i = 0; i < H_NBUS; i++)
	{
		printf("%.4e %c %.4ei\n", V[i].x, ((V[i].y < 0) ? '-' : '+'), ((V[i].y < 0) ? -V[i].y : V[i].y));
	}
#endif

	start =GetTimer();
	err = mkl_checkConvergence(
			buses,
			pv,
			pq,
			nnzYbus,
			csrRowPtrYbus,
			csrColIndYbus,
			csrValYbus,
			V,
			F);
	timeTable[TIME_CHECKCONVERGENCE] += GetTimer() - start;


#ifdef DEBUG
		printf("F = \n");
		for(int i = 0; i < length; i++){
			double value = F[i];
			printf("\t(%d)\t->\t%.4e\n", i+1, value);
		}
#endif

		if (err < EPS) {
			converged = true;
		}
	}
	return converged;
}


__global__ void hybrid_checkConvergence(
		int nTest,
		Bus* buses,
		unsigned int* pv,
		unsigned int* pq,
		int nnzYbus,
		int* csrRowPtrYbus,
		int* csrColIndYbus,
		cuDoubleComplex* csrValYbus,
		cuDoubleComplex *V,
		double *F) {

	int id = ID();
	if (id < D_NPV + D_NPQ) {
		int i, indice;
		if (id < D_NPV) {
			i = id;
			indice = pv[i];
		} else {
			i = id - D_NPV;
			indice = pq[i];
		}

		cuDoubleComplex c = make_cuDoubleComplex(0, 0);
		for (int k = csrRowPtrYbus[indice], endFor = csrRowPtrYbus[indice + 1]; k < endFor; k++) {
			int j = csrColIndYbus[k];
			c = cuCadd(c, cuCmul(csrValYbus[k], V[j]));
		}
		Bus l_bus = buses[indice];
		cuDoubleComplex pot = make_cuDoubleComplex(l_bus.P, l_bus.Q);
		cuDoubleComplex miss = cuCmul(V[indice], cuConj(c));

		miss = cuCsub(miss, pot);

		if (l_bus.type == l_bus.PV) {
			F[i] = cuCreal(miss);
		}
		if (l_bus.type == l_bus.PQ) {
			F[D_NPV + i ] = cuCreal(miss);
			F[D_NPV + D_NPQ + i] = cuCimag(miss);
		}
	}
}

__global__ void hybrid_computeDiagIbus(
		int test,
		int nnzYbus,
		int* csrRowPtrYbus,
		int* csrColIndYbus,
		cuDoubleComplex* csrValYbus,
		cuDoubleComplex* V,
		cuDoubleComplex* diagIbus)
{
	double real = 0.0;
	double imag = 0.0;
	int i = ID();
	if (i < D_NBUS) {
		for(int k = csrRowPtrYbus[i], endFor = csrRowPtrYbus[i + 1]; k < endFor; k++){
			int j = csrColIndYbus[k];
			cuDoubleComplex matrixAdmittance = csrValYbus[k];
			cuDoubleComplex voltage = V[j];
			real += cuCreal(matrixAdmittance) * cuCreal(voltage) - cuCimag(matrixAdmittance) * cuCimag(voltage);
			imag += cuCreal(matrixAdmittance) * cuCimag(voltage) + cuCimag(matrixAdmittance) * cuCreal(voltage);
		}
		diagIbus[i] = make_cuDoubleComplex(real, imag);
	}
}

__global__ void hybrid_compuateJacobianMatrix(
		int test,
		int nnzJ,
		int* d_cooRowJ,
		int* csrRowPtrJ,
		int* csrColIndJ,
		double* csrValJ,
		unsigned int* device_pq,
		unsigned int* device_pv,
		int nnzYbus,
		int* csrRowPtrYbus,
		int* csrColIndYbus,
		cuDoubleComplex* csrValYbus,
		cuDoubleComplex* diagIbus,
		cuDoubleComplex* V)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	if (id < nnzJ)
	{
		int length = (D_NPV + D_NPQ);
		int i = d_cooRowJ[id];
		int j = csrColIndJ[id];
		int ii, jj;

		if (i < length) {
			ii = (i < D_NPV) ? device_pv[i] : device_pq[i - D_NPV];
		} else {
			ii = device_pq[i - D_NPV - D_NPQ];
		}
		if (j < length) {
			jj = (j < D_NPV) ? device_pv[j] : device_pq[j - D_NPV];
		} else {
			jj = device_pq[j - D_NPV - D_NPQ];
		}

		cuDoubleComplex admittance;
		for(int k = csrRowPtrYbus[ii], endFor = csrRowPtrYbus[ii + 1]; k < endFor; k++)
		{
			if(jj == csrColIndYbus[k]){
				admittance = csrValYbus[k];
				break;
			}
		}
		double admittanceReal = cuCreal(admittance);
		double admittanceImag = cuCimag(admittance);
		double magnitude_j = cuCreal(V[jj]);
		double angle_j = cuCimag(V[jj]);
		double IbusReal = ((ii == jj) ? cuCreal(diagIbus[ii]) : 0.0);
		double IbusImag = ((ii == jj) ? cuCimag(diagIbus[ii]) : 0.0);
		double magnitude_i = cuCreal(V[ii]);
		double angle_i = cuCimag(V[ii]);

		if (i < length)
		{
			if (j < length) {

				double real = admittanceReal * magnitude_j - admittanceImag * angle_j;
				double imag = admittanceReal * angle_j + admittanceImag * magnitude_j;

				csrValJ[id] = -angle_i * (IbusReal - real) - magnitude_i * (-IbusImag + imag);

			}
			else // if (j < length)
			{
				double abs = sqrt(magnitude_j * magnitude_j + angle_j * angle_j);

				double real = admittanceReal * magnitude_j / abs - admittanceImag * angle_j / abs;
				double imag = admittanceReal * angle_j / abs + admittanceImag * magnitude_j / abs;

				csrValJ[id] = magnitude_i * real - angle_i * -imag + IbusReal * magnitude_j / abs + IbusImag * angle_j / abs;
			}
		}
		else // if (i < length)
		{
			if (j < length)
			{
				double real = admittanceReal * magnitude_j - admittanceImag * angle_j;
				double imag = admittanceReal * angle_j + admittanceImag * magnitude_j;

				csrValJ[id] = -angle_i * (-IbusImag + imag) + magnitude_i * (IbusReal - real);

			}
			else //if (j < length)
			{
				double abs = sqrt(magnitude_j * magnitude_j + angle_j * angle_j);
				double real = admittanceReal * magnitude_j / abs - admittanceImag * angle_j / abs;
				double imag = admittanceReal * angle_j / abs + admittanceImag * magnitude_j / abs;

				csrValJ[id] = magnitude_i * -imag + angle_i * real + IbusReal * angle_j / abs + -IbusImag * magnitude_j / abs;
			}
		}
	}
}

__global__ void hybrid_updateVoltage(
		int test,
		unsigned int *pv,
		unsigned int *pq,
		cuDoubleComplex *V,
		double *dx)
{
	int id = ID();
	int i;
	if (id < D_NPV + D_NPQ) {
		if (id < D_NPV) {
			i = pv[id];
			cuDoubleComplex voltage = V[i];
			V[i] = cuCmul(make_cuDoubleComplex(cuCabs(voltage), 0), cuCexp(make_cuDoubleComplex(0, cuCangle(voltage) - dx[id])));
		} else {
			i = pq[id - D_NPV];
			cuDoubleComplex voltage = V[i];
			V[i] = cuCmul(make_cuDoubleComplex(cuCabs(voltage) - dx[D_NPQ + id], 0),cuCexp(make_cuDoubleComplex(0,cuCangle(voltage) - dx[id])));
		}
	}
}

__host__ void hybrid_computeNnzJacobianMatrix()
{
	// #1 Predict nonzero numbers of Matrix J
	int *row, *col;
	row = (int*) malloc(sizeof(int) * (H_NBUS + 1));
	col = (int*) malloc(sizeof(int) * nnzYbus);
	checkCudaErrors(cudaMemcpy(row, csrRowPtrYbus, sizeof(int) * (H_NBUS + 1), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(col, csrColIndYbus, sizeof(int) * nnzYbus, cudaMemcpyDeviceToHost));
	for(int i = 0; i < H_NBUS; i++)
	{
		for(int k = row[i]; k < row[i + 1]; k++)
		{
			int j = col[k];
			if(buses[i].type == Bus::PV && buses[j].type == Bus::PV)
			{
				nnzJ++;
			}
			if(buses[i].type == Bus::PV && buses[j].type == Bus::PQ)
			{
				nnzJ += 2;
			}
			if(buses[i].type == Bus::PQ && buses[j].type == Bus::PV)
			{
				nnzJ += 2;
			}
			if(buses[i].type == Bus::PQ && buses[j].type == Bus::PQ)
			{
				nnzJ += 4;
			}
		}
	}

	// #2 Compute indexes of Matrix J with nonzero numbers
	int *cooRowJ, *cooColJ;
	cooRowJ = (int*) malloc(sizeof(int) * nnzJ);
	cooColJ = (int*) malloc(sizeof(int) * nnzJ);
	int ptr = 0;
	for(int i = 0; i < H_NBUS; i++)
	{
		for(int k = row[i]; k < row[i + 1]; k++)
		{
			int j = col[k];
			if(buses[i].type == Bus::PV && buses[j].type == Bus::PV)
			{
				cooRowJ[ptr] = buses[i].indicePVPQ;
				cooColJ[ptr] = buses[j].indicePVPQ;
				ptr++;
			}
			if(buses[i].type == Bus::PV && buses[j].type == Bus::PQ)
			{
				cooRowJ[ptr] = buses[i].indicePVPQ;
				cooColJ[ptr] = buses[j].indicePVPQ;
				ptr++;
				cooRowJ[ptr] = buses[i].indicePVPQ;
				cooColJ[ptr] = buses[j].indicePVPQ + H_NPQ;
				ptr++;
			}
			if(buses[i].type == Bus::PQ && buses[j].type == Bus::PV)
			{
				cooRowJ[ptr] = buses[i].indicePVPQ;
				cooColJ[ptr] = buses[j].indicePVPQ;
				ptr++;
				cooRowJ[ptr] = buses[i].indicePVPQ + H_NPQ;
				cooColJ[ptr] = buses[j].indicePVPQ;
				ptr++;
			}
			if(buses[i].type == Bus::PQ && buses[j].type == Bus::PQ)
			{
				cooRowJ[ptr] = buses[i].indicePVPQ;
				cooColJ[ptr] = buses[j].indicePVPQ;
				ptr++;
				cooRowJ[ptr] = buses[i].indicePVPQ + H_NPQ;
				cooColJ[ptr] = buses[j].indicePVPQ;
				ptr++;
				cooRowJ[ptr] = buses[i].indicePVPQ;
				cooColJ[ptr] = buses[j].indicePVPQ + H_NPQ;
				ptr++;
				cooRowJ[ptr] = buses[i].indicePVPQ + H_NPQ;
				cooColJ[ptr] = buses[j].indicePVPQ + H_NPQ;
				ptr++;
			}
		}
	}
	// #3 Sort Matrix J by ROW
	int *d_cooColJ;
	checkCudaErrors(cudaMalloc((void**) &d_cooColJ, sizeof(int) * nnzJ));
	if(d_cooRowJ == 0)
	{
		checkCudaErrors(cudaMalloc((void**) &d_cooRowJ, sizeof(int) * nnzJ));
		checkCudaErrors(cudaMalloc((void**) &csrColIndJ, sizeof(int) * nnzJ));
		checkCudaErrors(cudaMalloc((void**) &csrValJ, sizeof(double) * nnzJ * H_NTESTS));
	}
	checkCudaErrors(cudaMemcpy(d_cooColJ, cooColJ, sizeof(int) * nnzJ, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_cooRowJ, cooRowJ, sizeof(int) * nnzJ, cudaMemcpyHostToDevice));

	cusparseHandle_t handle;
	cusparseCreate(&handle);
	checkCudaErrors(cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST));
	int length = H_NPV + 2 * H_NPQ;
	size_t buffer = 0;
	void *pBuff;
	checkCudaErrors(cusparseXcoosort_bufferSizeExt(handle, length, length, nnzJ, d_cooRowJ, d_cooColJ, &buffer));
	checkCudaErrors(cudaMalloc((void**) &pBuff	, buffer * sizeof(char)));

	int *permu;
	checkCudaErrors(cudaMalloc((void**) &permu, nnzJ * sizeof(int)));

	checkCudaErrors(cusparseCreateIdentityPermutation(handle, nnzJ, permu));
	checkCudaErrors(cusparseXcoosortByRow(handle, length, length, nnzJ, d_cooRowJ, d_cooColJ, permu, pBuff));

	// #4 Convert Matrix J in Coordinate Format(COO) to Compressed Sparse Row Format(CSR)
	checkCudaErrors(cusparseXcoo2csr(handle, (const int*) d_cooRowJ, nnzJ, length, csrRowPtrJ, CUSPARSE_INDEX_BASE_ZERO));
	checkCudaErrors(cudaMemcpy(csrColIndJ, d_cooColJ, nnzJ * sizeof(int), cudaMemcpyDeviceToDevice));

	h_csrColIndJ = (int*) malloc(sizeof(int) * nnzJ);
	h_csrRowPtrJ = (int*) malloc(sizeof(int) * (length + 1));

	checkCudaErrors(cudaMemcpy(h_csrColIndJ, csrColIndJ, sizeof(int) * nnzJ, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(h_csrRowPtrJ, csrRowPtrJ, sizeof(int) * (length + 1), cudaMemcpyDeviceToHost));

	// #5 Clear Memory
	free(row);
	free(col);
	free(cooRowJ);
	free(cooColJ);
	checkCudaErrors(cudaFree(permu));
	checkCudaErrors(cudaFree(d_cooColJ));
	checkCudaErrors(cusparseDestroy(handle));


}

__host__ void linearSolverSp(int nTest) {
	int length = H_NPV + 2 * H_NPQ;
	for (int i = 0; i < nTest; i++) {
		cusolverSpHandle_t spHandle;
		csrluInfoHost_t info;

		checkCudaErrors(cusolverSpCreateCsrluInfoHost(&info));
		checkCudaErrors(cusolverSpCreate(&spHandle));

		cusparseMatDescr_t matDescA = 0;

		cusparseCreateMatDescr(&matDescA);
		cusparseSetMatType(matDescA, CUSPARSE_MATRIX_TYPE_GENERAL);
		cusparseSetMatIndexBase(matDescA, CUSPARSE_INDEX_BASE_ZERO);

		checkCudaErrors(cusolverSpXcsrluAnalysisHost(spHandle, length, nnzJ, matDescA, h_csrRowPtrJ, h_csrColIndJ, info));

		size_t size_internal;
		size_t size_lu;

		double *h_csrValJ;
		h_csrValJ = (double*) malloc(sizeof(double) * nnzJ);
		checkCudaErrors(cudaMemcpy(h_csrValJ, csrValJ, sizeof(double) * nnzJ, cudaMemcpyDeviceToHost));

		checkCudaErrors(cusolverSpDcsrluBufferInfoHost(spHandle, length, nnzJ, matDescA,h_csrValJ, h_csrRowPtrJ, h_csrColIndJ, info,&size_internal, &size_lu));

		char *buffer = (char*) malloc(size_lu * sizeof(char));
		int singularity = 0;
		const double tol = 1.e-14;
		const double pivot_threshold = 1.0;

		checkCudaErrors(cusolverSpDcsrluFactorHost(spHandle, length, nnzJ, matDescA,h_csrValJ, h_csrRowPtrJ, h_csrColIndJ, info, pivot_threshold, buffer));

		checkCudaErrors(cusolverSpDcsrluZeroPivotHost(spHandle, info, tol,&singularity));

		double *X1 = (double*) malloc(length * sizeof(double));
//		checkCudaErrors(cusolverSpDcsrluSolveHost(spHandle, n, B[i], X1[i], info,buffer));

		checkCudaErrors(cudaDeviceSynchronize());

		free(buffer);

		checkCudaErrors(cusolverSpDestroy(spHandle));
		checkCudaErrors(cusolverSpDestroyCsrluInfoHost(info));
		free(h_csrValJ);
	}
}

__host__ void solver_LS_with_RF()
{
	int length = H_NPV + 2 * H_NPQ;
	cusolverSpHandle_t spHandle;
	csrluInfoHost_t info;

	checkCudaErrors(cusolverSpCreateCsrluInfoHost(&info));
	checkCudaErrors(cusolverSpCreate(&spHandle));

	cusparseMatDescr_t matDescA = 0;

	cusparseCreateMatDescr(&matDescA);
	cusparseSetMatType(matDescA, CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatIndexBase(matDescA, CUSPARSE_INDEX_BASE_ZERO);

	checkCudaErrors(cusolverSpXcsrluAnalysisHost(
			spHandle,
			length,
			nnzJ,
			matDescA,
			h_csrRowPtrJ,
			h_csrColIndJ,
			info));

	size_t size_internal;
	size_t size_lu;

	double *h_csrValJ;
	h_csrValJ = (double*) malloc(sizeof(double) * nnzJ);
	checkCudaErrors(cudaMemcpy(h_csrValJ, csrValJ, sizeof(double) * nnzJ, cudaMemcpyDeviceToHost));

	checkCudaErrors(cusolverSpDcsrluBufferInfoHost(
			spHandle,
			length,
			nnzJ,
			matDescA,
			h_csrValJ,
			h_csrRowPtrJ,
			h_csrColIndJ,
			info,
			&size_internal,
			&size_lu));



	char *buffer = (char*) malloc(size_lu * sizeof(char));
	int singularity = 0;
	const double tol = 1.e-14;
	const double pivot_threshold = 1.0;

	checkCudaErrors(cusolverSpDcsrluFactorHost(
			spHandle,
			length,
			nnzJ,
			matDescA,
			h_csrValJ,
			h_csrRowPtrJ,
			h_csrColIndJ,
			info,
			pivot_threshold,
			buffer));

	checkCudaErrors(cusolverSpDcsrluZeroPivotHost(
			spHandle,
			info,
			tol,
			&singularity));

	double *h_F = (double*) malloc(length * sizeof(double));
	double *h_X = (double*) malloc(length * sizeof(double));
	checkCudaErrors(cudaMemcpy(h_F, F, length * sizeof(double), cudaMemcpyDeviceToHost));

	checkCudaErrors(cusolverSpDcsrluSolveHost(spHandle, length, h_F, h_X, info, buffer));

	checkCudaErrors(cudaMemcpy(F, h_X, length * sizeof(double), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaDeviceSynchronize());

	int nnzL;
	int nnzU;
	checkCudaErrors(cusolverSpXcsrluNnzHost(spHandle, &nnzL, &nnzU, info));

	int *h_P = (int*) malloc(sizeof(int) * length);
	int *h_Q = (int*) malloc(sizeof(int) * length);

	double *h_csrValL = (double*) malloc(sizeof(double) * nnzL);
	int *h_csrRowPtrL = (int*) malloc(sizeof(int) * (length + 1));
	int *h_csrColIndL = (int*) malloc(sizeof(int) * nnzL);

	double *h_csrValU = (double*) malloc(sizeof(double) * nnzU);
	int *h_csrRowPtrU = (int*) malloc(sizeof(int) * (length + 1));
	int *h_csrColIndU = (int*) malloc(sizeof(int) * nnzU);

	checkCudaErrors(cusolverSpDcsrluExtractHost(
			spHandle,
			h_P,
			h_Q,
			matDescA,
			h_csrValL,
			h_csrRowPtrL,
			h_csrColIndL,
			matDescA,
			h_csrValU,
			h_csrRowPtrU,
			h_csrColIndU,
			info,
			buffer));

	cusolverRfHandle_t rfHandle;
	checkCudaErrors(cusolverRfCreate(&rfHandle));

	checkCudaErrors(cusolverRfSetNumericProperties(rfHandle, 0.0, 0.0));

	checkCudaErrors(cusolverRfSetAlgs(
			rfHandle,
			CUSOLVERRF_FACTORIZATION_ALG0,
			CUSOLVERRF_TRIANGULAR_SOLVE_ALG1));

	checkCudaErrors(cusolverRfSetMatrixFormat(
			rfHandle,
			CUSOLVERRF_MATRIX_FORMAT_CSR,
			CUSOLVERRF_UNIT_DIAGONAL_ASSUMED_L));

	checkCudaErrors(cusolverRfSetResetValuesFastMode(
			rfHandle,
			CUSOLVERRF_RESET_VALUES_FAST_MODE_ON));

	int *d_P;
	int *d_Q;
	double *d_x;
	double *d_T;

	checkCudaErrors(cudaMalloc((void** ) &d_P, length * sizeof(int)));
	checkCudaErrors(cudaMalloc((void** ) &d_Q, length * sizeof(int)));
	checkCudaErrors(cudaMalloc((void** ) &d_x, length * sizeof(double)));
	checkCudaErrors(cudaMalloc((void** ) &d_T, length * sizeof(double)));

	checkCudaErrors(cusolverRfSetupHost(
			length,
			nnzJ,
			h_csrRowPtrJ,
			h_csrColIndJ,
			h_csrValJ,
			nnzL,
			h_csrRowPtrL,
			h_csrColIndL,
			h_csrValL,
			nnzU,
			h_csrRowPtrU,
			h_csrColIndU,
			h_csrValU,
			h_P,
			h_Q,
			rfHandle));

	checkCudaErrors(cudaDeviceSynchronize());

	checkCudaErrors(cusolverRfAnalyze(rfHandle));


	for (int i = 1; i < H_NTESTS; i++)
	{
		checkCudaErrors(cudaMemcpy(d_P, h_P, sizeof(int) * length, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_Q, h_Q, sizeof(int) * length, cudaMemcpyHostToDevice));

		checkCudaErrors(cusolverRfResetValues(length, nnzJ, csrRowPtrJ, csrColIndJ, csrValJ + nnzJ * i, d_P, d_Q, rfHandle));

		checkCudaErrors(cudaDeviceSynchronize());

		checkCudaErrors(cusolverRfRefactor(rfHandle));

		checkCudaErrors(cusolverRfSolve(rfHandle, d_P, d_Q, 1, d_T, length, F + length * i, length));
	}
		checkCudaErrors(cudaDeviceSynchronize());

		checkCudaErrors(cusolverRfDestroy(rfHandle));

		checkCudaErrors(cudaFree(d_P));
		checkCudaErrors(cudaFree(d_Q));
		checkCudaErrors(cudaFree(d_T));
		checkCudaErrors(cudaFree(d_x));

		free(h_csrValJ);
		free(h_P);
		free(h_Q);
		free(h_csrValL);
		free(h_csrColIndL);
		free(h_csrRowPtrL);
		free(h_csrValU);
		free(h_csrRowPtrU);
		free(h_csrColIndU);
		free(buffer);

		checkCudaErrors(cusolverSpDestroy(spHandle));
		checkCudaErrors(cusolverSpDestroyCsrluInfoHost(info));
}

__host__ void hybrid_solver_MKL_DSS()
{
	static int length = H_NPV + 2 * H_NPQ;

	static double *h_csrValJ = new double[nnzJ * H_NTESTS];
	static double *h_F = new double[length * H_NTESTS];
	static double *h_X = new double[length * H_NTESTS];

	double start = GetTimer();
	checkCudaErrors(cudaMemcpy(h_csrValJ, csrValJ, sizeof(double) * nnzJ * H_NTESTS, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(h_F, F, length * sizeof(double) * H_NTESTS, cudaMemcpyDeviceToHost));
	timeTable[TIME_D2H_MEM_COPY] += GetTimer() - start;

	//#pragma omp parallel for
	for(int t = 0; t < H_NTESTS; t++)
	{
/*		double *h_csrValJ = (double*) malloc(sizeof(double) * nnzJ);
		double *h_F = (double*) malloc(length * sizeof(double));
		double *h_X = (double*) malloc(length * sizeof(double));
*/
		_MKL_DSS_HANDLE_t handle;
		MKL_INT opt;
		opt  = MKL_DSS_MSG_LVL_WARNING;
	//	opt += MKL_DSS_TERM_LVL_ERROR;
		opt += MKL_DSS_ZERO_BASED_INDEXING;
		MKL_INT result;
		result = DSS_CREATE(handle, opt); if(result != MKL_DSS_SUCCESS){printf("MKL Library error in %s at %d", __FILE__, __LINE__);exit(1);}
		MKL_INT opt_define = MKL_DSS_NON_SYMMETRIC;
		result = DSS_DEFINE_STRUCTURE(handle, opt_define, h_csrRowPtrJ, length, length, h_csrColIndJ, nnzJ);if(result != MKL_DSS_SUCCESS){printf("MKL Library error in %s at %d", __FILE__, __LINE__);exit(1);}
		int *perm = (int*) MKL_malloc(sizeof(int) * length, 64);
		for(int i = 0; i < length; i++){
			perm[i] = i;
		}
		MKL_INT opt_REORDER = MKL_DSS_AUTO_ORDER;
		result = DSS_REORDER(handle, opt_REORDER,perm);if(result != MKL_DSS_SUCCESS){printf("MKL Library error in %s at %d", __FILE__, __LINE__);exit(1);}
	//	MKL_INT opt_REORDER2 = MKL_DSS_GET_ORDER;
	//	result = DSS_REORDER(handle, opt_REORDER2,perm);if(result != MKL_DSS_SUCCESS){printf("MKL Library error in %s at %d", __FILE__, __LINE__);exit(1);}

/*		start = GetTimer();
		checkCudaErrors(cudaMemcpy(h_csrValJ, csrValJ + t * nnzJ, sizeof(double) * nnzJ, cudaMemcpyDeviceToHost));
		timeTable[TIME_D2H_MEM_COPY] += GetTimer() - start;
*/
		MKL_INT opt_FACTOR = MKL_DSS_POSITIVE_DEFINITE;
		result = DSS_FACTOR_REAL(handle, opt_FACTOR, h_csrValJ + t * nnzJ);if(result != MKL_DSS_SUCCESS){printf("MKL Library error in %s at %d", __FILE__, __LINE__);exit(1);}
		MKL_INT opt_DEFAULT = MKL_DSS_DEFAULTS;
		MKL_INT nrhs = 1;

/*		start = GetTimer();
		checkCudaErrors(cudaMemcpy(h_F, F + t * length, length * sizeof(double), cudaMemcpyDeviceToHost));
		timeTable[TIME_D2H_MEM_COPY] += GetTimer() - start;
*/
		result = DSS_SOLVE_REAL(handle, opt_DEFAULT, h_F + t * length, nrhs, h_X + t * length);if(result != MKL_DSS_SUCCESS){printf("MKL Library error in %s at %d", __FILE__, __LINE__);exit(1);}
		result = DSS_DELETE(handle, opt_DEFAULT);if(result != MKL_DSS_SUCCESS){printf("MKL Library error in %s at %d", __FILE__, __LINE__);exit(1);}
		MKL_free(perm);

/*		start = GetTimer();
		checkCudaErrors(cudaMemcpy(F + t * length, h_X, length * sizeof(double), cudaMemcpyHostToDevice));
		timeTable[TIME_H2D_MEM_COPY] += GetTimer() - start;

		free(h_csrValJ);
		free(h_F);
		free(h_X);
*/
	}
	start = GetTimer();
	checkCudaErrors(cudaMemcpy(F, h_X, length * sizeof(double) * H_NTESTS, cudaMemcpyHostToDevice));
	timeTable[TIME_H2D_MEM_COPY] += GetTimer() - start;

/*	free(h_csrValJ);
	free(h_F);
	free(h_X);
*/
}


__host__ void hybrid_eigen_sparseLU_solver(){
	int length = H_NPV + 2 * H_NPQ;

	#pragma omp parallel for
	for(int t = 0; t < H_NTESTS; t++)
	{
		double *h_csrValJ = (double*) malloc(sizeof(double) * nnzJ);
		double *h_F = (double*) malloc(length * sizeof(double));
		double *h_X = (double*) malloc(length * sizeof(double));

		checkCudaErrors(cudaMemcpy(h_csrValJ, csrValJ + t * nnzJ, sizeof(double) * nnzJ, cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(h_F, F + t * length, length * sizeof(double), cudaMemcpyDeviceToHost));

		SparseMatrix<double> A(length, length);
		for(int i = 0; i < length; i++){
			for(int k = h_csrRowPtrJ[i]; k < h_csrRowPtrJ[i+1]; k++){
				int j = h_csrColIndJ[k];
				A.insert(i, j) = h_csrValJ[k];
			}
		}
		A.makeCompressed();
		SparseLU<SparseMatrix<double>, COLAMDOrdering<int> > solverA;
		solverA.compute(A);

		VectorXd B(length);
		for(int i = 0; i < length; i++){
			B(i) = h_F[i];
		}
		VectorXd X = solverA.solve(B);
		for(int i = 0; i < length; i++){
			h_X[i] = X(i);
		}
		checkCudaErrors(cudaMemcpy(F + t * length, h_X, length * sizeof(double), cudaMemcpyHostToDevice));
		free(h_csrValJ);
		free(h_F);
		free(h_X);
	}
}

__host__ void hybrid_newtonpf()
{

	int length = H_NPV + 2 * H_NPQ;
	double err[H_NTESTS];
	double start;
	start = GetTimer();
	for(int t = 0; t < H_NTESTS; t++)
	{
		hybrid_checkConvergence<<<BLOCKS((H_NPV + H_NPQ), H_THREADS), H_THREADS, 0, stream[t]>>>(
				t,
				device_buses,
				device_pv,
				device_pq,
				nnzYbus,
				csrRowPtrYbus,
				csrColIndYbus,
				csrValYbus + t * nnzYbus,
				V + t * H_NBUS,
				F + t * length);
	}
	checkCudaErrors(cudaDeviceSynchronize());


#ifdef DEBUG
	for (int t = 0; t < H_NTESTS; t++)
	{
		double *h_val = (double*) malloc(sizeof(double) * length);
		cudaMemcpy(h_val, F + length * t, sizeof(double) * length, cudaMemcpyDeviceToHost);
		printf("F[%d] = \n", t);
		for(int i = 0; i < length; i++){
			double value = h_val[i];
			printf("\t(%d)\t->\t%.4e\n", i+1, value);
		}
		free(h_val);
	}
#endif

	int iter = 0;
	bool converged = true;
	double* h_F = (double*) malloc(sizeof(double) * length * H_NTESTS);
	checkCudaErrors(cudaMemcpy(h_F, F, sizeof(double) * length * H_NTESTS, cudaMemcpyDeviceToHost));
	for(int t = 0; t < H_NTESTS; t++)
	{
		err[t] = 0.0;
		for(int i = 0; i < length; i++){
			err[t] = max(err[t], abs(h_F[i + length * t]));
		}
		if (err[t] < EPS) {
			converged_test[t] = true;
		} else {
			converged_test[t] = false;
		}
		converged &= converged_test[t];
	}
	timeTable[TIME_CHECKCONVERGENCE] += GetTimer() - start;

	while (!converged && iter < MAX_IT_NR) {
		iter++;
		start = GetTimer();
		for(int t = 0; t < H_NTESTS && !converged_test[t]; t++)
		{
			hybrid_computeDiagIbus<<<BLOCKS(H_NBUS, H_THREADS), H_THREADS, 0, stream[t]>>>
					(t,
					nnzYbus,
					csrRowPtrYbus,
					csrColIndYbus,
					csrValYbus + t * nnzYbus,
					V + t * H_NBUS,
					diagIbus + t * H_NBUS);
		}
		cudaDeviceSynchronize();
		timeTable[TIME_COMPUTEDIAGIBUS] += GetTimer() - start;

#ifdef DEBUG
	for (int t = 0; t < H_NTESTS; t++)
	{
		cuDoubleComplex *h_val = (cuDoubleComplex*) malloc(sizeof(cuDoubleComplex) * H_NBUS);
		cudaMemcpy(h_val, diagIbus + H_NBUS * t, sizeof(cuDoubleComplex) * H_NBUS, cudaMemcpyDeviceToHost);
		printf("diagIbus[%d] = \n", t);
		for(int i = 0; i < H_NBUS; i++){
			cuDoubleComplex value = h_val[i];
			printf("\t(%d)\t->\t%.4e %c %.4ei\n", i+1, value.x,((value.y < 0.0) ? '-' : '+'),((value.y < 0.0) ? -value.y : value.y));
		}
		free(h_val);
	}
#endif

		if(nnzJ == 0)
		{
			start = GetTimer();
			hybrid_computeNnzJacobianMatrix();
			timeTable[TIME_COMPUTENNZJACOBIANMATRIX] += GetTimer() - start;
		}
		start = GetTimer();
		for(int t = 0; t < H_NTESTS && !converged_test[t]; t++)
		{
		hybrid_compuateJacobianMatrix<<<BLOCKS(nnzJ, H_THREADS), H_THREADS, 0, stream[t]>>>(
				t,
				nnzJ,
				d_cooRowJ,
				csrRowPtrJ,
				csrColIndJ,
				csrValJ + t * nnzJ,
				device_pq,
				device_pv,
				nnzYbus,
				csrRowPtrYbus,
				csrColIndYbus,
				csrValYbus + t * nnzYbus,
				diagIbus + t * H_NBUS,
				V + t * H_NBUS);
		}
		cudaDeviceSynchronize();
		timeTable[TIME_COMPUTEJACOBIANMATRIX] += GetTimer() - start;

#ifdef DEBUG
	for (int t = 0; t < H_NTESTS; t++)
	{
		int *h_row = (int*) malloc(sizeof(int) * (length + 1));
		int *h_col = (int*) malloc(sizeof(int) * nnzJ);
		double *h_val = (double*) malloc(sizeof(double) * nnzJ);
		cudaMemcpy(h_row, csrRowPtrJ, sizeof(int) * (length + 1), cudaMemcpyDeviceToHost);
		cudaMemcpy(h_col, csrColIndJ, sizeof(int) * nnzJ, cudaMemcpyDeviceToHost);
		cudaMemcpy(h_val, csrValJ + nnzJ * t, sizeof(double) * nnzJ, cudaMemcpyDeviceToHost);
		printf("J[%d] = \n", t);
		printf("\tCompressed Sparse Column(rows = %d, cols = %d, nnz = %d [%.2lf])\n",
					length, length,
					nnzJ, nnzJ * 100.0f / (length * length));
		for(int j = 0; j < length; j++){
			for(int i = 0; i < length; i++){
				for(int k = h_row[i]; k < h_row[i + 1]; k++){
					if(j == h_col[k]){
						double value = h_val[k];
						printf("\t(%d, %d)\t->\t%.4e\n", i+1, j+1, value);
						break;
					}
				}
			}
		}
		free(h_row);
		free(h_col);
		free(h_val);
	}
#endif

		// compute update step ------------------------------------------------

//		solver_LS_with_RF();

		switch(H_LinearSolver){
		case MKL_DSS:
			start = GetTimer();
			hybrid_solver_MKL_DSS();
			timeTable[TIME_SOLVER_MKL_DSS] += GetTimer() - start;
			break;
		case Eigen_SparseLU:
			start = GetTimer();
			hybrid_eigen_sparseLU_solver();
			timeTable[TIME_SOLVER_MKL_DSS] += GetTimer() - start;
			break;
		case cuSolver:
			start = GetTimer();
			//linearSolverSp(H_NTESTS);
			solver_LS_with_RF();
			timeTable[TIME_SOLVER_MKL_DSS] += GetTimer() - start;
			break;
		}



#ifdef DEBUG
	for (int t = 0; t < H_NTESTS; t++)
	{
		double *h_val = (double*) malloc(sizeof(double) * length);
		cudaMemcpy(h_val, F + length * t, sizeof(double) * length, cudaMemcpyDeviceToHost);
		printf("dx[%d] = \n", t);
		for(int i = 0; i < length; i++){
			double value = h_val[i];
			printf("\t(%d)\t->\t%.4e\n", i+1, -value);
		}
		free(h_val);
	}
#endif
	start = GetTimer();
		for(int t = 0; t < H_NTESTS; t++)
		{
			hybrid_updateVoltage<<<BLOCKS((H_NPV + H_NPQ), H_THREADS), H_THREADS, 0, stream[t]>>>(
					t,
					device_pv,
					device_pq,
					V  + t * H_NBUS,
					F  + t * length);
		}
		cudaDeviceSynchronize();
		timeTable[TIME_UPDATEVOLTAGE] += GetTimer() - start;

#ifdef DEBUG
	checkCudaErrors(cudaDeviceSynchronize());
	for (int t = 0; t < H_NTESTS; t++)
	{
		cuDoubleComplex *h_V = (cuDoubleComplex*) malloc(sizeof(cuDoubleComplex) * H_NBUS);
		cudaMemcpy(h_V, V + H_NBUS * t, sizeof(cuDoubleComplex) * H_NBUS, cudaMemcpyDeviceToHost);
		printf("V[%d] = \n", t);
		for(int i = 0; i < H_NBUS; i++)
		{
			printf("\t[%d] -> %.4e %c %.4ei\n",i , h_V[i].x, ((h_V[i].y < 0) ? '-' : '+'), ((h_V[i].y < 0) ? -h_V[i].y : h_V[i].y));
		}
		free(h_V);
	}
#endif

		start = GetTimer();
		for(int t = 0; t < H_NTESTS; t++)
		{
			hybrid_checkConvergence<<<BLOCKS((H_NPV + H_NPQ), H_THREADS), H_THREADS, 0, stream[t]>>>(
					t,
					device_buses,
					device_pv,
					device_pq,
					nnzYbus,
					csrRowPtrYbus,
					csrColIndYbus,
					csrValYbus  + t * nnzYbus,
					V  + t * H_NBUS,
					F  + t * length);
		}
		checkCudaErrors(cudaDeviceSynchronize());
		timeTable[TIME_COMPUTE_POWER] += GetTimer() - start;

#ifdef DEBUG
	for (int t = 0; t < H_NTESTS; t++)
	{
		double *h_val = (double*) malloc(sizeof(double) * length);
		cudaMemcpy(h_val, F + length * t, sizeof(double) * length, cudaMemcpyDeviceToHost);
		printf("F[%d] = \n", t);
		for(int i = 0; i < length; i++){
			double value = h_val[i];
			printf("\t(%d)\t->\t%.4e\n", i+1, value);
		}
		free(h_val);
	}
#endif
		start = GetTimer();
	    converged = true;
		checkCudaErrors(cudaMemcpy(h_F, F, sizeof(double) * length * H_NTESTS, cudaMemcpyDeviceToHost));
		for(int t = 0; t < H_NTESTS; t++)
		{
			err[t] = 0.0;
			for(int i = 0; i < length; i++)
			{
				err[t] = max(err[t], h_F[i + length * t]);
			}
			if (err[t] < EPS)
			{
				converged_test[t] = true;
			}
			else // if (err[t] < EPS)
			{
				converged_test[t] = false;
			}
			converged &= converged_test[t];
		}
		timeTable[TIME_CHECKCONVERGENCE] += GetTimer() - start;
	}
	free(h_F);
}
#endif /* NEWTONPF_CUH_ */

