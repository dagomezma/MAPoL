/*
 * makYbus.cuh
 *
 *  Created on: 23/09/2015
 *      Author: Igor M. Araújo
 */

#ifndef MAKEYBUS_CUH_
#define MAKEYBUS_CUH_

#include <util/complexUtils.h>
#include <iostream>

using namespace std;

__host__ void mkl_computeCfCt(
		Branch *branches,
		cuDoubleComplex *cooValCf,
		int *cooRowCf,
		int *cooColCf,
		cuDoubleComplex *cooValCt,
		int *cooRowCt,
		int *cooColCt)
{
	#pragma omp parallel for
	for (int id = 0; id < H_NBRANCH; id++)
	{
		Branch l_branch = branches[id];
		cooValCf[id] = make_cuDoubleComplex(1, 0);
		cooRowCf[id] = l_branch.from + BASE_INDEX;
		cooColCf[id] = id + BASE_INDEX;

		cooValCt[id] = make_cuDoubleComplex(1, 0);
		cooRowCt[id] = l_branch.to + BASE_INDEX;
		cooColCt[id] = id + BASE_INDEX;
	}
}

__host__ void mkl_computeYfYt(
		Bus *buses,
		Branch *branches,
		cuDoubleComplex *csrValYt,
		int *csrRowPtrYt,
		int *csrColIndYt,
		cuDoubleComplex *csrValYf,
		int *csrRowPtrYf,
		int *csrColIndYf,
		cuDoubleComplex *csrValYsh,
		int *csrRowPtrYsh,
		int *csrColIndYsh,
		vector<pso::Particula::Estrutura> estrutura,
		pso::Particula particula) {
	#pragma omp parallel for
	for (int id  = 0; id < H_NBRANCH; id++) {
		if (id < H_NBUS) {
			Bus l_bus = buses[id];
			double Bsh = (l_bus.indiceEstrutura != -1 && estrutura[l_bus.indiceEstrutura].tipo == pso::Particula::Estrutura::SHC) ? particula[l_bus.indiceEstrutura] : l_bus.Bsh ;
			csrValYsh[id] = make_cuDoubleComplex(l_bus.Gsh, Bsh);
			csrRowPtrYsh[id] = id + BASE_INDEX;
			csrColIndYsh[id] = id + BASE_INDEX;
		}
		cuDoubleComplex Ytt;
		cuDoubleComplex Yff;
		cuDoubleComplex Yft;
		cuDoubleComplex Ytf;
		Branch l_branch = branches[id];

		int stat = (l_branch.inservice) ? 1 : 0;
		cuDoubleComplex impedance = make_cuDoubleComplex(l_branch.R, l_branch.X);
		cuDoubleComplex Ys = cuCdiv(make_cuDoubleComplex(stat, 0), impedance);
		cuDoubleComplex susceptance = make_cuDoubleComplex(0, l_branch.B);
		cuDoubleComplex Bc = cuCmul(make_cuDoubleComplex(stat, 0), susceptance);
		cuDoubleComplex tap = (l_branch.tap != 0) ? make_cuDoubleComplex(particula[l_branch.indiceEstrutura], 0) : make_cuDoubleComplex(1, 0);
		cuDoubleComplex phase_shifter = make_cuDoubleComplex(0, M_PI / 180.0 * l_branch.shift);
		tap = cuCmul(tap, cuCexp(phase_shifter));
		Ytt = cuCadd(Ys, cuCdiv(Bc, make_cuDoubleComplex(2, 0)));
		Yff = cuCdiv(Ytt, cuCmul(tap, cuConj(tap)));
		Yft = cuCdiv(cuCmul(Ys, make_cuDoubleComplex(-1, 0)), cuConj(tap));
		Ytf = cuCdiv(cuCmul(Ys, make_cuDoubleComplex(-1, 0)), tap);

		int offsetTo, offsetFrom;

		csrRowPtrYf[id] = id * 2  + BASE_INDEX;
		offsetTo = (l_branch.from > l_branch.to) ? 0 : 1;
		offsetFrom = 1 - offsetTo;
		csrColIndYf[id * 2 + offsetTo] = l_branch.to  + BASE_INDEX;
		csrValYf[id * 2 + offsetTo] = Yft;
		csrColIndYf[id * 2 + offsetFrom] = l_branch.from  + BASE_INDEX;
		csrValYf[id * 2 + offsetFrom] = Yff;

		csrRowPtrYt[id] = id * 2  + BASE_INDEX;
		offsetTo = (l_branch.from > l_branch.to) ? 0 : 1;
		offsetFrom = 1 - offsetTo;
		csrColIndYt[id * 2 + offsetTo] = l_branch.to + BASE_INDEX;
		csrValYt[id * 2 + offsetTo] = Ytt;
		csrColIndYt[id * 2 + offsetFrom] = l_branch.from + BASE_INDEX;
		csrValYt[id * 2 + offsetFrom] = Ytf;

		if(id == (H_NBRANCH -1)){
			id++;
			csrRowPtrYt[id] = id * 2 + BASE_INDEX;
			csrRowPtrYf[id] = id * 2 + BASE_INDEX;
			csrRowPtrYsh[H_NBUS] = H_NBUS + BASE_INDEX;
		}
	}
}

/* autor: Igor Araújo
 * Date : 03/02/2016
 * Description: Compute Admittance Matrix using a hybrid approach CPU and GPU, with cuSparse library.
 * */
__host__ void mkl_makeYbus(
		vector<pso::Particula::Estrutura> estrutura,
		pso::Particula particula,
		Bus* buses,
		Branch* branches)
{
	// #1 Matrix Cf and Ct is the same to All tests, so compute only once in the first time.
	// #1.1 Compute Matrix Cf and Ct in Coordinate Format (COO).
	mkl_computeCfCt(
			branches,
			cooValCf,
			cooRowCf,
			cooColCf,
			cooValCt,
			cooRowCt,
			cooColCt);
	// #1.2 Sort Matrix Cf by ROW
	// #1.3 Convert Matrix Cf in Coordinate Format(COO) to Compressed Sparse Row Format(CSR)
	int job[6];
	/*job - Array, contains the following conversion parameters:
	job[0]

	If job[0]=0, the matrix in the CSR format is converted to the coordinate format;
	if job[0]=1, the matrix in the coordinate format is converted to the CSR format.
	if job[0]=2, the matrix in the coordinate format is converted to the CSR format, and the column indices in CSR representation are sorted in the increasing order within each row.

	job[1]

	If job[1]=0, zero-based indexing for the matrix in CSR format is used;
	if job[1]=1, one-based indexing for the matrix in CSR format is used.

	job[2]

	If job[2]=0, zero-based indexing for the matrix in coordinate format is used;
	if job[2]=1, one-based indexing for the matrix in coordinate format is used.

	job[4]

	job[4]=nzmax - maximum number of the non-zero elements allowed if job[0]=0.

	job[5] - job indicator.

	For conversion to the coordinate format:
	If job[5]=1, only array rowind is filled in for the output storage.
	If job[5]=2, arrays rowind, colind are filled in for the output storage.
	If job[5]=3, all arrays rowind, colind, acoo are filled in for the output storage.
	For conversion to the CSR format:
	If job[5]=0, all arrays acsr, ja, ia are filled in for the output storage.
	If job[5]=1, only array ia is filled in for the output storage.
	If job[5]=2, then it is assumed that the routine already has been called with the job[5]=1, and the user allocated the required space for storing the output arrays acsr and ja.
	*/
	job[0] = 2;
	job[1] = BASE_INDEX;
	job[2] = BASE_INDEX;
	job[4] = nnzCf;
	job[5] = 0;
	int info;
	MKL_ZCSRCOO((const int*) &job,(const int*) &H_NBUS, csrValCf, csrColIndCf,csrRowPtrCf, &nnzCf,cooValCf, cooRowCf, cooColCf, &info);
	if(info) {printf("MKL Library error in %s at %d", __FILE__, __LINE__);exit(1);}

	// #1.4 Sort Matrix Ct by ROW
	// #1.5 Convert Matrix Ct in Coordinate Format(COO) to Compressed Sparse Row Format(CSR)
	job[0] = 2;
	job[1] = BASE_INDEX;
	job[2] = BASE_INDEX;
	job[4] = nnzCt;
	job[5] = 0;
	MKL_ZCSRCOO((const int*) &job,(const int*) &H_NBUS, csrValCt, csrColIndCt,csrRowPtrCt, &nnzCt,cooValCt, cooRowCt, cooColCt, &info);
	if(info) {printf("MKL Library error in %s at %d", __FILE__, __LINE__);exit(1);}

	// #2 Compute Matrix Yf and Yt
	mkl_computeYfYt(
			buses,
			branches,
			csrValYt,
			csrRowPtrYt,
			csrColIndYt,
			csrValYf,
			csrRowPtrYf,
			csrColIndYf,
			csrValYsh,
			csrRowPtrYsh,
			csrColIndYsh,
			estrutura,
			particula);

	// #3 Compute Admittance Matrix(Ybus) by equation Ybus = Cf * Yf + Ct * Yt + Ysh
	// #3.1 Compute Cf * Yf from equation
	{
		const char transa = 'N';const int request = 1;const int sort = 0;const int m = H_NBUS; const int n = H_NBRANCH; const int k = H_NBUS;const int nnz = 0;

		MKL_ZCSRMULTCSR( &transa, &request, &sort, &m, &n, &k, csrValCf, csrColIndCf, csrRowPtrCf, csrValYf, csrColIndYf, csrRowPtrYf,csrValCfYf, csrColIndCfYf, csrRowPtrCfYf, &nnz, &info);if(info) {printf("MKL Library error in %s at %d", __FILE__, __LINE__);exit(1);}
		nnzCfYf = csrRowPtrCfYf[m] - 1;

		csrColIndCfYf = (int*) MKL_malloc(sizeof(int) * nnzCfYf, 64);
		csrValCfYf = (cuDoubleComplex*) MKL_malloc(sizeof(cuDoubleComplex) * nnzCfYf, 64);
	}
	{
		const char transa = 'N';const int request = 2;const int sort = 0;const int m = H_NBUS; const int n = H_NBRANCH; const int k = H_NBUS;const int nnz = nnzCfYf;
		MKL_ZCSRMULTCSR( &transa, &request, &sort, &m, &n, &k, csrValCf, csrColIndCf, csrRowPtrCf, csrValYf, csrColIndYf, csrRowPtrYf,csrValCfYf, csrColIndCfYf, csrRowPtrCfYf, &nnz, &info);if(info) {printf("MKL Library error in %s at %d", __FILE__, __LINE__);exit(1);}
	}
	{
		const char transa = 'N';const int request = 0;const int sort = 0;const int m = H_NBUS; const int n = H_NBRANCH; const int k = H_NBUS;const int nnz = nnzCfYf;
		MKL_ZCSRMULTCSR( &transa, &request, &sort, &m, &n, &k, csrValCf, csrColIndCf, csrRowPtrCf, csrValYf, csrColIndYf, csrRowPtrYf,csrValCfYf, csrColIndCfYf, csrRowPtrCfYf, &nnz, &info);if(info) {printf("MKL Library error in %s at %d", __FILE__, __LINE__);exit(1);}
	}

	// #3.2 Compute Ct * Yt from equation
	{
		const char transa = 'N';const int request = 1;const int sort = 0;const int m = H_NBUS; const int n = H_NBRANCH; const int k = H_NBUS;const int nnz = nnzCtYt;
		MKL_ZCSRMULTCSR( &transa, &request, &sort, &m, &n, &k, csrValCt, csrColIndCt, csrRowPtrCt, csrValYt, csrColIndYt, csrRowPtrYt,csrValCtYt, csrColIndCtYt, csrRowPtrCtYt, &nnz, &info);if(info) {printf("MKL Library error in %s at %d", __FILE__, __LINE__);exit(1);}
		nnzCtYt = csrRowPtrCtYt[m] - 1;
		csrColIndCtYt = (int*) MKL_malloc(sizeof(int) * nnzCtYt, 64);
		csrValCtYt = (cuDoubleComplex*) MKL_malloc(sizeof(cuDoubleComplex) * nnzCtYt, 64);
	}
	{
		const char transa = 'N';const int request = 2;const int sort = 0;const int m = H_NBUS; const int n = H_NBRANCH; const int k = H_NBUS;const int nnz = nnzCtYt;
		MKL_ZCSRMULTCSR( &transa, &request, &sort, &m, &n, &k, csrValCt, csrColIndCt, csrRowPtrCt, csrValYt, csrColIndYt, csrRowPtrYt,csrValCtYt, csrColIndCtYt, csrRowPtrCtYt, &nnz, &info);if(info) {printf("MKL Library error in %s at %d", __FILE__, __LINE__);exit(1);}
	}
	{
		const char transa = 'N';const int request = 0;const int sort = 0;const int m = H_NBUS; const int n = H_NBRANCH; const int k = H_NBUS;const int nnz = nnzCtYt;
		MKL_ZCSRMULTCSR( &transa, &request, &sort, &m, &n, &k, csrValCt, csrColIndCt, csrRowPtrCt, csrValYt, csrColIndYt, csrRowPtrYt,csrValCtYt, csrColIndCtYt, csrRowPtrCtYt, &nnz, &info);if(info) {printf("MKL Library error in %s at %d", __FILE__, __LINE__);exit(1);}
	}

	// #3.3 Compute CfYf + CtYt from equation
	cuDoubleComplex scalar = make_cuDoubleComplex(1.0,0);
	{
		const char transa = 'N';const int request = 1;const int sort = 0;const int m = H_NBUS; const int n = H_NBUS;const int nnz = nnzCfYfCtYt;
		MKL_ZCSRADD( &transa, &request, &sort, &m, &n, csrValCfYf, csrColIndCfYf, csrRowPtrCfYf, &scalar, csrValCtYt, csrColIndCtYt, csrRowPtrCtYt, csrValCfYfCtYt, csrColIndCfYfCtYt, csrRowPtrCfYfCtYt, &nnz, &info);if(info) {printf("MKL Library error in %s at %d", __FILE__, __LINE__);exit(1);}
		nnzCfYfCtYt = csrRowPtrCfYfCtYt[m] - 1;
		csrColIndCfYfCtYt = (int*) MKL_malloc(sizeof(int) * nnzCfYfCtYt, 64);
		csrValCfYfCtYt = (cuDoubleComplex*) MKL_malloc(sizeof(cuDoubleComplex) * nnzCfYfCtYt, 64);
	}
	{
		const char transa = 'N';const int request = 2;const int sort = 0;const int m = H_NBUS; const int n = H_NBUS;const int nnz = nnzCfYfCtYt;
		MKL_ZCSRADD( &transa, &request, &sort, &m, &n, csrValCfYf, csrColIndCfYf, csrRowPtrCfYf, &scalar, csrValCtYt, csrColIndCtYt, csrRowPtrCtYt, csrValCfYfCtYt, csrColIndCfYfCtYt, csrRowPtrCfYfCtYt, &nnz, &info);if(info) {printf("MKL Library error in %s at %d", __FILE__, __LINE__);exit(1);}
	}
	{
		const char transa = 'N';const int request = 0;const int sort = 0;const int m = H_NBUS; const int n = H_NBUS;const int nnz = nnzCfYfCtYt;
		MKL_ZCSRADD( &transa, &request, &sort, &m, &n, csrValCfYf, csrColIndCfYf, csrRowPtrCfYf, &scalar, csrValCtYt, csrColIndCtYt, csrRowPtrCtYt, csrValCfYfCtYt, csrColIndCfYfCtYt, csrRowPtrCfYfCtYt, &nnz, &info);if(info) {printf("MKL Library error in %s at %d", __FILE__, __LINE__);exit(1);}
	}

	// #3.4 Compute CfYfCtYt + Ysh from equation
	{
		const char transa = 'N';const int request = 1;const int sort = 0;const int m = H_NBUS; const int n = H_NBUS;const int nnz = nnzYbus;
		MKL_ZCSRADD( &transa, &request, &sort, &m, &n, csrValCfYfCtYt, csrColIndCfYfCtYt, csrRowPtrCfYfCtYt, &scalar, csrValYsh, csrColIndYsh, csrRowPtrYsh, csrValYbus, csrColIndYbus, csrRowPtrYbus, &nnz, &info);if(info) {printf("MKL Library error in %s at %d", __FILE__, __LINE__);exit(1);}
		nnzYbus = csrRowPtrYbus[m] - 1;
		csrColIndYbus = (int*) MKL_malloc(sizeof(int) * nnzYbus, 64);
		csrValYbus = (cuDoubleComplex*) MKL_malloc(sizeof(cuDoubleComplex) * nnzYbus, 64);
	}
	{
		const char transa = 'N';const int request = 2;const int sort = 0;const int m = H_NBUS; const int n = H_NBUS;const int nnz = nnzYbus;
		MKL_ZCSRADD( &transa, &request, &sort, &m, &n, csrValCfYfCtYt, csrColIndCfYfCtYt, csrRowPtrCfYfCtYt, &scalar, csrValYsh, csrColIndYsh, csrRowPtrYsh, csrValYbus, csrColIndYbus, csrRowPtrYbus, &nnz, &info);if(info) {printf("MKL Library error in %s at %d", __FILE__, __LINE__);exit(1);}
	}
	{
		const char transa = 'N';const int request = 0;const int sort = 0;const int m = H_NBUS; const int n = H_NBUS;const int nnz = nnzYbus;
		MKL_ZCSRADD( &transa, &request, &sort, &m, &n, csrValCfYfCtYt, csrColIndCfYfCtYt, csrRowPtrCfYfCtYt, &scalar, csrValYsh, csrColIndYsh, csrRowPtrYsh, csrValYbus, csrColIndYbus, csrRowPtrYbus, &nnz, &info);if(info) {printf("MKL Library error in %s at %d", __FILE__, __LINE__);exit(1);}
	}

	MKL_free(csrColIndCfYf);
	MKL_free(csrValCfYf);
	MKL_free(csrColIndCtYt);
	MKL_free(csrValCtYt);
	MKL_free(csrColIndCfYfCtYt);
	MKL_free(csrValCfYfCtYt);
}

__global__ void hybrid_computeCfCt(
		Branch *branches,
		cuDoubleComplex *cooValCf,
		int *cooRowCf,
		int *cooColCf,
		cuDoubleComplex *cooValCt,
		int *cooRowCt,
		int *cooColCt)
{
	int id = ID();
	if (id < D_NBRANCH)
	{
		Branch l_branch = branches[id];

		cooValCf[id] = make_cuDoubleComplex(1, 0);
		cooRowCf[id] = l_branch.from;
		cooColCf[id] = id;

		cooValCt[id] = make_cuDoubleComplex(1, 0);
		cooRowCt[id] = l_branch.to;
		cooColCt[id] = id;

//		if(id == (D_NBRANCH -1 )){
//			id++;
//			cooColCt[id] = id;
//			cooColCf[id] = id;
//		}
	}
}

__global__ void hybrid_computeYfYt(Bus *buses, Branch *branches,
		cuDoubleComplex *csrValYt, int *csrRowPtrYt, int *csrColIndYt,
		cuDoubleComplex *csrValYf, int *csrRowPtrYf, int *csrColIndYf,
		cuDoubleComplex *csrValYsh, int *csrRowPtrYsh, int *csrColIndYsh,
		pso::Particula::Estrutura *d_estrutura, double *d_enxame) {
	int id = ID();
	if (id < D_NBRANCH) {
		if (id < D_NBUS) {
			Bus l_bus = buses[id];
			double Bsh = (l_bus.indiceEstrutura != -1 && d_estrutura[l_bus.indiceEstrutura].tipo == pso::Particula::Estrutura::SHC) ? d_enxame[l_bus.indiceEstrutura] : l_bus.Bsh ;
			csrValYsh[id] = make_cuDoubleComplex(l_bus.Gsh, Bsh);
			csrRowPtrYsh[id] = id;
			csrColIndYsh[id] = id;
		}
		cuDoubleComplex Ytt;
		cuDoubleComplex Yff;
		cuDoubleComplex Yft;
		cuDoubleComplex Ytf;
		Branch l_branch = branches[id];

		int stat = (l_branch.inservice) ? 1 : 0;
		cuDoubleComplex impedance = make_cuDoubleComplex(l_branch.R,
				l_branch.X);
		cuDoubleComplex Ys = cuCdiv(make_cuDoubleComplex(stat, 0), impedance);
		cuDoubleComplex susceptance = make_cuDoubleComplex(0, l_branch.B);
		cuDoubleComplex Bc = cuCmul(make_cuDoubleComplex(stat, 0), susceptance);
		cuDoubleComplex tap = (l_branch.tap != 0) ? ((l_branch.indiceEstrutura != -1) ? make_cuDoubleComplex(d_enxame[l_branch.indiceEstrutura], 0) : make_cuDoubleComplex(l_branch.tap, 0)) : make_cuDoubleComplex(1, 0);
		cuDoubleComplex phase_shifter = make_cuDoubleComplex(0,
				M_PI / 180.0 * l_branch.shift);
		tap = cuCmul(tap, cuCexp(phase_shifter));
		Ytt = cuCadd(Ys, cuCdiv(Bc, make_cuDoubleComplex(2, 0)));
		Yff = cuCdiv(Ytt, cuCmul(tap, cuConj(tap)));
		Yft = cuCdiv(cuCmul(Ys, make_cuDoubleComplex(-1, 0)), cuConj(tap));
		Ytf = cuCdiv(cuCmul(Ys, make_cuDoubleComplex(-1, 0)), tap);

		int offsetTo, offsetFrom;

		csrRowPtrYf[id] = id * 2;
		offsetTo = (l_branch.from > l_branch.to) ? 0 : 1;
		offsetFrom = 1 - offsetTo;
		csrColIndYf[id * 2 + offsetTo] = l_branch.to;
		csrValYf[id * 2 + offsetTo] = Yft;
		csrColIndYf[id * 2 + offsetFrom] = l_branch.from;
		csrValYf[id * 2 + offsetFrom] = Yff;

		csrRowPtrYt[id] = id * 2;
		offsetTo = (l_branch.from > l_branch.to) ? 0 : 1;
		offsetFrom = 1 - offsetTo;
		csrColIndYt[id * 2 + offsetTo] = l_branch.to;
		csrValYt[id * 2 + offsetTo] = Ytt;
		csrColIndYt[id * 2 + offsetFrom] = l_branch.from;
		csrValYt[id * 2 + offsetFrom] = Ytf;

		if(id == (D_NBRANCH -1)){
			id++;
			csrRowPtrYt[id] = id * 2;
			csrRowPtrYf[id] = id * 2;
			csrRowPtrYsh[D_NBUS] = D_NBUS;
		}
	}
}

/* autor: Igor Araújo
 * Date : 03/02/2016
 * Description: Compute Admittance Matrix using a hybrid approach CPU and GPU, with cuSparse library.
 * */
__host__ void hybrid_makeYbus(
		int nTest,
		int sizeEstrutura,
		Bus *buses,
		Branch *branches)
{
	// #1 Matrix Cf and Ct is the same to All tests, so compute only once in the first time.
	if (nTest == 0)
	{
		// #1.1 Compute Matrix Cf and Ct in Coordinate Format (COO).
		hybrid_computeCfCt<<<BLOCKS(H_NBRANCH, H_THREADS), H_THREADS, 0, stream[nTest]>>>(
				branches,
				cooValCf,
				cooRowCf,
				cooColCf,
				cooValCt,
				cooRowCt,
				cooColCt);

		// #1.2 Sort Matrix Cf by ROW
		size_t before = pBufferSizeInBytes;
		checkCudaErrors(cusparseXcoosort_bufferSizeExt(sparseHandle, H_NBUS, H_NBRANCH, nnzCfcoo, cooRowCf, cooColCf, &pBufferSizeInBytes));
		if(pBufferSizeInBytes > before){
			checkCudaErrors(cudaMalloc((void**) &pBuffer	, pBufferSizeInBytes * sizeof(char)));
		}
		checkCudaErrors(cusparseCreateIdentityPermutation(sparseHandle, nnzCfcoo, permutation));
		checkCudaErrors(cusparseXcoosortByRow(sparseHandle, H_NBUS, H_NBRANCH, nnzCfcoo, cooRowCf, cooColCf, permutation, pBuffer));
		checkCudaErrors(cusparseZgthr(sparseHandle, nnzCfcoo, cooValCf, csrValCf, permutation, CUSPARSE_INDEX_BASE_ZERO));

		// #1.3 Convert Matrix Cf in Coordinate Format(COO) to Compressed Sparse Row Format(CSR)
		checkCudaErrors(cusparseXcoo2csr(sparseHandle, (const int*) cooRowCf, nnzCf, H_NBUS, csrRowPtrCf, CUSPARSE_INDEX_BASE_ZERO));
		checkCudaErrors(cudaMemcpy(csrColIndCf, cooColCf, nnzCf * sizeof(int), cudaMemcpyDeviceToDevice));

		// #1.4 Sort Matrix Ct by ROW
		before = pBufferSizeInBytes;
		checkCudaErrors(cusparseXcoosort_bufferSizeExt(sparseHandle, H_NBUS, H_NBRANCH, nnzCtcoo, cooRowCt, cooColCt, &pBufferSizeInBytes));
		if(pBufferSizeInBytes > before){
			checkCudaErrors(cudaMalloc((void**) &pBuffer	, pBufferSizeInBytes * sizeof(char)));
		}
		checkCudaErrors(cusparseCreateIdentityPermutation(sparseHandle, nnzCtcoo, permutation));
		checkCudaErrors(cusparseXcoosortByRow(sparseHandle, H_NBUS, H_NBRANCH, nnzCtcoo, cooRowCt, cooColCt, permutation, pBuffer));
		checkCudaErrors(cusparseZgthr(sparseHandle, nnzCtcoo, cooValCt, csrValCt, permutation, CUSPARSE_INDEX_BASE_ZERO));

		// #1.5 Convert Matrix Ct in Coordinate Format(COO) to Compressed Sparse Row Format(CSR)
		checkCudaErrors(cusparseXcoo2csr(sparseHandle, (const int*) cooRowCt, nnzCt, H_NBUS, csrRowPtrCt, CUSPARSE_INDEX_BASE_ZERO));
		checkCudaErrors(cudaMemcpy(csrColIndCt, cooColCt, nnzCt * sizeof(int), cudaMemcpyDeviceToDevice));

	}
	// #2 Compute Matrix Yf and Yt
	hybrid_computeYfYt<<<BLOCKS(H_NBRANCH, H_THREADS), H_THREADS, 0, stream[nTest]>>>(
			buses,
			branches,
			csrValYt + nnzYt * nTest,
			csrRowPtrYt,
			csrColIndYt,
			csrValYf + nnzYf * nTest,
			csrRowPtrYf,
			csrColIndYf,
			csrValYsh + nnzYsh * nTest,
			csrRowPtrYsh,
			csrColIndYsh,
			d_estrutura,
			d_enxame + nTest * sizeEstrutura);


	// #3 Compute Admittance Matrix(Ybus) by equation Ybus = Cf * Yf + Ct * Yt + Ysh
	// #3.1 Compute Cf * Yf from equation
	if(nTest == 0)
	{
		checkCudaErrors(cusparseCreateMatDescr(&descrCf));
		checkCudaErrors(cusparseCreateMatDescr(&descrYf));
		checkCudaErrors(cusparseCreateMatDescr(&descrCfYf));
		checkCudaErrors(cusparseSetMatType(descrCf, CUSPARSE_MATRIX_TYPE_GENERAL));
		checkCudaErrors(cusparseSetMatType(descrYf, CUSPARSE_MATRIX_TYPE_GENERAL));
		checkCudaErrors(cusparseSetMatType(descrCfYf, CUSPARSE_MATRIX_TYPE_GENERAL));
		checkCudaErrors(cusparseXcsrgemmNnz(sparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, H_NBUS, H_NBUS, H_NBRANCH, descrCf, nnzCf, csrRowPtrCf, csrColIndCf, descrYf, nnzYf, csrRowPtrYf, csrColIndYf, descrCfYf, csrRowPtrCfYf, &nnzCfYf));
		checkCudaErrors(cudaMalloc((void**)&csrColIndCfYf	, sizeof(int)				* nnzCfYf));
		checkCudaErrors(cudaMalloc((void**)&csrValCfYf		, sizeof(cuDoubleComplex) 	* nnzCfYf));
		checkCudaErrors(cusparseZcsrgemm(sparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, H_NBUS, H_NBUS, H_NBRANCH, descrCf, nnzCf, csrValCf, csrRowPtrCf, csrColIndCf, descrYf, nnzYf, csrValYf, csrRowPtrYf, csrColIndYf, descrCfYf, csrValCfYf, csrRowPtrCfYf, csrColIndCfYf));
	}
	else
	{
		checkCudaErrors(cusparseZcsrgemm(sparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, H_NBUS, H_NBUS, H_NBRANCH, descrCf, nnzCf, csrValCf, csrRowPtrCf, csrColIndCf, descrCfYf, nnzYf, csrValYf + nnzYf * nTest, csrRowPtrYf, csrColIndYf, descrCfYf, csrValCfYf, csrRowPtrCfYf, csrColIndCfYf));
	}
	// #3.2 Compute Ct * Yt from equation
	if(nTest == 0)
	{
		checkCudaErrors(cusparseCreateMatDescr(&descrCt));
		checkCudaErrors(cusparseCreateMatDescr(&descrYt));
		checkCudaErrors(cusparseCreateMatDescr(&descrCtYt));
		checkCudaErrors(cusparseSetMatType(descrCt, CUSPARSE_MATRIX_TYPE_GENERAL));
		checkCudaErrors(cusparseSetMatType(descrYt, CUSPARSE_MATRIX_TYPE_GENERAL));
		checkCudaErrors(cusparseSetMatType(descrCtYt, CUSPARSE_MATRIX_TYPE_GENERAL));
		checkCudaErrors(cusparseXcsrgemmNnz(sparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, H_NBUS, H_NBUS, H_NBRANCH, descrCt, nnzCt, csrRowPtrCt, csrColIndCt, descrYt, nnzYt, csrRowPtrYt, csrColIndYt, descrCtYt, csrRowPtrCtYt, &nnzCtYt));
		checkCudaErrors(cudaMalloc((void**)&csrColIndCtYt	, sizeof(int)				* nnzCtYt));
		checkCudaErrors(cudaMalloc((void**)&csrValCtYt		, sizeof(cuDoubleComplex) 	* nnzCtYt));
		checkCudaErrors(cusparseZcsrgemm(sparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, H_NBUS, H_NBUS, H_NBRANCH, descrCt, nnzCt, csrValCt, csrRowPtrCt, csrColIndCt, descrYt, nnzYt, csrValYt, csrRowPtrYt, csrColIndYt, descrCtYt, csrValCtYt, csrRowPtrCtYt, csrColIndCtYt));
	}
	else
	{
		checkCudaErrors(cusparseZcsrgemm(sparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, H_NBUS, H_NBUS, H_NBRANCH, descrCt, nnzCt, csrValCt, csrRowPtrCt, csrColIndCt, descrCtYt, nnzYt, csrValYt + nnzYt * nTest, csrRowPtrYt, csrColIndYt, descrCtYt, csrValCtYt, csrRowPtrCtYt, csrColIndCtYt));
	}
	// #3.3 Compute CfYf + CtYt from equation
	if(nTest == 0)
	{
		checkCudaErrors(cusparseCreateMatDescr(&descrCfYfCtYt));
		checkCudaErrors(cusparseSetMatType(descrCfYfCtYt, CUSPARSE_MATRIX_TYPE_GENERAL));
		checkCudaErrors(cusparseXcsrgeamNnz(sparseHandle, H_NBUS, H_NBUS, descrCfYf, nnzCfYf, csrRowPtrCfYf, csrColIndCfYf, descrCtYt, nnzCtYt, csrRowPtrCtYt, csrColIndCtYt, descrCfYfCtYt, csrRowPtrCfYfCtYt, &nnzCfYfCtYt));
		checkCudaErrors(cudaMalloc((void**)&csrColIndCfYfCtYt	, sizeof(int)				* nnzCfYfCtYt));
		checkCudaErrors(cudaMalloc((void**)&csrValCfYfCtYt		, sizeof(cuDoubleComplex) 	* nnzCfYfCtYt));
		cuDoubleComplex fator = make_cuDoubleComplex(1,0);
		checkCudaErrors(cusparseZcsrgeam(sparseHandle, H_NBUS, H_NBUS, &fator, descrCfYf, nnzCfYf, (const cuDoubleComplex*)csrValCfYf, csrRowPtrCfYf, csrColIndCfYf, &fator, descrCtYt, nnzCtYt,(const cuDoubleComplex*) csrValCtYt, csrRowPtrCtYt, csrColIndCtYt, descrCfYfCtYt, csrValCfYfCtYt, csrRowPtrCfYfCtYt, csrColIndCfYfCtYt));

	}
	else
	{
		cuDoubleComplex fator = make_cuDoubleComplex(1,0);
		checkCudaErrors(cusparseZcsrgeam(sparseHandle, H_NBUS, H_NBUS, &fator, descrCfYf, nnzCfYf, (const cuDoubleComplex*)csrValCfYf, csrRowPtrCfYf, csrColIndCfYf, &fator, descrCtYt, nnzCtYt,(const cuDoubleComplex*) csrValCtYt, csrRowPtrCtYt, csrColIndCtYt, descrCfYfCtYt, csrValCfYfCtYt, csrRowPtrCfYfCtYt, csrColIndCfYfCtYt));
	}
	// #3.4 Compute CfYfCtYt + Ysh from equation
	if(nTest == 0)
	{
		checkCudaErrors(cusparseCreateMatDescr(&descrYsh));
		checkCudaErrors(cusparseCreateMatDescr(&descrYbus));
		checkCudaErrors(cusparseSetMatType(descrYsh, CUSPARSE_MATRIX_TYPE_GENERAL));
		checkCudaErrors(cusparseSetMatType(descrYbus, CUSPARSE_MATRIX_TYPE_GENERAL));

		checkCudaErrors(cusparseXcsrgeamNnz(sparseHandle, H_NBUS, H_NBUS, descrCfYfCtYt, nnzCfYfCtYt, csrRowPtrCfYfCtYt, csrColIndCfYfCtYt, descrYsh, nnzYsh, csrRowPtrYsh, csrColIndYsh, descrYbus, csrRowPtrYbus, &nnzYbus));
		checkCudaErrors(cudaMalloc((void**)&csrColIndYbus	, sizeof(int)				* nnzYbus			));
		checkCudaErrors(cudaMalloc((void**)&csrValYbus		, sizeof(cuDoubleComplex) 	* nnzYbus * H_NTESTS));
		cuDoubleComplex fator = make_cuDoubleComplex(1,0);
		checkCudaErrors(cusparseZcsrgeam(sparseHandle, H_NBUS, H_NBUS, &fator, descrCfYfCtYt, nnzCfYfCtYt, (const cuDoubleComplex*)csrValCfYfCtYt, csrRowPtrCfYfCtYt, csrColIndCfYfCtYt, &fator, descrYsh, nnzYsh,(const cuDoubleComplex*) csrValYsh, csrRowPtrYsh, csrColIndYsh, descrYbus, csrValYbus, csrRowPtrYbus, csrColIndYbus));
	}
	else
	{
		cuDoubleComplex fator = make_cuDoubleComplex(1,0);
		checkCudaErrors(cusparseZcsrgeam(sparseHandle, H_NBUS, H_NBUS, &fator, descrCfYfCtYt, nnzCfYfCtYt, (const cuDoubleComplex*)csrValCfYfCtYt, csrRowPtrCfYfCtYt, csrColIndCfYfCtYt, &fator, descrYsh, nnzYsh,(const cuDoubleComplex*) (csrValYsh + nnzYsh * nTest), csrRowPtrYsh, csrColIndYsh, descrYbus, csrValYbus + nnzYbus * nTest, csrRowPtrYbus, csrColIndYbus));
	}
}

#endif /* MAKEYBUS_CUH_ */
