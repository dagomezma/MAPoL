/*
 * runpf.cuh
 *
 *  Created on: 16/10/2015
 *      Author: Igor M. Araújo
 */

#ifndef RUNPF_CUH_
#define RUNPF_CUH_

#include <float.h>
#include <cuComplex.h>
#include <util/reduce.h>
#include "pso/Particula.h"

const int NR = 0;
const int FDXB = 1;
const int FDBX = 2;

const int MKL_DSS = 0;
const int Eigen_SimplicialLLT = 1;
const int Eigen_SimplicialLDLT = 2;
const int Eigen_ConjugateGradient = 3;
const int Eigen_BiCGSTAB = 4;
const int Eigen_SparseLU = 5;
const int Eigen_SparseQR = 6;
const int cuSolver = 6;

__constant__ int D_ALG;
__constant__ int D_NBUS;
__constant__ int D_NPV;
__constant__ int D_NPQ;
__constant__ int D_NBRANCH;
__constant__ int D_THREADS;

int H_ALG = NR;
int H_NBUS;
int H_NBRANCH;
int H_NPV;
int H_NPQ;
int H_THREADS;
int H_NTESTS;
int H_LinearSolver = MKL_DSS;

#define MAX_IT_NR 10 // Numero maximo de interações
#define MAX_IT_FD 30 // Numero maximo de interações
#define EPS 1e-8 // Erro aceitavel para condição de parada

#include <util/complexUtils.h>

cudaStream_t *stream = 0;

Bus *buses;
Branch *branches;
unsigned int *pv;
unsigned int *pq;

Bus *device_buses;
Branch *device_branches;
unsigned int *device_pv;
unsigned int *device_pq;

cuDoubleComplex *V;

int nnzYbus = 0;
cuDoubleComplex *csrValYbus;
int *csrRowPtrYbus;
int *csrColIndYbus;

int nnzYt = 0;
cuDoubleComplex *csrValYt;
int *csrRowPtrYt;
int *csrColIndYt;

int nnzYf = 0;
cuDoubleComplex *csrValYf;
int *csrRowPtrYf;
int *csrColIndYf;
int nnzYsh = 0;
cuDoubleComplex *csrValYsh;
int *csrRowPtrYsh;
int *csrColIndYsh;

int nnzCf = 0;
cuDoubleComplex *csrValCf;
int *csrRowPtrCf;
int *csrColIndCf;

int nnzCt = 0;
cuDoubleComplex *csrValCt;
int *csrRowPtrCt;
int *csrColIndCt;

int nnzCfcoo = 0;
cuDoubleComplex *cooValCf;
int *cooRowCf;
int *cooColCf;

int nnzCtcoo = 0;
cuDoubleComplex *cooValCt;
int *cooRowCt;
int *cooColCt;

size_t pBufferSizeInBytes = 0;

void *pBuffer;

int nPermutation;
int *permutation;

int nnzCfYf = 0;
cuDoubleComplex *csrValCfYf;
int *csrRowPtrCfYf;
int *csrColIndCfYf;

int nnzCtYt = 0;
cuDoubleComplex *csrValCtYt;
int *csrRowPtrCtYt;
int *csrColIndCtYt;

int nnzCfYfCtYt = 0;
cuDoubleComplex *csrValCfYfCtYt;
int *csrRowPtrCfYfCtYt;
int *csrColIndCfYfCtYt;

cusparseHandle_t sparseHandle;

cusparseMatDescr_t descrCf;
cusparseMatDescr_t descrYf;
cusparseMatDescr_t descrCfYf;

cusparseMatDescr_t descrCt;
cusparseMatDescr_t descrYt;
cusparseMatDescr_t descrCtYt;

cusparseMatDescr_t descrCfYfCtYt;

cusparseMatDescr_t descrYbus;
cusparseMatDescr_t descrYsh;

double *F;
double *dx;
cuDoubleComplex *diagIbus;

int nnzJ = 0;
double *csrValJ;
int *csrRowPtrJ;
int *csrColIndJ;
int *d_cooRowJ = 0;
int *cooRowJ = 0;

int *h_csrRowPtrJ;
int *h_csrColIndJ;

bool *converged_test;

double *vLoss;

Bus *tmpBuses;
Branch *tmpBranches;


double *csrBpVal;
int *csrBpCol;
int *csrBpRow;
double *cooBpVal;
int *cooBpCol;
int *cooBpRow;

double *csrBppVal;
int *csrBppCol;
int *csrBppRow;
double *cooBppVal;
int *cooBppCol;
int *cooBppRow;


int tmpNnzYbus = 0;
cuDoubleComplex *tmpCsrValYbus;
int *tmpCsrRowPtrYbus;
int *tmpCsrColIndYbus;

int tmpNnzYt = 0;
cuDoubleComplex *tmpCsrValYt;
int *tmpCsrRowPtrYt;
int *tmpCsrColIndYt;

int tmpNnzYf = 0;
cuDoubleComplex *tmpCsrValYf;
int *tmpCsrRowPtrYf;
int *tmpCsrColIndYf;

double *P;
double *Q;

pso::Particula::Estrutura *d_estrutura;
double *d_enxame  = 0;

double* dReduceLoss;
double* dtReduceLoss;
double* hReduceLoss;
int reduceBlocks;
int reduceThreads;
int reduceThreadsBlocks;

#include <powerflow/makeYbus.h>
#include <powerflow/makeB.h>
#include <powerflow/newtonpf.h>
#include <powerflow/fdpf.h>

__host__ void mkl_computeVoltage( Bus *buses, cuDoubleComplex *V,
		                          vector<pso::Particula::Estrutura> &estrutura,
		                          pso::Particula &particula )
{
	#pragma omp parallel for
	for (int id = 0; id < H_NBUS; id++) {
		Bus l_bus = buses[id];
		double Vbus = ( l_bus.indiceEstrutura != -1 && estrutura[l_bus.indiceEstrutura].tipo == pso::Particula::Estrutura::AVR ) ? particula[l_bus.indiceEstrutura] :  l_bus.V ;
		V[id] = cuCmul(        make_cuDoubleComplex(Vbus, 0),
				        cuCexp(make_cuDoubleComplex(0, l_bus.O)) );
		if (l_bus.type == l_bus.PV || l_bus.type == l_bus.SLACK) {
			V[id] = cuCmul( make_cuDoubleComplex(Vbus / cuCabs(V[id]), 0.0),
					        V[id] );
		}
	}
}

__host__ double mkl_computeLoss(
		Branch *branches,
		cuDoubleComplex *V,
		int nnzYf,
		int* csrRowPtrYf,
		int* csrColIndYf,
		cuDoubleComplex* csrValYf,
		int nnzYt,
		int* csrRowPtrYt,
		int* csrColIndYt,
		cuDoubleComplex* csrValYt) {
	double sumLoss = 0.0;
	#pragma omp parallel for reduction(+:sumLoss)
	for ( int id = 0;id < H_NBRANCH; id++)
	{
		cuDoubleComplex sum = make_cuDoubleComplex(0, 0);
		cuDoubleComplex sum2 = make_cuDoubleComplex(0, 0);
		for(int k = csrRowPtrYf[id] - BASE_INDEX, endFor = csrRowPtrYf[id + 1] - BASE_INDEX; k < endFor; k++){
			sum = cuCadd(sum, cuCmul(csrValYf[k], V[csrColIndYf[k] - BASE_INDEX]));
		}
		for(int k = csrRowPtrYt[id] - BASE_INDEX, endFor = csrRowPtrYt[id + 1] - BASE_INDEX; k < endFor; k++){
			sum2 = cuCadd(sum2, cuCmul(csrValYt[k], V[csrColIndYt[k] - BASE_INDEX]));
		}
		Branch l_branch = branches[id];
		cuDoubleComplex l_loss;
		l_loss = cuCadd(cuCmul(cuConj(sum), V[l_branch.from]), cuCmul(cuConj(sum2), V[l_branch.to]));
		sumLoss += cuCreal(l_loss);
	}
	return sumLoss;
}

__host__ double mkl_runpf(vector<pso::Particula::Estrutura> &estrutura, pso::Particula &particula) {
	printf("DGM:: Entered mkl_runpf\n");

	double start;
	start = GetTimer();
	mkl_computeVoltage(buses, V, estrutura, particula);
	timeTable[TIME_COMPUTEVOLTAGE] += GetTimer() - start;

#ifdef DEBUG
	printf("V = \n");
	for(int i = 0; i < H_NBUS; i++)
	{
		printf("\t[%d] -> %.4e %c %.4ei\n",i , V[i].x, ((V[i].y < 0) ? '-' : '+'), ((V[i].y < 0) ? -V[i].y : V[i].y));
	}
#endif

	start = GetTimer();
	mkl_makeYbus(estrutura, particula, buses, branches);
	timeTable[TIME_MAKEYBUS] += GetTimer() - start;

#ifdef DEBUG
	printf("Yf = \n");
	printf("\tCompressed Sparse Column(rows = %d, cols = %d, nnz = %d [%.2lf])\n", H_NBRANCH, H_NBUS,nnzYf, nnzYf * 100.0f / (H_NBRANCH * H_NBUS));
	for(int j = 0; j < H_NBUS; j++){
		for(int i = 0; i < H_NBRANCH; i++){
			for(int k = csrRowPtrYf[i] - BASE_INDEX; k < csrRowPtrYf[i + 1] - BASE_INDEX; k++){
				if(j == csrColIndYf[k] - BASE_INDEX){
					cuDoubleComplex value = csrValYf[k];
					printf("\t(%d, %d)\t->\t%.4e%c%.4ei\n", i+1, j+1, value.x,((value.y < 0.0) ? '-' : '+'),((value.y < 0.0) ? -value.y : value.y));
					break;
				}
			}
		}
	}

	printf("Yt = \n");
	printf("\tCompressed Sparse Column(rows = %d, cols = %d, nnz = %d [%.2lf])\n",H_NBRANCH, H_NBUS,nnzYt, nnzYt * 100.0f / (H_NBRANCH * H_NBUS));
	for(int j = 0; j < H_NBUS; j++){
		for(int i = 0; i < H_NBRANCH; i++){
			for(int k = csrRowPtrYt[i] - BASE_INDEX; k < csrRowPtrYt[i + 1] - BASE_INDEX; k++){
				if(j == csrColIndYt[k] - BASE_INDEX){
					cuDoubleComplex value = csrValYt[k];
					printf("\t(%d, %d)\t->\t%.4e%c%.4ei\n", i+1, j+1, value.x,((value.y < 0.0) ? '-' : '+'),((value.y < 0.0) ? -value.y : value.y));
					break;
				}
			}
		}
	}
	printf("Ybus = \n");
	printf("\tCompressed Sparse Column(rows = %d, cols = %d, nnz = %d [%.2lf])\n",H_NBUS, H_NBUS,nnzYbus, nnzYbus * 100.0f / (H_NBUS * H_NBUS));
	for(int j = 0; j < H_NBUS; j++){
		for(int i = 0; i < H_NBUS; i++){
			for(int k = csrRowPtrYbus[i] - BASE_INDEX; k < csrRowPtrYbus[i + 1] - BASE_INDEX; k++){
				if(j == csrColIndYbus[k] - BASE_INDEX){
					cuDoubleComplex value = csrValYbus[k];
					printf("\t(%d, %d)\t->\t%.4e%c%.4ei\n", i+1, j+1, value.x,((value.y < 0.0) ? '-' : '+'),((value.y < 0.0) ? -value.y : value.y));
					break;
				}
			}
		}
	}
#endif
	bool converged = false;
	switch (H_ALG) {
	case NR:
		start =GetTimer();
		converged = mkl_newtonpf();
		timeTable[TIME_NEWTONPF] += GetTimer() - start;
		break;
	case FDXB:
	case FDBX:
		mkl_makeB(estrutura, particula);
		converged = mkl_fdpf();
		break;
	}

	double loss = 0;

	if (converged) {
		start =GetTimer();
		loss = mkl_computeLoss(
				branches,
				V,
				nnzYf,
				csrRowPtrYf,
				csrColIndYf,
				csrValYf,
				nnzYt,
				csrRowPtrYt,
				csrColIndYt,
				csrValYt);
		timeTable[TIME_COMPUTELOSS] += GetTimer() - start;
	} else {
		loss  = DBL_MAX;
	}
	MKL_free(csrColIndYbus);
	MKL_free(csrValYbus);
	return loss;
}

__host__ void mkl_init(Topology& topology, int nTest, vector<pso::Particula::Estrutura> estrutura, int algPF) {
	printf("DGM:: Entered mkl_init\n");

	H_NBUS = topology.buses.size();
	H_NBRANCH = topology.branches.size();
	H_NPV = topology.idPVbuses.size();
	H_NPQ = topology.idPQbuses.size();
	H_ALG = algPF;
	H_NTESTS = nTest;

	buses = thrust::raw_pointer_cast(topology.buses.data());
	branches = thrust::raw_pointer_cast(topology.branches.data());
	pv = thrust::raw_pointer_cast(topology.idPVbuses.data());
	pq = thrust::raw_pointer_cast(topology.idPQbuses.data());

	V = (cuDoubleComplex*) MKL_malloc(H_NBUS  * sizeof(cuDoubleComplex), 64);

	nnzYf = 2 * H_NBRANCH;
	csrValYf = (cuDoubleComplex*) MKL_malloc(nnzYf * sizeof(cuDoubleComplex), 64);
	csrColIndYf = (int*) MKL_malloc(nnzYf * sizeof(int), 64);
	csrRowPtrYf = (int*) MKL_malloc((H_NBRANCH + 1)	* sizeof(int), 64);

	nnzYt = 2 * H_NBRANCH;
	csrValYt = (cuDoubleComplex*) MKL_malloc(nnzYt 	* sizeof(cuDoubleComplex), 64);
	csrColIndYt = (int*) MKL_malloc(nnzYt * sizeof(int), 64);
	csrRowPtrYt = (int*) MKL_malloc((H_NBRANCH + 1)	* sizeof(int), 64);

	nnzYsh = H_NBUS;
	csrValYsh = (cuDoubleComplex*) MKL_malloc(nnzYsh * sizeof(cuDoubleComplex), 64);
	csrColIndYsh = (int*) MKL_malloc(nnzYsh * sizeof(int), 64);
	csrRowPtrYsh = (int*) MKL_malloc((H_NBUS + 1) * sizeof(int), 64);

	nnzCf = H_NBRANCH;
	csrValCf = (cuDoubleComplex*) MKL_malloc(nnzCf * sizeof(cuDoubleComplex), 64);
	csrColIndCf = (int*) MKL_malloc(nnzCf * sizeof(int), 64);
	csrRowPtrCf = (int*) MKL_malloc((H_NBUS + 1) * sizeof(int), 64);

	nnzCt = H_NBRANCH;
	csrValCt = (cuDoubleComplex*) MKL_malloc(nnzCt * sizeof(cuDoubleComplex), 64);
	csrColIndCt = (int*) MKL_malloc(nnzCt * sizeof(int), 64);
	csrRowPtrCt = (int*) malloc((H_NBUS + 1) * sizeof(int));

	nnzCfcoo = H_NBRANCH;
	cooValCf = (cuDoubleComplex*) MKL_malloc(nnzCfcoo 	* sizeof(cuDoubleComplex), 64);
	cooColCf = (int*) MKL_malloc(nnzCfcoo 	* sizeof(int), 64);
	cooRowCf = (int*) MKL_malloc(nnzCfcoo 	* sizeof(int), 64);


	nnzCtcoo = H_NBRANCH;
	cooValCt = (cuDoubleComplex*) MKL_malloc(nnzCtcoo * sizeof(cuDoubleComplex), 64);
	cooColCt = (int*) MKL_malloc(nnzCtcoo * sizeof(int), 64);
	cooRowCt = (int*) MKL_malloc(nnzCtcoo * sizeof(int), 64);

	csrRowPtrYbus = (int*) MKL_malloc((H_NBUS + 1) * sizeof(int), 64);
	csrRowPtrCfYf = (int*) MKL_malloc((H_NBUS + 1) * sizeof(int), 64);
	csrRowPtrCtYt = (int*) MKL_malloc((H_NBUS + 1) * sizeof(int), 64);
	csrRowPtrCfYfCtYt = (int*) MKL_malloc((H_NBUS + 1) * sizeof(int), 64);

	int length = H_NPV + 2 * H_NPQ;
	switch(H_ALG){
	case NR:
		F = (double*)MKL_malloc(length * sizeof(double), 64);
		dx = (double*)MKL_malloc(length * sizeof(double), 64);
		diagIbus = (cuDoubleComplex*) MKL_malloc( H_NBUS * sizeof(cuDoubleComplex), 64);
		nnzJ = 0;
		csrRowPtrJ = (int*) MKL_malloc((length + 1) * sizeof(int), 64);
		break;
	case FDBX:
	case FDXB:
		csrBpRow = (int*) MKL_malloc((H_NPV + H_NPQ + 2) * sizeof(int), 64);
		csrBppRow = (int*) MKL_malloc( (H_NPQ + 2)  * sizeof(int), 64);

		tmpBuses = (Bus*) MKL_malloc( H_NBUS * sizeof(Bus), 64);
		tmpBranches = (Branch*) MKL_malloc( H_NBRANCH * sizeof(Branch), 64);

		tmpNnzYf = 2 * H_NBRANCH;
		tmpCsrValYf = (cuDoubleComplex*) MKL_malloc( nnzYf		* sizeof(cuDoubleComplex), 64);
		tmpCsrColIndYf = (int*) MKL_malloc( nnzYf 			* sizeof(		int		), 64);
		tmpCsrRowPtrYf = (int*) MKL_malloc( (H_NBRANCH + 1) 	* sizeof(		int		), 64);

		tmpNnzYt = 2 * H_NBRANCH;
		tmpCsrValYt = (cuDoubleComplex*) MKL_malloc( nnzYt 	* sizeof(cuDoubleComplex), 64);
		tmpCsrColIndYt = (int*) MKL_malloc( nnzYt 			* sizeof(		int		), 64);
		tmpCsrRowPtrYt = (int*) MKL_malloc( (H_NBRANCH + 1) 	* sizeof(		int		), 64);

		tmpCsrRowPtrYbus = (int*) MKL_malloc( (H_NBUS + 1) 	* sizeof(		int		), 64);

		P = (double*) MKL_malloc( (H_NPV + H_NPQ) 	* sizeof(		double		), 64);
		Q = (double*) MKL_malloc( H_NPQ 	* sizeof(		double		), 64);
		break;
	}
}

__host__ void mkl_clean(){
	MKL_free(V);
	MKL_free(csrColIndYf	);
	MKL_free(csrColIndYt	);
	MKL_free(csrColIndYsh	);
	MKL_free(csrColIndCt	);
	MKL_free(csrColIndCf	);
	MKL_free(cooColCt		);
	MKL_free(cooColCf		);
	MKL_free(csrRowPtrYbus	);
	MKL_free(csrRowPtrYf	);
	MKL_free(csrRowPtrYt	);
	MKL_free(csrRowPtrYsh	);
	free(csrRowPtrCt	);
	MKL_free(csrRowPtrCf	);
	MKL_free(cooRowCt		);
	MKL_free(cooRowCf		);
	MKL_free(csrRowPtrCfYf	);
	MKL_free(csrRowPtrCtYt	);
	MKL_free(csrRowPtrCfYfCtYt	);
	MKL_free(csrValYf		);
	MKL_free(csrValYt		);
	MKL_free(csrValYsh		);
	MKL_free(csrValCf		);
	MKL_free(csrValCt		);
	MKL_free(cooValCf		);
	MKL_free(cooValCt		);
	MKL_free(pBuffer);
	free(converged_test);
	switch(H_ALG){
			case NR:
				MKL_free(F);
				MKL_free(dx);
				MKL_free(csrValJ);
				MKL_free(csrRowPtrJ);
				MKL_free(csrColIndJ);
				MKL_free(diagIbus);
				free(cooRowJ);
				free(h_csrColIndJ);
				free(h_csrRowPtrJ);
				break;
			case FDBX:
			case FDXB:
				MKL_free(cooBpRow);
				MKL_free(cooBpCol);
				MKL_free(cooBpVal);
				MKL_free(csrBpVal);
				MKL_free(csrBpCol);
				MKL_free(csrBpRow);
				MKL_free(cooBppRow);
				MKL_free(cooBppCol);
				MKL_free(cooBppVal);
				MKL_free(csrBppVal);
				MKL_free(csrBppCol);
				MKL_free(csrBppRow);
				MKL_free(tmpBuses);
				MKL_free(tmpBranches);
				MKL_free(tmpCsrColIndYbus	);
				MKL_free(tmpCsrColIndYf	);
				MKL_free(tmpCsrColIndYt	);
				MKL_free(tmpCsrRowPtrYbus	);
				MKL_free(tmpCsrRowPtrYf	);
				MKL_free(tmpCsrRowPtrYt	);
				MKL_free(tmpCsrValYbus		);
				MKL_free(tmpCsrValYf		);
				MKL_free(tmpCsrValYt		);
				MKL_free(P		);
				MKL_free(Q		);
				break;
		}
}

__global__ void hybrid_computeVoltage(
			Bus *buses,
			cuDoubleComplex *V,
			int i,
			pso::Particula::Estrutura *d_estrutura,
			double *d_enxame) {
	int id = ID();
	if (id < D_NBUS) {
		Bus l_bus = buses[id];
		double Vbus = (l_bus.indiceEstrutura != -1 && d_estrutura[l_bus.indiceEstrutura].tipo == pso::Particula::Estrutura::AVR) ? d_enxame[l_bus.indiceEstrutura] :  l_bus.V ;
		V[id] = cuCmul(make_cuDoubleComplex(Vbus, 0),
				cuCexp(make_cuDoubleComplex(0, l_bus.O)));
		if (l_bus.type == l_bus.PV || l_bus.type == l_bus.SLACK) {
			V[id] = cuCmul(make_cuDoubleComplex(Vbus / cuCabs(V[id]), 0.0),
					V[id]);
		}
	}
}

__global__ void hybrid_computeLoss(
		int nTest,
		Branch *branches,
		cuDoubleComplex *V,
		int nnzYf,
		int* csrRowPtrYf,
		int* csrColIndYf,
		cuDoubleComplex* csrValYf,
		int nnzYt,
		int* csrRowPtrYt,
		int* csrColIndYt,
		cuDoubleComplex* csrValYt,
		double *vLoss) {
	int id = ID();
	if (id < D_NBRANCH) {
		cuDoubleComplex sum = make_cuDoubleComplex(0, 0);
		cuDoubleComplex sum2 = make_cuDoubleComplex(0, 0);
		for(int k = csrRowPtrYf[id], endFor = csrRowPtrYf[id + 1]; k < endFor; k++){
			sum = cuCadd(sum, cuCmul(csrValYf[k], V[csrColIndYf[k]]));
		}
		for(int k = csrRowPtrYt[id], endFor = csrRowPtrYt[id + 1]; k < endFor; k++){
			sum2 = cuCadd(sum2, cuCmul(csrValYt[k], V[csrColIndYt[k]]));
		}
		Branch l_branch = branches[id];
		cuDoubleComplex l_loss;
		l_loss = cuCadd(cuCmul(cuConj(sum), V[l_branch.from]), cuCmul(cuConj(sum2), V[l_branch.to]));
		vLoss[id] = cuCreal(l_loss);
	}

}

__host__ void reduceLoss(double* d_idata, double* d_odata, int threads, int blocks, int nElements, cudaStream_t* stream){

	int smemSize = sizeof(double) * threads * 2;
	int dimGrid = blocks;
	int dimBlock = threads;
	switch (threads)
		{
		case 1024:
		reduce<1024><<< dimGrid, dimBlock,  smemSize, *stream>>>(d_idata, d_odata, nElements); break;
		case 512:
		reduce<512><<< dimGrid, dimBlock,  smemSize, *stream>>>(d_idata, d_odata, nElements); break;
		case 256:
		reduce<256><<< dimGrid, dimBlock, smemSize, *stream >>>(d_idata, d_odata, nElements); break;
		case 128:
		reduce<128><<< dimGrid, dimBlock, smemSize, *stream >>>(d_idata, d_odata, nElements); break;
		case 64:
		reduce< 64><<< dimGrid, dimBlock, smemSize, *stream >>>(d_idata, d_odata, nElements); break;
		case 32:
		reduce< 32><<< dimGrid, dimBlock, smemSize, *stream >>>(d_idata, d_odata, nElements); break;
		case 16:
		reduce< 16><<< dimGrid, dimBlock, smemSize, *stream >>>(d_idata, d_odata, nElements); break;
		case 8:
		reduce< 8><<< dimGrid, dimBlock, smemSize, *stream >>>(d_idata, d_odata, nElements); break;
		case 4:
		reduce< 4><<< dimGrid, dimBlock, smemSize, *stream >>>(d_idata, d_odata, nElements); break;
		case 2:
		reduce< 2><<< dimGrid, dimBlock, smemSize, *stream >>>(d_idata, d_odata, nElements); break;
		case 1:
		reduce< 1><<< dimGrid, threads, smemSize, *stream >>>(d_idata, d_odata, nElements); break;
		}
}

__host__ void hybrid_runpf(vector<pso::Particula::Estrutura> &estrutura, vector<pso::Particula> &enxame) {
	printf("DGM:: Entered hybrid_runpf\n");

	double start;
	start = GetTimer();
	if(d_enxame == 0){
		checkCudaErrors(cudaMalloc((void**) &d_enxame, sizeof(double) * enxame.size() * estrutura.size()));
	}
	for(int i = 0; i < enxame.size(); i++){
		checkCudaErrors(cudaMemcpy(d_enxame + i * estrutura.size(),enxame[i].X.data(), sizeof(double) * estrutura.size(), cudaMemcpyHostToDevice));
	}
	timeTable[TIME_INIT_STRUCT_PSO] += GetTimer() - start;
	start = GetTimer();
	for (int i = 0; i < H_NTESTS; i++) {
		//checkCudaErrors(cudaStreamCreate(&stream[i]));
		hybrid_computeVoltage<<<BLOCKS(H_NBUS, H_THREADS), H_THREADS, 0, stream[i]>>>(
				device_buses,
				V + H_NBUS * i,
				i,
				d_estrutura,
				d_enxame + estrutura.size() * i);
	}

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


	for (int i = 0; i < H_NTESTS; i++) {
		hybrid_makeYbus(
				i,
				estrutura.size(),
				device_buses,
				device_branches);
	}
	checkCudaErrors(cudaDeviceSynchronize());
	timeTable[TIME_COMPUTEVOLTAGE] += GetTimer() - start;
	timeTable[TIME_MAKEYBUS] += GetTimer() - start;
#ifdef DEBUG
	for (int t = 0; t < H_NTESTS; t++)
	{
		int *h_row = (int*) malloc(sizeof(int) * (H_NBRANCH + 1));
		int *h_col = (int*) malloc(sizeof(int) * nnzYf);
		cuDoubleComplex *h_val = (cuDoubleComplex*) malloc(sizeof(cuDoubleComplex) * nnzYf);
		cudaMemcpy(h_row, csrRowPtrYf, sizeof(int) * (H_NBRANCH + 1), cudaMemcpyDeviceToHost);
		cudaMemcpy(h_col, csrColIndYf, sizeof(int) * nnzYf, cudaMemcpyDeviceToHost);
		cudaMemcpy(h_val, csrValYf + nnzYf * t, sizeof(cuDoubleComplex) * nnzYf, cudaMemcpyDeviceToHost);
		printf("Yf[%d] = \n", t);
		printf("\tCompressed Sparse Column(rows = %d, cols = %d, nnz = %d [%.2lf])\n",
					H_NBRANCH, H_NBUS,
					nnzYf, nnzYf * 100.0f / (H_NBRANCH * H_NBUS));
		for(int j = 0; j < H_NBUS; j++){
			for(int i = 0; i < H_NBRANCH; i++){
				for(int k = h_row[i]; k < h_row[i + 1]; k++){
					if(j == h_col[k]){
						cuDoubleComplex value = h_val[k];
						printf("\t(%d, %d)\t->\t%.4e%c%.4ei\n", i+1, j+1, value.x,((value.y < 0.0) ? '-' : '+'),((value.y < 0.0) ? -value.y : value.y));
						break;
					}
				}
			}
		}
		free(h_row);
		free(h_col);
		free(h_val);

		h_row = (int*) malloc(sizeof(int) * (H_NBRANCH + 1));
		h_col = (int*) malloc(sizeof(int) * nnzYt);
		h_val = (cuDoubleComplex*) malloc(sizeof(cuDoubleComplex) * nnzYt);
		cudaMemcpy(h_row, csrRowPtrYt, sizeof(int) * (H_NBRANCH + 1), cudaMemcpyDeviceToHost);
		cudaMemcpy(h_col, csrColIndYt, sizeof(int) * nnzYt, cudaMemcpyDeviceToHost);
		cudaMemcpy(h_val, csrValYt + nnzYt * t, sizeof(cuDoubleComplex) * nnzYt, cudaMemcpyDeviceToHost);
		printf("Yt[%d] = \n", t);
		printf("\tCompressed Sparse Column(rows = %d, cols = %d, nnz = %d [%.2lf])\n",
					H_NBRANCH, H_NBUS,
					nnzYt, nnzYt * 100.0f / (H_NBRANCH * H_NBUS));
		for(int j = 0; j < H_NBUS; j++){
			for(int i = 0; i < H_NBRANCH; i++){
				for(int k = h_row[i]; k < h_row[i + 1]; k++){
					if(j == h_col[k]){
						cuDoubleComplex value = h_val[k];
						printf("\t(%d, %d)\t->\t%.4e%c%.4ei\n", i+1, j+1, value.x,((value.y < 0.0) ? '-' : '+'),((value.y < 0.0) ? -value.y : value.y));
						break;
					}
				}
			}
		}
		free(h_row);
		free(h_col);
		free(h_val);

		h_row = (int*) malloc(sizeof(int) * (H_NBRANCH + 1));
		h_col = (int*) malloc(sizeof(int) * nnzYbus);
		h_val = (cuDoubleComplex*) malloc(sizeof(cuDoubleComplex) * nnzYbus);
		cudaMemcpy(h_row, csrRowPtrYbus, sizeof(int) * (H_NBUS + 1), cudaMemcpyDeviceToHost);
		cudaMemcpy(h_col, csrColIndYbus, sizeof(int) * nnzYbus, cudaMemcpyDeviceToHost);
		cudaMemcpy(h_val, csrValYbus + nnzYbus * t, sizeof(cuDoubleComplex) * nnzYbus, cudaMemcpyDeviceToHost);
		printf("Ybus[%d] = \n", t);
		printf("\tCompressed Sparse Column(rows = %d, cols = %d, nnz = %d [%.2lf])\n",
					H_NBUS, H_NBUS,
					nnzYbus, nnzYbus * 100.0f / (H_NBUS * H_NBUS));
		for(int j = 0; j < H_NBUS; j++){
			for(int i = 0; i < H_NBUS; i++){
				for(int k = h_row[i]; k < h_row[i + 1]; k++){
					if(j == h_col[k]){
						cuDoubleComplex value = h_val[k];
						printf("\t(%d, %d)\t->\t%.4e%c%.4ei\n", i+1, j+1, value.x,((value.y < 0.0) ? '-' : '+'),((value.y < 0.0) ? -value.y : value.y));
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

	switch (H_ALG) {
	case NR:
		start = GetTimer();
		hybrid_newtonpf();
		timeTable[TIME_NEWTONPF] += GetTimer() - start;
		break;
	case FDXB:
	case FDBX:
		hybrid_makeB(H_NTESTS, estrutura.size());
		hybrid_fdpf();
		break;
	}

	double loss = 0;
	start = GetTimer();
	for(int t = 0; t < H_NTESTS; t++)
	{
		if (converged_test[t]) {
			hybrid_computeLoss<<<BLOCKS(H_NBRANCH, H_THREADS), H_THREADS, 0, stream[t]>>>(
					t,
					device_branches,
					V + t * H_NBUS,
					nnzYf,
					csrRowPtrYf,
					csrColIndYf,
					csrValYf  + t * nnzYf,
					nnzYt,
					csrRowPtrYt,
					csrColIndYt,
					csrValYt  + t * nnzYt,
					vLoss + t * H_NBRANCH);
		}
	}
	checkCudaErrors(cudaDeviceSynchronize());

	if(reduceBlocks == 1){
		for(int t = 0; t < H_NTESTS; t++){
			reduceLoss(vLoss + t * H_NBRANCH, dReduceLoss + t, reduceThreads, reduceBlocks, H_NBRANCH, &stream[t]);
		}
	} else {
		for(int t = 0; t < H_NTESTS; t++){
			reduceLoss(vLoss + t * H_NBRANCH, dtReduceLoss + t * reduceBlocks, reduceThreads, reduceBlocks, H_NBRANCH, &stream[t]);
			reduceLoss(dtReduceLoss + t * reduceBlocks, dReduceLoss + t, reduceThreadsBlocks, 1, reduceBlocks, &stream[t]);
		}
	}
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaMemcpy(hReduceLoss, dReduceLoss,sizeof(double) * H_NTESTS, cudaMemcpyDeviceToHost));
	for(int t = 0; t < H_NTESTS; t++)
	{
		if (converged_test[t]) {
			loss = hReduceLoss[t];
		} else {
			loss  = DBL_MAX;
		}
		enxame[t].mudarFitness(loss);
	}
	timeTable[TIME_COMPUTELOSS] += GetTimer() - start;

}

void hybrid_init(Topology& topology, int nTest, int nThreads, vector<pso::Particula::Estrutura> estrutura, int algPF) {
	printf("DGM:: Entered hybrid_init\n");

	H_NBUS = topology.buses.size();
	H_NBRANCH = topology.branches.size();
	H_NPV = topology.idPVbuses.size();
	H_NPQ = topology.idPQbuses.size();
	H_ALG = algPF;
	H_NTESTS = nTest;
	H_THREADS = nThreads;

	checkCudaErrors(cudaMalloc((void**) &d_estrutura, sizeof(pso::Particula::Estrutura) * estrutura.size()));
	checkCudaErrors(cudaMemcpy(d_estrutura, estrutura.data(),sizeof(pso::Particula::Estrutura) * estrutura.size(), cudaMemcpyHostToDevice));

	stream = (cudaStream_t*) malloc(sizeof(cudaStream_t) * H_NTESTS);
	for(int i = 0; i < H_NTESTS; i++){
		checkCudaErrors(cudaStreamCreate(&stream[i]));
	}

	checkCudaErrors(cudaMemcpyToSymbol(D_NBUS, &H_NBUS, sizeof(int)));
	checkCudaErrors(cudaMemcpyToSymbol(D_NBRANCH, &H_NBRANCH, sizeof(int)));
	checkCudaErrors(cudaMemcpyToSymbol(D_NPV, &H_NPV, sizeof(int)));
	checkCudaErrors(cudaMemcpyToSymbol(D_NPQ, &H_NPQ, sizeof(int)));
	checkCudaErrors(cudaMemcpyToSymbol(D_ALG, &H_ALG, sizeof(int)));
	checkCudaErrors(cudaMemcpyToSymbol(D_THREADS, &H_THREADS, sizeof(int)));


	buses = thrust::raw_pointer_cast(topology.buses.data());
	branches = thrust::raw_pointer_cast(topology.branches.data());
	pv = thrust::raw_pointer_cast(topology.idPVbuses.data());
	pq = thrust::raw_pointer_cast(topology.idPQbuses.data());

	checkCudaErrors(cudaDeviceSetLimit(cudaLimitMallocHeapSize, 128 * 1024 * 1024));

	checkCudaErrors(cudaMalloc((void**) &device_buses, H_NBUS * sizeof(Bus)));
	checkCudaErrors(cudaMalloc((void**) &device_branches, H_NBRANCH * sizeof(Branch)));
	checkCudaErrors(cudaMalloc((void**) &device_pv, H_NPV * sizeof(unsigned int)));
	checkCudaErrors(cudaMalloc((void**) &device_pq, H_NPQ * sizeof(unsigned int)));

	checkCudaErrors(cudaMemcpy(device_buses, buses, H_NBUS * sizeof(Bus), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(device_branches, branches, H_NBRANCH * sizeof(Branch), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(device_pv, pv, H_NPV * sizeof(unsigned int), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(device_pq, pq, H_NPQ * sizeof(unsigned int), cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc((void**) &V, H_NBUS * H_NTESTS * sizeof(cuDoubleComplex)));

	nnzYf = 2 * H_NBRANCH;
	checkCudaErrors(cudaMalloc((void**) &csrValYf	, nnzYf * H_NTESTS		* sizeof(cuDoubleComplex)));
	checkCudaErrors(cudaMalloc((void**) &csrColIndYf, nnzYf 			* sizeof(		int		)));
	checkCudaErrors(cudaMalloc((void**) &csrRowPtrYf, (H_NBRANCH + 1) 	* sizeof(		int		)));

	nnzYt = 2 * H_NBRANCH;
	checkCudaErrors(cudaMalloc((void**) &csrValYt	, nnzYt * H_NTESTS 	* sizeof(cuDoubleComplex)));
	checkCudaErrors(cudaMalloc((void**) &csrColIndYt, nnzYt 			* sizeof(		int		)));
	checkCudaErrors(cudaMalloc((void**) &csrRowPtrYt, (H_NBRANCH + 1) 	* sizeof(		int		)));

	nnzYsh = H_NBUS;
	checkCudaErrors(cudaMalloc((void**) &csrValYsh		, nnzYsh * H_NTESTS 	* sizeof(cuDoubleComplex)));
	checkCudaErrors(cudaMalloc((void**) &csrColIndYsh	, nnzYsh 			* sizeof(		int		)));
	checkCudaErrors(cudaMalloc((void**) &csrRowPtrYsh	, (H_NBUS + 1) 		* sizeof(		int		)));

	nnzCf = H_NBRANCH;
	checkCudaErrors(cudaMalloc((void**) &csrValCf	,	 nnzCf	 	* sizeof(cuDoubleComplex)));
	checkCudaErrors(cudaMalloc((void**) &csrColIndCf, 	 nnzCf	 	* sizeof(		int		)));
	checkCudaErrors(cudaMalloc((void**) &csrRowPtrCf, (H_NBUS + 1) 	* sizeof(		int		)));

	nnzCt = H_NBRANCH;
	checkCudaErrors(cudaMalloc((void**) &csrValCt	,	 nnzCt	 	* sizeof(cuDoubleComplex)));
	checkCudaErrors(cudaMalloc((void**) &csrColIndCt, 	 nnzCt	 	* sizeof(		int		)));
	checkCudaErrors(cudaMalloc((void**) &csrRowPtrCt, (H_NBUS + 1) 	* sizeof(		int		)));

	nnzCfcoo = H_NBRANCH;
	checkCudaErrors(cudaMalloc((void**) &cooValCf,	 nnzCfcoo 	* sizeof(cuDoubleComplex)));
	checkCudaErrors(cudaMalloc((void**) &cooColCf,	 nnzCfcoo 	* sizeof(		int		)));
	checkCudaErrors(cudaMalloc((void**) &cooRowCf, 	 nnzCfcoo 	* sizeof(		int		)));

	nnzCtcoo = H_NBRANCH;
	checkCudaErrors(cudaMalloc((void**) &cooValCt,	 nnzCtcoo 	* sizeof(cuDoubleComplex)));
	checkCudaErrors(cudaMalloc((void**) &cooColCt,	 nnzCtcoo 	* sizeof(		int		)));
	checkCudaErrors(cudaMalloc((void**) &cooRowCt, 	 nnzCtcoo 	* sizeof(		int		)));

	checkCudaErrors(cusparseCreate(&sparseHandle));
	checkCudaErrors(cusparseSetPointerMode(sparseHandle, CUSPARSE_POINTER_MODE_HOST));

	nPermutation = H_NBRANCH;
	checkCudaErrors(cudaMalloc((void**) &permutation, H_NBRANCH * sizeof(int)));

	checkCudaErrors(cudaMalloc((void**) &csrRowPtrYbus, (H_NBUS + 1) 	* sizeof(		int		)));
	checkCudaErrors(cudaMalloc((void**) &csrRowPtrCfYf, (H_NBUS + 1) 	* sizeof(		int		)));
	checkCudaErrors(cudaMalloc((void**) &csrRowPtrCtYt, (H_NBUS + 1) 	* sizeof(		int		)));
	checkCudaErrors(cudaMalloc((void**) &csrRowPtrCfYfCtYt, (H_NBUS + 1) 	* sizeof(		int		)));

	int length = H_NPV + 2 * H_NPQ;
	switch(H_ALG){
	case NR:
		checkCudaErrors(cudaMalloc((void**) &F, H_NTESTS * length * sizeof(double)));
		checkCudaErrors(cudaMalloc((void**) &dx, length * sizeof(double)));
		checkCudaErrors(cudaMalloc((void**) &diagIbus, H_NTESTS * H_NBUS * sizeof(cuDoubleComplex)));
		nnzJ = 0;
		checkCudaErrors(cudaMalloc((void**) &csrRowPtrJ, (length + 1) * sizeof(int)));
		break;
	case FDBX:
	case FDXB:
		checkCudaErrors(cudaMalloc((void**) &csrBpRow, (H_NPV + H_NPQ + 2) * sizeof(int)));
		checkCudaErrors(cudaMalloc((void**) &csrBppRow, (H_NPQ + 2)  * sizeof(int)));

		checkCudaErrors(cudaMalloc((void**) &tmpBuses, H_NBUS * sizeof(Bus)));
		checkCudaErrors(cudaMalloc((void**) &tmpBranches, H_NBRANCH * sizeof(Branch)));

		tmpNnzYf = 2 * H_NBRANCH;
		checkCudaErrors(cudaMalloc((void**) &tmpCsrValYf	, nnzYf * H_NTESTS		* sizeof(cuDoubleComplex)));
		checkCudaErrors(cudaMalloc((void**) &tmpCsrColIndYf, nnzYf 			* sizeof(		int		)));
		checkCudaErrors(cudaMalloc((void**) &tmpCsrRowPtrYf, (H_NBRANCH + 1) 	* sizeof(		int		)));

		tmpNnzYt = 2 * H_NBRANCH;
		checkCudaErrors(cudaMalloc((void**) &tmpCsrValYt	, nnzYt * H_NTESTS 	* sizeof(cuDoubleComplex)));
		checkCudaErrors(cudaMalloc((void**) &tmpCsrColIndYt, nnzYt 			* sizeof(		int		)));
		checkCudaErrors(cudaMalloc((void**) &tmpCsrRowPtrYt, (H_NBRANCH + 1) 	* sizeof(		int		)));

		checkCudaErrors(cudaMalloc((void**) &tmpCsrRowPtrYbus, (H_NBUS + 1) 	* sizeof(		int		)));

		checkCudaErrors(cudaMalloc((void**) &P, (H_NPV + H_NPQ) * H_NTESTS 	* sizeof(		double		)));
		checkCudaErrors(cudaMalloc((void**) &Q, H_NPQ * H_NTESTS 	* sizeof(		double		)));
		break;
	}

	converged_test = (bool*) malloc(sizeof(bool) * H_NTESTS);

	checkCudaErrors(cudaMalloc((void**) &vLoss, H_NBRANCH * H_NTESTS * sizeof(double)));

	reduceBlocks = (H_NBRANCH > 2048) ? BLOCKS(H_NBRANCH, 2048) :  1;
	reduceBlocks = min(reduceBlocks, 2048);
	if(reduceBlocks != 1){
		checkCudaErrors(cudaMalloc((void**) &dtReduceLoss, sizeof(double) * reduceBlocks * H_NTESTS));
		if(reduceBlocks > 2048){
			reduceThreadsBlocks = 1024;
		} else if (reduceBlocks > 1024){
			reduceThreadsBlocks = 1024;
		}else if (reduceBlocks > 512){
			reduceThreadsBlocks = 512;
		}else if (reduceBlocks > 256){
			reduceThreadsBlocks = 256;
		}else if (reduceBlocks > 128){
			reduceThreadsBlocks = 128;
		}else if (reduceBlocks > 64){
			reduceThreadsBlocks = 64;
		}else if (reduceBlocks > 32){
			reduceThreadsBlocks = 32;
		}else if (reduceBlocks > 16){
			reduceThreadsBlocks = 16;
		}else if (reduceBlocks > 8){
			reduceThreadsBlocks = 8;
		}else if (reduceBlocks > 4){
			reduceThreadsBlocks = 4;
		}else if (reduceBlocks > 2){
			reduceThreadsBlocks = 2;
		}else {
			reduceThreadsBlocks = 1;
		}
	}
	hReduceLoss = (double*) malloc(sizeof(double) * H_NTESTS);
	checkCudaErrors(cudaMalloc((void**) &dReduceLoss, sizeof(double) * H_NTESTS));
	if(H_NBRANCH > 2048){
		reduceThreads = 1024;
	} else if (H_NBRANCH > 1024){
		reduceThreads = 1024;
	}else if (H_NBRANCH > 512){
		reduceThreads = 512;
	}else if (H_NBRANCH > 256){
		reduceThreads = 256;
	}else if (H_NBRANCH > 128){
		reduceThreads = 128;
	}else if (H_NBRANCH > 64){
		reduceThreads = 64;
	}else if (H_NBRANCH > 32){
		reduceThreads = 32;
	}else if (H_NBRANCH > 16){
		reduceThreads = 16;
	}else if (H_NBRANCH > 8){
		reduceThreads = 8;
	}else if (H_NBRANCH > 4){
		reduceThreads = 4;
	}else if (H_NBRANCH > 2){
		reduceThreads = 2;
	}else {
		reduceThreads = 1;
	}
}

__host__ void hybrid_free(){
	checkCudaErrors(cudaFree(V));
	checkCudaErrors(cudaFree(csrColIndYbus	));
	checkCudaErrors(cudaFree(csrColIndYf	));
	checkCudaErrors(cudaFree(csrColIndYt	));
	checkCudaErrors(cudaFree(csrColIndYsh	));
	checkCudaErrors(cudaFree(csrColIndCt	));
	checkCudaErrors(cudaFree(csrColIndCf	));
	checkCudaErrors(cudaFree(cooColCt		));
	checkCudaErrors(cudaFree(cooColCf		));
	checkCudaErrors(cudaFree(csrColIndCfYf	));
	checkCudaErrors(cudaFree(csrColIndCtYt	));
	checkCudaErrors(cudaFree(csrColIndCfYfCtYt	));
	checkCudaErrors(cudaFree(csrRowPtrYbus	));
	checkCudaErrors(cudaFree(csrRowPtrYf	));
	checkCudaErrors(cudaFree(csrRowPtrYt	));
	checkCudaErrors(cudaFree(csrRowPtrYsh	));
	checkCudaErrors(cudaFree(csrRowPtrCt	));
	checkCudaErrors(cudaFree(csrRowPtrCf	));
	checkCudaErrors(cudaFree(cooRowCt		));
	checkCudaErrors(cudaFree(cooRowCf		));
	checkCudaErrors(cudaFree(csrRowPtrCfYf	));
	checkCudaErrors(cudaFree(csrRowPtrCtYt	));
	checkCudaErrors(cudaFree(csrRowPtrCfYfCtYt	));
	checkCudaErrors(cudaFree(csrValYbus		));
	checkCudaErrors(cudaFree(csrValYf		));
	checkCudaErrors(cudaFree(csrValYt		));
	checkCudaErrors(cudaFree(csrValYsh		));
	checkCudaErrors(cudaFree(csrValCf		));
	checkCudaErrors(cudaFree(csrValCt		));
	checkCudaErrors(cudaFree(cooValCf		));
	checkCudaErrors(cudaFree(cooValCt		));
	checkCudaErrors(cudaFree(csrValCtYt		));
	checkCudaErrors(cudaFree(csrValCfYf		));
	checkCudaErrors(cudaFree(csrValCfYfCtYt	));
	checkCudaErrors(cudaFree(pBuffer		));
	checkCudaErrors(cudaFree(permutation	));
	checkCudaErrors(cusparseDestroy(sparseHandle));
	checkCudaErrors(cusparseDestroyMatDescr(descrCf));
	checkCudaErrors(cusparseDestroyMatDescr(descrYf));
	checkCudaErrors(cusparseDestroyMatDescr(descrCfYf));
	checkCudaErrors(cusparseDestroyMatDescr(descrCt));
	checkCudaErrors(cusparseDestroyMatDescr(descrYt));
	checkCudaErrors(cusparseDestroyMatDescr(descrCtYt));
	checkCudaErrors(cusparseDestroyMatDescr(descrCfYfCtYt));
	checkCudaErrors(cusparseDestroyMatDescr(descrYsh));
	checkCudaErrors(cusparseDestroyMatDescr(descrYbus));
	switch(H_ALG){
		case NR:
			checkCudaErrors(cudaFree(F));
			checkCudaErrors(cudaFree(csrValJ));
			checkCudaErrors(cudaFree(csrRowPtrJ));
			checkCudaErrors(cudaFree(csrColIndJ));
			checkCudaErrors(cudaFree(d_cooRowJ));
			checkCudaErrors(cudaFree(dx));
			free(h_csrColIndJ);
			free(h_csrRowPtrJ);
			break;
		case FDBX:
		case FDXB:
			checkCudaErrors(cudaFree(cooBpRow));
			checkCudaErrors(cudaFree(cooBpCol));
			checkCudaErrors(cudaFree(cooBpVal));
			checkCudaErrors(cudaFree(csrBpVal));
			checkCudaErrors(cudaFree(csrBpCol));
			checkCudaErrors(cudaFree(csrBpRow));
			checkCudaErrors(cudaFree(cooBppRow));
			checkCudaErrors(cudaFree(cooBppCol));
			checkCudaErrors(cudaFree(cooBppVal));
			checkCudaErrors(cudaFree(csrBppVal));
			checkCudaErrors(cudaFree(csrBppCol));
			checkCudaErrors(cudaFree(csrBppRow));
			checkCudaErrors(cudaFree(tmpBuses));
			checkCudaErrors(cudaFree(tmpBranches));
			checkCudaErrors(cudaFree(tmpCsrColIndYbus	));
			checkCudaErrors(cudaFree(tmpCsrColIndYf	));
			checkCudaErrors(cudaFree(tmpCsrColIndYt	));
			checkCudaErrors(cudaFree(tmpCsrRowPtrYbus	));
			checkCudaErrors(cudaFree(tmpCsrRowPtrYf	));
			checkCudaErrors(cudaFree(tmpCsrRowPtrYt	));
			checkCudaErrors(cudaFree(tmpCsrValYbus		));
			checkCudaErrors(cudaFree(tmpCsrValYf		));
			checkCudaErrors(cudaFree(tmpCsrValYt		));
			checkCudaErrors(cudaFree(P		));
			checkCudaErrors(cudaFree(Q		));
			break;
	}
	checkCudaErrors(cudaFree(diagIbus));
	checkCudaErrors(cudaFree(d_estrutura));
	checkCudaErrors(cudaFree(d_enxame));
	free(converged_test);
	free(stream);
	//checkCudaErrors(cudaFree(vLoss));
	cudaDeviceReset();
}
#endif /* RUNPF_CUH_ */
