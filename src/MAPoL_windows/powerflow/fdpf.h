/*
 * fdpf.cuh
 *
 *  Created on: 19/11/2015
 *      Author: Igor M. Araújo
 */

#ifndef FDPF_CUH_
#define FDPF_CUH_

template <typename T> struct maximum_abs
{
	__host__ __device__ T operator()(const T& x, const T& y)
	{
		return abs(x) > abs(y) ? abs(x) : abs(y);
	}
};


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
		double *P,
		double *Q) {
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
		miss = cuCdiv(miss, make_cuDoubleComplex(cuCabs(V[indice]), 0));
		if (l_bus.type == l_bus.PV) {
			P[i] = cuCreal(miss);
		}
		if (l_bus.type == l_bus.PQ) {
			P[D_NPV + i ] = cuCreal(miss);
			Q[i] = cuCimag(miss);
		}
	}
}

__global__ void hybrid_updateVoltage_dVa(
		unsigned int *pv,
		unsigned int *pq,
		cuDoubleComplex *V,
		double *dVa)
{
	int id = ID();
	int i;
	if (id < D_NPV + D_NPQ) {
		if (id < D_NPV) {
			i = pv[id];
			cuDoubleComplex voltage = V[i];
			V[i] = cuCmul(make_cuDoubleComplex(cuCabs(voltage), 0), cuCexp(make_cuDoubleComplex(0, cuCangle(voltage) - dVa[id])));
		} else {
			i = pq[id - D_NPV];
			cuDoubleComplex voltage = V[i];
			V[i] = cuCmul(make_cuDoubleComplex(cuCabs(voltage), 0),cuCexp(make_cuDoubleComplex(0,cuCangle(voltage) - dVa[id])));
		}
	}
}

__global__ void hybrid_updateVoltage_dVm(
		unsigned int *pv,
		unsigned int *pq,
		cuDoubleComplex *V,
		double *dVa)
{
	int id = ID();
	int i;
	if (id < D_NPQ) {
		i = pq[id];
		cuDoubleComplex voltage = V[i];
		V[i] = cuCmul(make_cuDoubleComplex(cuCabs(voltage) - dVa[id], 0),cuCexp(make_cuDoubleComplex(0,cuCangle(voltage))));
	}
}

void hybrid_fdpf() {
	maximum_abs<double> normf;
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
				P + t * (H_NPV + H_NPQ),
				Q + t * H_NPQ);
	}
	checkCudaErrors(cudaDeviceSynchronize());

	bool converged = true;
	for(int t = 0; t < H_NTESTS; t++){
		thrust::device_ptr<double> reduceP(P);
		double errP = thrust::reduce(reduceP + t * (H_NPV + H_NPQ), reduceP + t * (H_NPV + H_NPQ) + (H_NPV + H_NPQ), 0.0, normf);
		thrust::device_ptr<double> reduceQ(Q);
		double errQ = thrust::reduce(reduceQ + t * H_NPQ, reduceQ + t * H_NPQ + H_NPQ, 0.0, normf);
		if(max(errP, errQ) < EPS){
			converged_test[t] = true;
		} else {
			converged_test[t] = false;
		}
		converged &= converged_test[t];
	}

	int nnzBp = 0;
	checkCudaErrors(cudaMemcpy(&nnzBp, csrBpRow + H_NPQ + H_NPV, sizeof(int), cudaMemcpyDeviceToHost));
	int *h_csrBpRow = (int*) malloc(sizeof(int) * (H_NPQ + H_NPV + 1));
	int *h_csrBpCol = (int*) malloc(sizeof(int) * nnzBp);
	double *h_csrBpVal = (double*) malloc( sizeof(double) * nnzBp);

	checkCudaErrors(cudaMemcpyAsync(h_csrBpRow, csrBpRow, sizeof(int) * (H_NPQ + H_NPV + 1), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpyAsync(h_csrBpCol, csrBpCol, sizeof(int) * nnzBp, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpyAsync(h_csrBpVal, csrBpVal, sizeof(double) * nnzBp, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaDeviceSynchronize());

	int lengthP = (H_NPQ + H_NPV);
	_MKL_DSS_HANDLE_t handleP;
	MKL_INT opt;
	opt  = MKL_DSS_MSG_LVL_WARNING;
	opt += MKL_DSS_ZERO_BASED_INDEXING;
	MKL_INT result;
	result = DSS_CREATE(handleP, opt); if(result != MKL_DSS_SUCCESS){printf("MKL Library error in %s at %d", __FILE__, __LINE__);exit(1);}
	MKL_INT opt_define = MKL_DSS_NON_SYMMETRIC;
	result = DSS_DEFINE_STRUCTURE(handleP, opt_define, h_csrBpRow, lengthP, lengthP, h_csrBpCol, nnzBp);if(result != MKL_DSS_SUCCESS){printf("MKL Library error in %s at %d", __FILE__, __LINE__);exit(1);}
	int *permP = (int*) MKL_malloc(sizeof(int) * (H_NPQ + H_NPV), 64);
	for(int i = 0; i < (H_NPQ + H_NPV); i++){
		permP[i] = i;
	}
	MKL_INT opt_REORDER = MKL_DSS_AUTO_ORDER;
	result = DSS_REORDER(handleP, opt_REORDER,permP);if(result != MKL_DSS_SUCCESS){printf("MKL Library error in %s at %d", __FILE__, __LINE__);exit(1);}

	int nnzBpp = 0;
	checkCudaErrors(cudaMemcpy(&nnzBpp, csrBppRow + H_NPQ, sizeof(int), cudaMemcpyDeviceToHost));
	int *h_csrBppRow = (int*) malloc(sizeof(int) * (H_NPQ + 1));
	int *h_csrBppCol = (int*) malloc(sizeof(int) * nnzBpp);
	double *h_csrBppVal = (double*) malloc( sizeof(double) * nnzBpp * H_NTESTS);

	checkCudaErrors(cudaMemcpyAsync(h_csrBppRow, csrBppRow, sizeof(int) * (H_NPQ + 1), cudaMemcpyDeviceToHost, stream[0]));
	checkCudaErrors(cudaMemcpyAsync(h_csrBppCol, csrBppCol, sizeof(int) * nnzBpp, cudaMemcpyDeviceToHost, stream[1]));
	checkCudaErrors(cudaMemcpyAsync(h_csrBppVal, csrBppVal, sizeof(double) * nnzBpp * H_NTESTS, cudaMemcpyDeviceToHost, stream[2]));
	checkCudaErrors(cudaDeviceSynchronize());

	_MKL_DSS_HANDLE_t* handleQ = new _MKL_DSS_HANDLE_t [H_NTESTS];
	int *permQ = (int*) MKL_malloc(sizeof(int) * H_NPQ, 64);
	for(int t = 0; t < H_NTESTS; t++){
		MKL_INT opt;
		opt  = MKL_DSS_MSG_LVL_WARNING;
		opt += MKL_DSS_ZERO_BASED_INDEXING;
		MKL_INT result;
		result = DSS_CREATE(handleQ[t], opt); if(result != MKL_DSS_SUCCESS){printf("MKL Library error in %s at %d", __FILE__, __LINE__);exit(1);}
		MKL_INT opt_define = MKL_DSS_NON_SYMMETRIC;
		result = DSS_DEFINE_STRUCTURE(handleQ[t], opt_define, h_csrBppRow, H_NPQ, H_NPQ, h_csrBppCol, nnzBpp);if(result != MKL_DSS_SUCCESS){printf("MKL Library error in %s at %d", __FILE__, __LINE__);exit(1);}
		for(int i = 0; i < H_NPQ; i++){
			permQ[i] = i;
		}
		MKL_INT opt_REORDER = MKL_DSS_AUTO_ORDER;
		result = DSS_REORDER(handleQ[t], opt_REORDER,permQ);if(result != MKL_DSS_SUCCESS){printf("MKL Library error in %s at %d", __FILE__, __LINE__);exit(1);}
	}



	double *dVa = (double*) malloc(sizeof(double) * (H_NPV + H_NPQ) * H_NTESTS);
	double *dVm = (double*) malloc(sizeof(double) * H_NPQ * H_NTESTS);

	double *h_P = (double*) malloc(sizeof(double) * (H_NPV + H_NPQ)  * H_NTESTS);
	double *h_Q = (double*) malloc(sizeof(double) * H_NPQ * H_NTESTS);

	int i = 0;
	while (!converged && i < MAX_IT_FD) {
		i++;

		//-----  do P iteration, update Va  -----
		checkCudaErrors(cudaMemcpy(h_P, P, sizeof(double) * H_NTESTS * (H_NPV + H_NPQ), cudaMemcpyDeviceToHost));
		MKL_INT opt_FACTOR = MKL_DSS_POSITIVE_DEFINITE;
		result = DSS_FACTOR_REAL(handleP, opt_FACTOR, h_csrBpVal);if(result != MKL_DSS_SUCCESS){printf("MKL Library error in %s at %d", __FILE__, __LINE__);exit(1);}
		MKL_INT opt_DEFAULT = MKL_DSS_DEFAULTS;
		MKL_INT nrhs = H_NTESTS;
		result = DSS_SOLVE_REAL(handleP, opt_DEFAULT, h_P, nrhs, dVa);if(result != MKL_DSS_SUCCESS){printf("MKL Library error in %s at %d", __FILE__, __LINE__);exit(1);}
#ifdef DEBUG
		printf("P:\n");
		for(int j = 0; j < (H_NPV + H_NPQ); j++){
			printf("\t%.4e\n", h_P[j]);
		}

		printf("dVa:\n");
		for(int j = 0; j < (H_NPV + H_NPQ); j++){
			printf("\t%.4e\n", -dVa[j]);
		}
#endif
		//-----  update voltage  -----
		checkCudaErrors(cudaMemcpy(P, dVa, sizeof(double) * H_NTESTS * (H_NPV + H_NPQ), cudaMemcpyHostToDevice));
		for(int t = 0; t < H_NTESTS; t++)
		{
			if(!converged_test[t]){
				hybrid_updateVoltage_dVa<<<BLOCKS((H_NPV + H_NPQ), H_THREADS), H_THREADS, 0, stream[t]>>>(
						device_pv,
						device_pq,
						V  + t * H_NBUS,
						P  + t * (H_NPV + H_NPQ));
			}
		}
		checkCudaErrors(cudaDeviceSynchronize());

#ifdef DEBUG
	for (int t = 0; t < H_NTESTS; t++)
	{
		cuDoubleComplex *h_V = (cuDoubleComplex*) malloc(sizeof(cuDoubleComplex) * H_NBUS);
		cudaMemcpy(h_V, V + H_NBUS * t, sizeof(cuDoubleComplex) * H_NBUS, cudaMemcpyDeviceToHost);
		printf("V[%d] = \n", t);
		for(int i = 0; i < H_NBUS; i++)
		{
			printf("\t%.4e %c %.4ei\n", h_V[i].x, ((h_V[i].y < 0) ? '-' : '+'), ((h_V[i].y < 0) ? -h_V[i].y : h_V[i].y));
		}
		free(h_V);
	}
#endif

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
					P + t * (H_NPV + H_NPQ),
					Q + t * H_NPQ);
		}
		checkCudaErrors(cudaDeviceSynchronize());

		converged = true;
		for(int t = 0; t < H_NTESTS; t++){
			thrust::device_ptr<double> reduceP(P);
			double errP = thrust::reduce(reduceP + t * (H_NPV + H_NPQ), reduceP + t * (H_NPV + H_NPQ) + (H_NPV + H_NPQ), 0.0, normf);
			thrust::device_ptr<double> reduceQ(Q);
			double errQ = thrust::reduce(reduceQ + t * H_NPQ, reduceQ + t * H_NPQ + H_NPQ, 0.0, normf);
			checkCudaErrors(cudaDeviceSynchronize());
			if(max(errP, errQ) < EPS){
				converged_test[t] = true;
			} else {
				converged_test[t] = false;
			}
			converged &= converged_test[t];
		}
		if(converged){
			break;
		}


		// -----  do Q iteration, update Vm  -----
		checkCudaErrors(cudaMemcpy(h_Q, Q, sizeof(double) * H_NTESTS * H_NPQ, cudaMemcpyDeviceToHost));
		for(int t = 0; t < H_NTESTS; t++){
			MKL_INT opt_FACTOR = MKL_DSS_POSITIVE_DEFINITE;
			result = DSS_FACTOR_REAL(handleQ[t], opt_FACTOR, h_csrBppVal + t * nnzBpp);if(result != MKL_DSS_SUCCESS){printf("MKL Library error in %s at %d", __FILE__, __LINE__);exit(1);}
			MKL_INT opt_DEFAULT = MKL_DSS_DEFAULTS;
			MKL_INT nrhs = 1;
			result = DSS_SOLVE_REAL(handleQ[t], opt_DEFAULT, h_Q + t * H_NPQ, nrhs, dVm + t * H_NPQ);if(result != MKL_DSS_SUCCESS){printf("MKL Library error in %s at %d", __FILE__, __LINE__);exit(1);}
		}

		//-----  update voltage  -----
		checkCudaErrors(cudaMemcpy(Q, dVm, sizeof(double) * H_NTESTS * H_NPQ, cudaMemcpyHostToDevice));
		for(int t = 0; t < H_NTESTS; t++)
		{
			if(!converged_test[t]){
				hybrid_updateVoltage_dVm<<<BLOCKS(H_NPQ, H_THREADS), H_THREADS, 0, stream[t]>>>(
						device_pv,
						device_pq,
						V  + t * H_NBUS,
						Q  + t * H_NPQ);
			}
		}
		checkCudaErrors(cudaDeviceSynchronize());

#ifdef DEBUG
	checkCudaErrors(cudaDeviceSynchronize());
	for (int t = 0; t < H_NTESTS; t++)
	{
		cuDoubleComplex *h_V = (cuDoubleComplex*) malloc(sizeof(cuDoubleComplex) * H_NBUS);
		cudaMemcpy(h_V, V + H_NBUS * t, sizeof(cuDoubleComplex) * H_NBUS, cudaMemcpyDeviceToHost);
		printf("V[%d] = \n", t);
		for(int i = 0; i < H_NBUS; i++)
		{
			printf("\t%.4e %c %.4ei\n", h_V[i].x, ((h_V[i].y < 0) ? '-' : '+'), ((h_V[i].y < 0) ? -h_V[i].y : h_V[i].y));
		}
		free(h_V);
	}
#endif

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
					P + t * (H_NPV + H_NPQ),
					Q + t * H_NPQ);
		}
		checkCudaErrors(cudaDeviceSynchronize());

		converged = true;
		for(int t = 0; t < H_NTESTS; t++){
			thrust::device_ptr<double> reduceP(P);
			double errP = thrust::reduce(reduceP + t * (H_NPV + H_NPQ), reduceP + t * (H_NPV + H_NPQ) + (H_NPV + H_NPQ), 0.0, normf);
			thrust::device_ptr<double> reduceQ(Q);
			double errQ = thrust::reduce(reduceQ + t * H_NPQ, reduceQ + t * H_NPQ + H_NPQ, 0.0, normf);
			checkCudaErrors(cudaDeviceSynchronize());
			if(max(errP, errQ) < EPS){
				converged_test[t] = true;
			} else {
				converged_test[t] = false;
			}
			converged &= converged_test[t];
		}
		if(converged){
			break;
		}
	}
	free(h_csrBpCol);
	free(h_csrBpRow);
	free(h_csrBpVal);
	free(h_csrBppCol);
	free(h_csrBppRow);
	free(h_csrBppVal);
	free(h_P);
	free(h_Q);
	free(dVa);
	free(dVm);
	MKL_free(permP);
	MKL_free(permQ);
}

double mkl_checkConvergence(
		Bus* buses,
		unsigned int* pv,
		unsigned int* pq,
		int nnzYbus,
		int* csrRowPtrYbus,
		int* csrColIndYbus,
		cuDoubleComplex* csrValYbus,
		cuDoubleComplex *V,
		double *P,
		double *Q) {
	double errP = -1.0;
	double errQ = -1.0;
	for (int id = 0; id < H_NPV + H_NPQ; id++) {
		int i, indice;
		if (id < H_NPV) {
			i = id;
			indice = pv[i];
		} else {
			i = id - H_NPV;
			indice = pq[i];
		}

		cuDoubleComplex c = make_cuDoubleComplex(0, 0);
		for (int k = csrRowPtrYbus[indice] - BASE_INDEX, endFor = csrRowPtrYbus[indice + 1] - BASE_INDEX; k < endFor; k++) {
			int j = csrColIndYbus[k]  - BASE_INDEX;
			c = cuCadd(c, cuCmul(csrValYbus[k], V[j]));
		}
		Bus l_bus = buses[indice];
		cuDoubleComplex pot = make_cuDoubleComplex(l_bus.P, l_bus.Q);
		cuDoubleComplex miss = cuCmul(V[indice], cuConj(c));
		miss = cuCsub(miss, pot);
		miss = cuCdiv(miss, make_cuDoubleComplex(cuCabs(V[indice]), 0));
		if (l_bus.type == l_bus.PV) {
			P[i] = cuCreal(miss);
			errP = max(errP, abs(P[i]));
		}
		if (l_bus.type == l_bus.PQ) {
			P[H_NPV + i ] = cuCreal(miss);
			errP = max(errP, abs(P[H_NPV + i ]));
			Q[i] = cuCimag(miss);
			errQ = max(errQ, abs(Q[i]));
		}
	}
	return max(errP, errQ);
}

void mkl_updateVoltage_dVa(
		unsigned int *pv,
		unsigned int *pq,
		cuDoubleComplex *V,
		double *dVa)
{
	int i;
	for (int id  = 0; id < H_NPV + H_NPQ; id++) {
		if (id < H_NPV) {
			i = pv[id];
			cuDoubleComplex voltage = V[i];
			V[i] = cuCmul(make_cuDoubleComplex(cuCabs(voltage), 0), cuCexp(make_cuDoubleComplex(0, cuCangle(voltage) - dVa[id])));
		} else {
			i = pq[id - H_NPV];
			cuDoubleComplex voltage = V[i];
			V[i] = cuCmul(make_cuDoubleComplex(cuCabs(voltage), 0),cuCexp(make_cuDoubleComplex(0,cuCangle(voltage) - dVa[id])));
		}
	}
}

void mkl_updateVoltage_dVm(
		unsigned int *pv,
		unsigned int *pq,
		cuDoubleComplex *V,
		double *dVa)
{
	int i;
	for (int id  = 0; id < H_NPQ; id++) {
		i = pq[id];
		cuDoubleComplex voltage = V[i];
		V[i] = cuCmul(make_cuDoubleComplex(cuCabs(voltage) - dVa[id], 0),cuCexp(make_cuDoubleComplex(0,cuCangle(voltage))));
	}
}

bool mkl_fdpf() {
	bool converged = false;
	double err = mkl_checkConvergence(
			buses,
			pv,
			pq,
			nnzYbus,
			csrRowPtrYbus,
			csrColIndYbus,
			csrValYbus,
			V,
			P,
			Q);
	if(err < EPS){
		converged = true;
	}

	int lengthP = (H_NPQ + H_NPV);
	_MKL_DSS_HANDLE_t handleP;
	MKL_INT opt;
	opt  = MKL_DSS_MSG_LVL_WARNING;
	opt += MKL_DSS_ZERO_BASED_INDEXING;
	MKL_INT result;
	result = DSS_CREATE(handleP, opt); if(result != MKL_DSS_SUCCESS){printf("MKL Library error in %s at %d", __FILE__, __LINE__);exit(1);}
	MKL_INT opt_define = MKL_DSS_NON_SYMMETRIC;
	result = DSS_DEFINE_STRUCTURE(handleP, opt_define, csrBpRow, lengthP, lengthP, csrBpCol, csrBpRow[H_NPV+H_NPQ]);if(result != MKL_DSS_SUCCESS){printf("MKL Library error in %s at %d", __FILE__, __LINE__);exit(1);}
	int *permP = (int*) MKL_malloc(sizeof(int) * (H_NPQ + H_NPV), 64);
	for(int i = 0; i < (H_NPQ + H_NPV); i++){
		permP[i] = i;
	}
	MKL_INT opt_REORDER = MKL_DSS_AUTO_ORDER;
	result = DSS_REORDER(handleP, opt_REORDER,permP);if(result != MKL_DSS_SUCCESS){printf("MKL Library error in %s at %d", __FILE__, __LINE__);exit(1);}

	_MKL_DSS_HANDLE_t handleQ;
	int *permQ = (int*) MKL_malloc(sizeof(int) * H_NPQ, 64);
	{
	MKL_INT opt;
	opt  = MKL_DSS_MSG_LVL_WARNING;
	opt += MKL_DSS_ZERO_BASED_INDEXING;
	MKL_INT result;
	result = DSS_CREATE(handleQ, opt); if(result != MKL_DSS_SUCCESS){printf("MKL Library error in %s at %d", __FILE__, __LINE__);exit(1);}
	MKL_INT opt_define = MKL_DSS_NON_SYMMETRIC;
	result = DSS_DEFINE_STRUCTURE(handleQ, opt_define, csrBppRow, H_NPQ, H_NPQ, csrBppCol, csrBppRow[H_NPQ]);if(result != MKL_DSS_SUCCESS){printf("MKL Library error in %s at %d", __FILE__, __LINE__);exit(1);}
	for(int i = 0; i < H_NPQ; i++){
		permQ[i] = i;
	}
	MKL_INT opt_REORDER = MKL_DSS_AUTO_ORDER;
	result = DSS_REORDER(handleQ, opt_REORDER,permQ);if(result != MKL_DSS_SUCCESS){printf("MKL Library error in %s at %d", __FILE__, __LINE__);exit(1);}
	}



	double *dVa = (double*) malloc(sizeof(double) * (H_NPV + H_NPQ));
	double *dVm = (double*) malloc(sizeof(double) * H_NPQ);

	int i = 0;
	while (!converged && i < MAX_IT_FD) {
		i++;

		//-----  do P iteration, update Va  -----
		MKL_INT opt_FACTOR = MKL_DSS_POSITIVE_DEFINITE;
		result = DSS_FACTOR_REAL(handleP, opt_FACTOR, csrBpVal);if(result != MKL_DSS_SUCCESS){printf("MKL Library error in %s at %d", __FILE__, __LINE__);exit(1);}
		MKL_INT opt_DEFAULT = MKL_DSS_DEFAULTS;
		MKL_INT nrhs = 1;
		result = DSS_SOLVE_REAL(handleP, opt_DEFAULT, P, nrhs, dVa);if(result != MKL_DSS_SUCCESS){printf("MKL Library error in %s at %d", __FILE__, __LINE__);exit(1);}
#ifdef DEBUG
		printf("P:\n");
		for(int j = 0; j < (H_NPV + H_NPQ); j++){
			printf("\t%.4e\n", P[j]);
		}

		printf("dVa:\n");
		for(int j = 0; j < (H_NPV + H_NPQ); j++){
			printf("\t%.4e\n", -dVa[j]);
		}
#endif
		//-----  update voltage  -----

		mkl_updateVoltage_dVa(pv, pq, V, dVa);


#ifdef DEBUG
	printf("V = \n");
	for(int i = 0; i < H_NBUS; i++)
	{
		printf("\t%.4e %c %.4ei\n", V[i].x, ((V[i].y < 0) ? '-' : '+'), ((V[i].y < 0) ? -V[i].y : V[i].y));
	}
#endif
		err = mkl_checkConvergence(
				buses,
				pv,
				pq,
				nnzYbus,
				csrRowPtrYbus,
				csrColIndYbus,
				csrValYbus,
				V,
				P,
				Q);
		if(err < EPS){
			converged = true;
			break;
		}


		// -----  do Q iteration, update Vm  -----
		{
		MKL_INT opt_FACTOR = MKL_DSS_POSITIVE_DEFINITE;
		result = DSS_FACTOR_REAL(handleQ, opt_FACTOR, csrBppVal);if(result != MKL_DSS_SUCCESS){printf("MKL Library error in %s at %d", __FILE__, __LINE__);exit(1);}
		MKL_INT opt_DEFAULT = MKL_DSS_DEFAULTS;
		MKL_INT nrhs = 1;
		result = DSS_SOLVE_REAL(handleQ, opt_DEFAULT, Q, nrhs, dVm);if(result != MKL_DSS_SUCCESS){printf("MKL Library error in %s at %d", __FILE__, __LINE__);exit(1);}
		}

		//-----  update voltage  -----
		mkl_updateVoltage_dVm(pv, pq, V, dVm);


#ifdef DEBUG
	printf("V = \n");
	for(int i = 0; i < H_NBUS; i++)
	{
		printf("\t%.4e %c %.4ei\n", V[i].x, ((V[i].y < 0) ? '-' : '+'), ((V[i].y < 0) ? -V[i].y : V[i].y));
	}
#endif

	err = mkl_checkConvergence(
				buses,
				pv,
				pq,
				nnzYbus,
				csrRowPtrYbus,
				csrColIndYbus,
				csrValYbus,
				V,
				P,
				Q);
		if(err < EPS){
			converged = true;
		}
	}
	free(dVa);
	free(dVm);
	MKL_free(permP);
	MKL_free(permQ);
	return converged;
}
#endif /* FDPF_CUH_ */
