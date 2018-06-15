/*
 * makeB.cuh
 *
 *  Created on: 13/11/2015
 *      Author: Igor M. Araújo
 */

#ifndef MAKEB_CUH_
#define MAKEB_CUH_

__global__ void hybrid_computeNnzBpp(
		double *BppIndex,
		int *BppCol,
		int *BppRow,
		cuDoubleComplex *Ybus,
		int *YbusCol,
		int *YbusRow,
		Bus *buses,
		unsigned int *pq,
		int nnzYbus)
{
	int id = ID();
	if(id < nnzYbus)
	{
		int j = YbusCol[id];
		int i = 0;
		while( !(id >= YbusRow[i] && id < YbusRow[i + 1]) ){
			i++;
		}
		BppIndex[id] = id;
		BppCol[id] = ( buses[j].indicePVPQ < D_NPV) ? D_NPQ : buses[j].indicePVPQ - D_NPV;
		BppRow[id] = ( buses[i].indicePVPQ < D_NPV) ? D_NPQ : buses[i].indicePVPQ - D_NPV;
		BppRow[id] = ( BppCol[id] == D_NPQ) ? D_NPQ : BppRow[id];
	}
}

__global__ void hybrid_computeBpp(
		double *BppIndex,
		double *BppVal,
		cuDoubleComplex *Ybus,
		int nnzYbus,
		int nnzBpp,
		int total)
{
	int id = ID();
	if( id < nnzBpp * total)
	{
		int i =  id - nnzBpp * ( id / nnzBpp);//id % nnzBpp;
		int n = id / nnzBpp;

		BppVal[id] = -cuCimag(Ybus[n * nnzYbus + ((int) BppIndex[i])]);
	}
}

__global__ void hybrid_computeBp(
		double *BpVal,
		int *BpCol,
		int *BpRow,
		cuDoubleComplex *Ybus,
		int *YbusCol,
		int *YbusRow,
		Bus *buses,
		unsigned int *pq,
		unsigned int *pv,
		int nnzYbus)
{
	int id = ID();
	if(id < nnzYbus)
	{
		int j = YbusCol[id];
		int i = 0;
		while( !(id >= YbusRow[i] && id < YbusRow[i + 1]) ){
			i++;
		}

		BpVal[id] = -cuCimag(Ybus[id]);
		BpCol[id] = ( buses[j].indicePVPQ == -1) ? D_NBUS - 1 : buses[j].indicePVPQ;
		BpRow[id] = ( buses[i].indicePVPQ == -1) ? D_NBUS - 1 : buses[i].indicePVPQ;
		BpRow[id] = ( BpCol[id] == D_NBUS - 1) ? D_NBUS - 1 : BpRow[id];
	}
}

__global__ void hybrid_zeroBusesAndBranches(Bus *buses, Branch *branches, int ALG){
	int id = ID();
	if(id < D_NBRANCH){
		branches[id].indiceEstrutura = -1;
		branches[id].B = 0.0;
		branches[id].tap = 0.0;
		if(ALG == 1){ // FDXB
			branches[id].R = 0.0;
		}
		if(id < D_NBUS){
			buses[id].indiceEstrutura = -1;
			buses[id].Bsh = 0.0;
		}
	}
}

__global__ void hybrid_zeroBranches(Branch *branches, int ALG){
	int id = ID();
	if(id < D_NBRANCH){
		branches[id].shift = 0.0;
		if(ALG == 2){ // FDBX
			branches[id].R = 0.0;
		}
	}
}

void hybrid_makeB(int nTest, int sizeEstrutura) {
	checkCudaErrors(cudaMalloc((void**) &tmpCsrValYbus, nnzYbus  * H_NTESTS	* sizeof(		cuDoubleComplex		)));
	checkCudaErrors(cudaMalloc((void**) &tmpCsrColIndYbus, nnzYbus 	* sizeof(		int		)));

	checkCudaErrors(cudaMemcpyAsync(tmpBuses, device_buses, H_NBUS * sizeof(Bus), cudaMemcpyDeviceToDevice));
	checkCudaErrors(cudaMemcpyAsync(tmpBranches, device_branches, H_NBRANCH * sizeof(Branch), cudaMemcpyDeviceToDevice));

	checkCudaErrors(cudaMemcpyAsync(tmpCsrValYbus, csrValYbus, nnzYbus * H_NTESTS * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice));
	checkCudaErrors(cudaMemcpyAsync(tmpCsrColIndYbus, csrColIndYbus, nnzYbus * sizeof(int), cudaMemcpyDeviceToDevice));
	checkCudaErrors(cudaMemcpyAsync(tmpCsrRowPtrYbus, csrRowPtrYbus, (H_NBUS + 1) * sizeof(int), cudaMemcpyDeviceToDevice));

	checkCudaErrors(cudaMemcpyAsync(tmpCsrValYf, csrValYf, nnzYf  * H_NTESTS * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice));
	checkCudaErrors(cudaMemcpyAsync(tmpCsrColIndYf, csrColIndYf, nnzYf * sizeof(int), cudaMemcpyDeviceToDevice));
	checkCudaErrors(cudaMemcpyAsync(tmpCsrRowPtrYf, csrRowPtrYf, (H_NBRANCH + 1) * sizeof(int), cudaMemcpyDeviceToDevice));

	checkCudaErrors(cudaMemcpyAsync(tmpCsrValYt, csrValYt, nnzYt  * H_NTESTS * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice));
	checkCudaErrors(cudaMemcpyAsync(tmpCsrColIndYt, csrColIndYt, nnzYt * sizeof(int), cudaMemcpyDeviceToDevice));
	checkCudaErrors(cudaMemcpyAsync(tmpCsrRowPtrYt, csrRowPtrYt, (H_NBRANCH + 1) * sizeof(int), cudaMemcpyDeviceToDevice));
	tmpNnzYbus = nnzYbus;
	tmpNnzYf = nnzYf;
	tmpNnzYt = nnzYt;
	checkCudaErrors(cudaDeviceSynchronize());

	hybrid_zeroBusesAndBranches<<<BLOCKS(H_NBRANCH, H_THREADS), H_THREADS>>>(tmpBuses, tmpBranches, H_ALG);
	checkCudaErrors(cudaDeviceSynchronize());

	hybrid_makeYbus(0, 0, tmpBuses, tmpBranches);

	checkCudaErrors(cudaDeviceSynchronize());

	checkCudaErrors(cudaMalloc((void**) &csrBpCol, nnzYbus * sizeof(int)));
	checkCudaErrors(cudaMalloc((void**) &csrBpVal, nnzYbus * sizeof(double)));
	checkCudaErrors(cudaMalloc((void**) &cooBpVal, nnzYbus * sizeof(double)));
	checkCudaErrors(cudaMalloc((void**) &cooBpCol, nnzYbus * sizeof(int)));
	checkCudaErrors(cudaMalloc((void**) &cooBpRow, nnzYbus * sizeof(int)));

	hybrid_computeBp<<<BLOCKS(nnzYbus, H_THREADS), H_THREADS>>>(
			cooBpVal,
			cooBpCol,
			cooBpRow,
			csrValYbus,
			csrColIndYbus,
			csrRowPtrYbus,
			tmpBuses,
			device_pq,
			device_pv,
			nnzYbus);
	checkCudaErrors(cudaDeviceSynchronize());

	size_t before = pBufferSizeInBytes;
	checkCudaErrors(cusparseXcoosort_bufferSizeExt(sparseHandle, H_NPV + H_NPQ + 1, H_NPV + H_NPQ + 1, nnzYbus, cooBpRow, cooBpCol, &pBufferSizeInBytes));
	if(pBufferSizeInBytes > before){
		checkCudaErrors(cudaFree(pBuffer));
		checkCudaErrors(cudaMalloc((void**) &pBuffer	, pBufferSizeInBytes * sizeof(char)));
	}
	if(nnzYbus > nPermutation){
		nPermutation = nnzYbus;
		checkCudaErrors(cudaFree(permutation));
		checkCudaErrors(cudaMalloc((void**) &permutation, nnzYbus * sizeof(int)));
	}

	checkCudaErrors(cusparseCreateIdentityPermutation(sparseHandle, nnzYbus, permutation));
	checkCudaErrors(cusparseXcoosortByRow(sparseHandle, H_NPV + H_NPQ + 1, H_NPV + H_NPQ + 1, nnzYbus, cooBpRow, cooBpCol, permutation, pBuffer));
	checkCudaErrors(cusparseDgthr(sparseHandle, nnzYbus, cooBpVal, csrBpVal, permutation, CUSPARSE_INDEX_BASE_ZERO));

	// #1.3 Convert Matrix Cf in Coordinate Format(COO) to Compressed Sparse Row Format(CSR)
	checkCudaErrors(cusparseXcoo2csr(sparseHandle, (const int*) cooBpRow, nnzYbus, H_NPV + H_NPQ, csrBpRow, CUSPARSE_INDEX_BASE_ZERO));
	checkCudaErrors(cudaMemcpy(csrBpCol, cooBpCol, nnzYbus * sizeof(int), cudaMemcpyDeviceToDevice));

#ifdef DEBUG
	int *h_row = (int*) malloc(sizeof(int) * (H_NPV + H_NPQ + 1));
	int *h_col = (int*) malloc(sizeof(int) * nnzYbus);
	double *h_val = (double*) malloc(sizeof(double) * nnzYbus);
	cudaMemcpy(h_row, csrBpRow, sizeof(int) * (H_NPV + H_NPQ + 1), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_col, csrBpCol, sizeof(int) * nnzYbus, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_val, csrBpVal, sizeof(double) * nnzYbus, cudaMemcpyDeviceToHost);
	printf("Bp = \n");
	printf("\tCompressed Sparse Column(rows = %d, cols = %d, nnz = %d [%.2lf])\n",H_NPV + H_NPQ, H_NPV + H_NPQ,nnzYbus, nnzYbus * 100.0f / ((H_NPV + H_NPQ) * (H_NPV + H_NPQ)));
	for(int j = 0; j < H_NPV + H_NPQ; j++){
		for(int i = 0; i < H_NPV + H_NPQ; i++){
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
#endif

	checkCudaErrors(cudaMemcpy(tmpBranches, device_branches, H_NBRANCH * sizeof(Branch), cudaMemcpyDeviceToDevice));
	hybrid_zeroBranches<<<BLOCKS(H_NBRANCH, H_THREADS), H_THREADS>>>(tmpBranches, H_ALG);
	checkCudaErrors(cudaDeviceSynchronize());


	for(int i = 0; i < nTest; i++){
		hybrid_makeYbus(i, sizeEstrutura, device_buses, tmpBranches);
	}
	checkCudaErrors(cudaDeviceSynchronize());
	double *cooBppVal2;
	checkCudaErrors(cudaMalloc((void**) &csrBppCol, nnzYbus * sizeof(int)));
	checkCudaErrors(cudaMalloc((void**) &cooBppCol, nnzYbus * sizeof(int)));
	checkCudaErrors(cudaMalloc((void**) &cooBppVal, nnzYbus * sizeof(double)));
	checkCudaErrors(cudaMalloc((void**) &cooBppVal2, nnzYbus * sizeof(double)));
	checkCudaErrors(cudaMalloc((void**) &cooBppRow, nnzYbus * sizeof(int)));

	hybrid_computeNnzBpp<<<BLOCKS(nnzYbus, H_THREADS), H_THREADS>>>(
			cooBppVal,
			cooBppCol,
			cooBppRow,
			csrValYbus,
			csrColIndYbus,
			csrRowPtrYbus,
			device_buses,
			device_pq,
			nnzYbus);
	checkCudaErrors(cudaDeviceSynchronize());

	before = pBufferSizeInBytes;
	checkCudaErrors(cusparseXcoosort_bufferSizeExt(sparseHandle, H_NPQ + 1, H_NPQ + 1, nnzYbus, cooBppRow, cooBppCol, &pBufferSizeInBytes));
	if(pBufferSizeInBytes > before){
		checkCudaErrors(cudaFree(pBuffer));
		checkCudaErrors(cudaMalloc((void**) &pBuffer	, pBufferSizeInBytes * sizeof(char)));
	}
	if(nnzYbus > nPermutation){
		nPermutation = nnzYbus;
		checkCudaErrors(cudaFree(permutation));
		checkCudaErrors(cudaMalloc((void**) &permutation, nnzYbus * sizeof(int)));
	}

	checkCudaErrors(cusparseCreateIdentityPermutation(sparseHandle, nnzYbus, permutation));checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cusparseXcoosortByRow(sparseHandle, H_NPQ + 1, H_NPQ + 1, nnzYbus, cooBppRow, cooBppCol, permutation, pBuffer));
	checkCudaErrors(cusparseDgthr(sparseHandle, nnzYbus, cooBppVal, cooBppVal2, permutation, CUSPARSE_INDEX_BASE_ZERO));

	// #1.3 Convert Matrix Cf in Coordinate Format(COO) to Compressed Sparse Row Format(CSR)
	checkCudaErrors(cusparseXcoo2csr(sparseHandle, (const int*) cooBppRow, nnzYbus, H_NPQ + 1, csrBppRow, CUSPARSE_INDEX_BASE_ZERO));
	checkCudaErrors(cudaMemcpy(csrBppCol, cooBppCol, nnzYbus * sizeof(int), cudaMemcpyDeviceToDevice));

	int nnzBpp = 0;
	checkCudaErrors(cudaMemcpy(&nnzBpp, csrBppRow + H_NPQ, sizeof(int), cudaMemcpyDeviceToHost));

	checkCudaErrors(cudaMalloc((void**) &csrBppVal, H_NTESTS * nnzBpp * sizeof(double)));


	hybrid_computeBpp<<<BLOCKS(nnzBpp * H_NTESTS, H_THREADS), H_THREADS>>>(
			cooBppVal2,
			csrBppVal,
			csrValYbus,
			nnzYbus,
			nnzBpp,
			H_NTESTS);
	checkCudaErrors(cudaDeviceSynchronize());

#ifdef DEBUG
	{
	int *h_row = (int*) malloc(sizeof(int) * (H_NPQ + 1));
	int *h_col = (int*) malloc(sizeof(int) * nnzBpp);
	double *h_val = (double*) malloc(sizeof(double) * nnzBpp);
	cudaMemcpy(h_row, csrBppRow, sizeof(int) * (H_NPQ + 1), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_col, csrBppCol, sizeof(int) * nnzBpp, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_val, csrBppVal, sizeof(double) * nnzBpp, cudaMemcpyDeviceToHost);
	printf("Bpp = \n");
	printf("\tCompressed Sparse Column(rows = %d, cols = %d, nnz = %d [%.2lf])\n",H_NPQ, H_NPQ, nnzBpp, nnzBpp * 100.0f / ((H_NPQ) * (H_NPQ)));
	for(int j = 0; j < H_NPQ; j++){
		for(int i = 0; i < H_NPQ; i++){
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

	checkCudaErrors(cudaMemcpyAsync(csrValYbus, tmpCsrValYbus, nnzYbus * H_NTESTS * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice));
	checkCudaErrors(cudaMemcpyAsync(csrColIndYbus, tmpCsrColIndYbus, nnzYbus * sizeof(int), cudaMemcpyDeviceToDevice));
	checkCudaErrors(cudaMemcpyAsync(csrRowPtrYbus, tmpCsrRowPtrYbus, (H_NBUS + 1) * sizeof(int), cudaMemcpyDeviceToDevice));

	checkCudaErrors(cudaMemcpyAsync(csrValYf, tmpCsrValYf, nnzYf  * H_NTESTS * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice));
	checkCudaErrors(cudaMemcpyAsync(csrColIndYf, tmpCsrColIndYf, nnzYf * sizeof(int), cudaMemcpyDeviceToDevice));
	checkCudaErrors(cudaMemcpyAsync(csrRowPtrYf, tmpCsrRowPtrYf, (H_NBRANCH + 1) * sizeof(int), cudaMemcpyDeviceToDevice));

	checkCudaErrors(cudaMemcpyAsync(csrValYt, tmpCsrValYt, nnzYt  * H_NTESTS * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice));
	checkCudaErrors(cudaMemcpyAsync(csrColIndYt, tmpCsrColIndYt, nnzYt * sizeof(int), cudaMemcpyDeviceToDevice));
	checkCudaErrors(cudaMemcpyAsync(csrRowPtrYt, tmpCsrRowPtrYt, (H_NBRANCH + 1) * sizeof(int), cudaMemcpyDeviceToDevice));
	nnzYbus = tmpNnzYbus;
	nnzYf = tmpNnzYf;
	nnzYt = tmpNnzYt;
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaFree(cooBppVal2));
}

void mkl_zeroBusesAndBranches(Bus *buses, Branch *branches, int ALG){
	for(int id = 0; id < H_NBRANCH; id++){
		branches[id].indiceEstrutura = -1;
		branches[id].B = 0.0;
		branches[id].tap = 0.0;
		if(ALG == 1){ // FDXB
			branches[id].R = 0.0;
		}
		if(id < H_NBUS){
			buses[id].indiceEstrutura = -1;
			buses[id].Bsh = 0.0;
		}
	}
}

void mkl_computeBp(
		double *BpVal,
		int *BpCol,
		int *BpRow,
		cuDoubleComplex *Ybus,
		int *YbusCol,
		int *YbusRow,
		Bus *buses,
		unsigned int *pq,
		unsigned int *pv,
		int nnzYbus)
{
	BpRow[0] = 0;
	for(int i = 0; i < H_NPV+H_NPQ; i++){
		int indiceI = (i < H_NPV) ? pv[i] : pq[i - H_NPV];
		BpRow[i + 1] = BpRow[i];
		for(int j = 0; j < H_NPV+H_NPQ; j++){
			int indiceJ = (j < H_NPV) ? pv[j] : pq[j - H_NPV];
			for(int k = YbusRow[indiceI] - BASE_INDEX, endFor = YbusRow[indiceI + 1] - BASE_INDEX; k < endFor; k++){
				int jYbus = YbusCol[k] - BASE_INDEX;
				if(jYbus == indiceJ){
					BpCol[BpRow[i+1]] = j;
					BpVal[BpRow[i+1]] = -cuCimag(Ybus[k]);
					BpRow[i+1]++;
				}
			}
		}
	}
}

void mkl_computeBpp(
		double *BppVal,
		int *BppCol,
		int *BppRow,
		cuDoubleComplex *Ybus,
		int *YbusCol,
		int *YbusRow,
		unsigned int *pq)
{
	BppRow[0] = 0;
	for(int i = 0; i < H_NPQ; i++){
		int indiceI = pq[i];
		BppRow[i + 1] = BppRow[i];
		for(int j = 0; j < H_NPQ; j++){
			int indiceJ = pq[j];
			for(int k = YbusRow[indiceI] - BASE_INDEX, endFor = YbusRow[indiceI + 1] - BASE_INDEX; k < endFor; k++){
				int jYbus = YbusCol[k] - BASE_INDEX;
				if(jYbus == indiceJ){
					BppCol[BppRow[i+1]] = j;
					BppVal[BppRow[i+1]] = -cuCimag(Ybus[k]);
					BppRow[i+1]++;
				}
			}
		}
	}
}

void mkl_zeroBranches(Branch *branches, int ALG){
	for(int id = 0; id < H_NBRANCH; id++){
		branches[id].shift = 0.0;
		if(ALG == 2){ // FDBX
			branches[id].R = 0.0;
		}
	}
}

void mkl_makeB(vector<pso::Particula::Estrutura> estrutura, pso::Particula particula) {
	tmpCsrValYbus = (cuDoubleComplex*) MKL_malloc(nnzYbus 	* sizeof(		cuDoubleComplex		), 64);
	tmpCsrColIndYbus = (int*) MKL_malloc(nnzYbus 	* sizeof(		int		), 64);

	memcpy(tmpBuses, buses, H_NBUS * sizeof(Bus));
	memcpy(tmpBranches, branches, H_NBRANCH * sizeof(Branch));

	memcpy(tmpCsrValYbus, csrValYbus, nnzYbus * sizeof(cuDoubleComplex));
	memcpy(tmpCsrColIndYbus, csrColIndYbus, nnzYbus * sizeof(int));
	memcpy(tmpCsrRowPtrYbus, csrRowPtrYbus, (H_NBUS + 1) * sizeof(int));

	memcpy(tmpCsrValYf, csrValYf, nnzYf * sizeof(cuDoubleComplex));
	memcpy(tmpCsrColIndYf, csrColIndYf, nnzYf * sizeof(int));
	memcpy(tmpCsrRowPtrYf, csrRowPtrYf, (H_NBRANCH + 1) * sizeof(int));

	memcpy(tmpCsrValYt, csrValYt, nnzYt * sizeof(cuDoubleComplex));
	memcpy(tmpCsrColIndYt, csrColIndYt, nnzYt * sizeof(int));
	memcpy(tmpCsrRowPtrYt, csrRowPtrYt, (H_NBRANCH + 1) * sizeof(int));
	tmpNnzYbus = nnzYbus;
	tmpNnzYf = nnzYf;
	tmpNnzYt = nnzYt;

	mkl_zeroBusesAndBranches(tmpBuses, tmpBranches, H_ALG);

	mkl_makeYbus(estrutura, particula, tmpBuses, tmpBranches);

	csrBpCol = (int*) MKL_malloc(nnzYbus * sizeof(int), 64);
	csrBpVal = (double*) MKL_malloc(nnzYbus * sizeof(double), 64);

	mkl_computeBp(
			csrBpVal,
			csrBpCol,
			csrBpRow,
			csrValYbus,
			csrColIndYbus,
			csrRowPtrYbus,
			tmpBuses,
			pq,
			pv,
			nnzYbus);


#ifdef DEBUG
	printf("Bp = \n");
	printf("\tCompressed Sparse Column(rows = %d, cols = %d, nnz = %d [%.2lf])\n",H_NPV + H_NPQ, H_NPV + H_NPQ,csrBpRow[H_NPV + H_NPQ], csrBpRow[H_NPV + H_NPQ] * 100.0f / ((H_NPV + H_NPQ) * (H_NPV + H_NPQ)));
	for(int j = 0; j < H_NPV + H_NPQ; j++){
		for(int i = 0; i < H_NPV + H_NPQ; i++){
			for(int k = csrBpRow[i]; k < csrBpRow[i + 1]; k++){
				if(j == csrBpCol[k]){
					double value = csrBpVal[k];
					printf("\t(%d, %d)\t->\t%.4e\n", i+1, j+1, value);
					break;
				}
			}
		}
	}
#endif

	memcpy(tmpBranches, branches, H_NBRANCH * sizeof(Branch));
	mkl_zeroBranches(tmpBranches, H_ALG);

	mkl_makeYbus(estrutura, particula, buses, tmpBranches);

	csrBppCol = (int*) MKL_malloc(nnzYbus * sizeof(int), 64);
	csrBppVal = (double*) MKL_malloc(nnzYbus * sizeof(double), 64);


	mkl_computeBpp(
			csrBppVal,
			csrBppCol,
			csrBppRow,
			csrValYbus,
			csrColIndYbus,
			csrRowPtrYbus,
			pq);


#ifdef DEBUG
	{
	printf("Bpp = \n");
	printf("\tCompressed Sparse Column(rows = %d, cols = %d, nnz = %d [%.2lf])\n",H_NPQ, H_NPQ, csrBppRow[H_NPQ], csrBppRow[H_NPQ] * 100.0f / ((H_NPQ) * (H_NPQ)));
	for(int j = 0; j < H_NPQ; j++){
		for(int i = 0; i < H_NPQ; i++){
			for(int k = csrBppRow[i]; k < csrBppRow[i + 1]; k++){
				if(j == csrBppCol[k]){
					double value = csrBppVal[k];
					printf("\t(%d, %d)\t->\t%.4e\n", i+1, j+1, value);
					break;
				}
			}
		}
	}
	}
#endif

	memcpy(csrValYbus, tmpCsrValYbus, nnzYbus * sizeof(cuDoubleComplex));
	memcpy(csrColIndYbus, tmpCsrColIndYbus, nnzYbus * sizeof(int));
	memcpy(csrRowPtrYbus, tmpCsrRowPtrYbus, (H_NBUS + 1) * sizeof(int));

	memcpy(csrValYf, tmpCsrValYf, nnzYf  * sizeof(cuDoubleComplex));
	memcpy(csrColIndYf, tmpCsrColIndYf, nnzYf * sizeof(int));
	memcpy(csrRowPtrYf, tmpCsrRowPtrYf, (H_NBRANCH + 1) * sizeof(int));

	memcpy(csrValYt, tmpCsrValYt, nnzYt  * sizeof(cuDoubleComplex));
	memcpy(csrColIndYt, tmpCsrColIndYt, nnzYt * sizeof(int));
	memcpy(csrRowPtrYt, tmpCsrRowPtrYt, (H_NBRANCH + 1) * sizeof(int));
	nnzYbus = tmpNnzYbus;
	nnzYf = tmpNnzYf;
	nnzYt = tmpNnzYt;
}

#endif /* MAKEB_CUH_ */
