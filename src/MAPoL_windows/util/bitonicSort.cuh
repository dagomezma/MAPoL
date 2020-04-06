/*
 * bitonicSort.cuh
 *
 *  Created on: 30/10/2015
 *      Author: Igor M. Araújo
 */

#ifndef BITONICSORT_CUH_
#define BITONICSORT_CUH_

template<typename type, class Comparator>
__global__ void bitonic_sort_step(type *dev_values, type *aux, int j, int k,
		int n, int size) {
	unsigned int i, ixj; /* Sorting partners: i and ixj */

	i = threadIdx.x + blockDim.x * blockIdx.x;

	ixj = i ^ j;
	Comparator C;

	if (ixj >= n)
		return;

	/* The threads with the lowest ids sort the array. */
	if ((ixj) > i) {
		type A = (i < size) ? dev_values[i] : aux[i - size];
		type B = (ixj < size) ? dev_values[ixj] : aux[ixj - size];
#ifdef DEBUG
		int debug = 0;
#endif
		if ((i & k) == 0) {
			/* Sort ascending */
			if (C.compare(A, B) > 0) {
#ifdef DEBUG
				debug++;
#endif
				/* exchange(i,ixj); */
				if (i < size) {
					dev_values[i] = B;
				} else {
					aux[i - size] = B;
				}
				if (ixj < size) {
					dev_values[ixj] = A;
				} else {
					aux[ixj - size] = A;
				}

			}
		}
		if ((i & k) != 0) {
			/* Sort descending */
			if (C.compare(A, B) < 0) {
#ifdef DEBUG
				debug++;
#endif
				/* exchange(i,ixj); */
				if (i < size) {
					dev_values[i] = B;
				} else {
					aux[i - size] = B;
				}
				if (ixj < size) {
					dev_values[ixj] = A;
				} else {
					aux[ixj - size] = A;
				}

			}
		}
#ifdef DEBUG
		printf("[BitonicSort] -> checked %d(%1.3f) %d(%1.3f) %s with k = %d so sort %s\n", i+1, A, ixj+1, B, (debug > 0)?"swaped":"not swaped", k, (i&k)?"descending":"ascending");
#endif
	}
#ifdef DEBUG
	__syncthreads();
	if (i == 0) {
		printf("[BitonicSort] -> ");
		for (int i = 0; i < size; i++) {
			printf(" %1.3f", dev_values[i]);
		}
		for (int i = 0; i < n - size; i++) {
			printf(" %1.3f", aux[i]);
		}
		printf("\n");
	}
#endif
}

/**
 * Inplace bitonic sort using CUDA.
 */
template<typename type, class Comparator>
__host__ __device__ void bitonic_sort(type *values, int size, type &max_value) {

#ifdef __CUDA_ARCH__
//	unsigned int id = threadIdx.x + blockDim.x * blockIdx.x;
//	if (id == 0) {

		int n;
		for (n = 2; n < size; n <<= 1)
			;
		type *aux;
		if (n != size) {
			for (int i = 0; i < (n - size); i++) {
				aux[i] = max_value;
			}
		}

		dim3 blocks(BLOCKS(n, D_THREADS), 1); /* Number of blocks   */
		dim3 threads(D_THREADS, 1); /* Number of threads  */

		int j, k;

		/* Major step */
		for (k = 2; k <= n; k <<= 1) {
			/* Minor step */

			for (j = k >> 1; j > 0; j = j >> 1) {

				bitonic_sort_step<type, Comparator> <<<blocks, threads>>>(
						values, aux, j, k, n, size);
			}
		}
		if (n != size)
			cudaFree(aux);
//	}
	cudaDeviceSynchronize();
	__syncthreads();

#else

	type *dev_values;

	cudaMalloc((void**) &dev_values, size * sizeof(type));
	cudaMemcpy(dev_values, values, size * sizeof(type), cudaMemcpyHostToDevice);

	int n;
	for (n = 2; n < size; n <<= 1)
	;
	type *aux;
	if (n != size) {
		cudaMalloc((void**) &aux, (n - size) * sizeof(type));
		type *h_aux = (type*) malloc((n - size) * sizeof(type));
		for (int i = 0; i < (n - size); i++) {
			h_aux[i] = max_value;
		}
		cudaMemcpy(aux, h_aux, (n - size) * sizeof(type),
				cudaMemcpyHostToDevice);
		free(h_aux);
	}

	dim3 blocks(BLOCKS(n, D_THREADS), 1); /* Number of blocks   */
	dim3 threads(D_THREADS, 1); /* Number of threads  */

	int j, k;

	/* Major step */
	for (k = 2; k <= n; k <<= 1) {
		/* Minor step */

		for (j = k >> 1; j > 0; j = j >> 1) {

			bitonic_sort_step<type, Comparator> <<<blocks, threads>>>(
					dev_values, aux, j, k, n, size);
		}
	}

	cudaMemcpy(values, dev_values, size * sizeof(type), cudaMemcpyDeviceToHost);
	cudaFree(dev_values);
	if (n != size)
	cudaFree(aux);

#endif
}

#endif /* BITONICSORT_CUH_ */
