//
// Created by igor on 23/11/15.
//

#ifndef CUSPARSE_CUSPARSE_H
#define CUSPARSE_CUSPARSE_H

#include "quicksort.h"
#include "bitonicSort.cuh"

#define CSC 1
#define CSR 2

using namespace std;

template<typename type>
struct element {
	unsigned long lin;
	unsigned long col;
	type value;

	__host__ __device__ element() {
		this->lin = 0xFFFFFFu;
		this->col = 0xFFFFFFu;
	}
	__host__ __device__ element(unsigned long lin, unsigned long col,
			type value) {
		this->lin = lin;
		this->col = col;
		this->value = value;
	}
};

template<typename type>
class ComparatorCSR {
public:
	__host__ __device__ int compare(element<type> &s1, element<type> &s2) {
		if (s1.lin > s2.lin) {
			return 1;
		} else if (s1.lin == s2.lin) {
			if (s1.col > s2.col) {
				return 1;
			} else if (s1.col < s2.col) {
				return -1;
			} else {
				return 0;
			}
		} else {
			return -1;
		}
	}
};

template<typename type>
class ComparatorCSC {
public:
	__host__ __device__ int compare(element<type> &s1, element<type> &s2) {
		if (s1.col > s2.col) {
			return 1;
		} else if (s1.col == s2.col) {
			if (s1.lin > s2.lin) {
				return 1;
			} else if (s1.lin < s2.lin) {
				return -1;
			} else {
				return 0;
			}
		} else {
			return -1;
		}
	}
};

template<typename type>
class cuSparse {
private:
//	int lock;

public:
	element<type> *vetor;
	unsigned long *index;

	unsigned long lin;
	unsigned long col;
	int compress;
	bool orderned;

	__host__ __device__ void order();

	__host__ __device__ cuSparse();
	__host__ __device__ cuSparse(unsigned long, unsigned long);
	__host__ __device__ cuSparse(const cuSparse &);
	__host__ __device__ ~cuSparse();
	__host__    __device__ cuSparse& operator=(const cuSparse &);
	__host__    __device__ cuSparse operator+(const cuSparse &);
	__host__    __device__ cuSparse operator*(cuSparse &);
	__host__    __device__ type operator()(const unsigned long,
			const unsigned long);
	__host__ __device__ void operator()(const unsigned long,
			const unsigned long, const type);
	void print();
};

__global__ void cuSadd(cuSparse<cuDoubleComplex> *A,
		cuSparse<cuDoubleComplex> *B, cuSparse<cuDoubleComplex> *C, int *lock) {

	unsigned int id = ID();

	if (id >= A->lin) {
		return;
	}

	int indexA = 0;
	int indexB = 0;
	int sizeA = A->index[id + 1] - A->index[id];
	int sizeB = B->index[id + 1] - B->index[id];
	while (indexA < sizeA || indexB < sizeB) {
		if (indexB == sizeB) {
			LOCK(lock);
			C[0](id, A->vetor[A->index[id] + indexA].col,
					A->vetor[A->index[id] + indexA].value);
			UNLOCK(lock);
			indexA++;
		} else if (indexA == sizeA
				|| A->vetor[A->index[id] + indexA].col
						> B->vetor[B->index[id] + indexB].col) {
			LOCK(lock);
			C[0](id, B->vetor[B->index[id] + indexB].col,
					B->vetor[B->index[id] + indexB].value);
			UNLOCK(lock);
			indexB++;
		} else if (A->vetor[A->index[id] + indexA].col
				< B->vetor[B->index[id] + indexB].col) {
			LOCK(lock);
			C[0](id, A->vetor[A->index[id] + indexA].col,
					A->vetor[A->index[id] + indexA].value);
			UNLOCK(lock);
			indexA++;
		} else {
			LOCK(lock);
			C[0](id, A->vetor[A->index[id] + indexA].col,
					cuCadd(A->vetor[A->index[id] + indexA].value,
							B->vetor[B->index[id] + indexB].value));
			UNLOCK(lock);
			indexA++;
			indexB++;
		}
	}
}

__global__ void cuSmul(cuSparse<cuDoubleComplex> *A,
		cuSparse<cuDoubleComplex> *B, cuSparse<cuDoubleComplex> *C, int *lock) {

	unsigned id = ID();

	unsigned int i = id / A->lin;
	unsigned int j = id % B->col;

	if (i >= A->lin && j >= B->col) {
		return;
	}

	cuDoubleComplex aux = make_cuDoubleComplex(0, 0);
	int size_k = A->index[i + 1];
	int size_j = B->index[j + 1];
	for (int k = A->index[i]; k < size_k; k++) {
		int l_A_col = A->vetor[k].col;
		for (int l = B->index[j]; l < size_j; l++) {
			if (l_A_col == B->vetor[l].lin) {
				aux = cuCadd(aux, cuCmul(A->vetor[k].value, B->vetor[l].value));
				break;
			}
		}
	}

	if (aux.x != 0.0 || aux.y != 0.0) {
		LOCK(lock);
		C[0](i, j, aux);
		UNLOCK(lock);
	}

}

template<typename type>
__host__ __device__ cuSparse<type>::cuSparse() {
	this->lin = 0;
	this->col = 0;
	this->vetor = 0;
	this->index = 0;
//	this->lock = 0;

	this->compress = CSR;
	this->orderned = false;
}

template<typename type>
__host__ __device__ cuSparse<type>::cuSparse(unsigned long lin,
		unsigned long col) {
#ifdef __CUDA_ARCH__
	this->lin = lin;
	this->col = col;
//	this->lock = 0;

	CHECK_ERROR(
			cudaMalloc((void**) &vetor,
					(lin * col / 2 + 1) * sizeof(element<type> )));

	CHECK_ERROR(
			cudaMalloc((void**) &index,
					(max(this->lin, this->col) + 1) * sizeof(unsigned long)));

	this->index[lin] = 0;

	this->compress = CSR;
	this->orderned = false;

#else

	this->lin = lin;
	this->col = col;
	this->vetor = (element<type> *) malloc((lin * col / 2 + 1) * sizeof(element<type>));
	this->index = (unsigned long *) malloc((max(this->lin, this->col) + 1) * sizeof(unsigned long));
	this->index[lin] = 0;

	this->compress = CSR;
	this->orderned = false;
#endif
}

template<typename type>
__host__ __device__ cuSparse<type>::~cuSparse() {
#ifdef __CUDA_ARCH__
	CHECK_ERROR(cudaFree(vetor));
	CHECK_ERROR(cudaFree(index));
#else
	free(vetor);
	free(index);
#endif
}

template<typename type>
__host__ __device__ cuSparse<type>::cuSparse(const cuSparse<type> &sparse) {
#ifdef __CUDA_ARCH__

	this->lin = sparse.lin;
	this->col = sparse.col;
//	this->lock = sparse.lock;
	CHECK_ERROR(
			cudaMalloc((void**) &vetor,
					(this->lin * this->col / 2 + 1) * sizeof(element<type> )));
	CHECK_ERROR(
			cudaMalloc((void**) &index,
					(max(this->lin, this->col) + 1) * sizeof(unsigned long)));

	this->compress = sparse.compress;
	this->orderned = sparse.orderned;
	int size = this->lin * this->col / 2 + 1;
	for (int i = 0; i < size; i++) {
		this->vetor[i] = sparse.vetor[i];
	}
	size = max(this->lin, this->col) + 1;
	for (int i = 0; i < size; i++) {
		this->index[i] = sparse.index[i];
	}

#else

	this->lin = sparse.lin;
	this->col = sparse.col;
	this->vetor = (element<type> *) malloc(
			(this->lin * this->col / 2 + 1) * sizeof(element<type> ));
	this->index = (unsigned long *) malloc(
			(max(this->lin, this->col) + 1) * sizeof(unsigned long));
	this->compress = sparse.compress;
	this->orderned = sparse.orderned;
	int size = this->lin * this->col / 2 + 1;
	for (int i = 0; i < size; i++) {
		this->vetor[i] = sparse.vetor[i];
	}
	size = max(this->lin, this->col) + 1;
	for (int i = 0; i < size; i++) {
		this->index[i] = sparse.index[i];
	}

#endif
}

template<typename type>
__host__    __device__ type cuSparse<type>::operator()(const unsigned long i,
		const unsigned long j) {
	if (!this->orderned) {
		this->order();
	}

	switch (this->compress) {
	case CSR:
		for (unsigned int k = this->index[i]; k < this->index[i + 1]; k++) {
			if (this->vetor[k].col > j) {
				return 0;
			}
			if (this->vetor[k].col == j) {
				return this->vetor[k].value;
			}
		}
		break;
	case CSC:
		for (unsigned int k = this->index[j]; k < this->index[j + 1]; k++) {
			if (this->vetor[k].lin > i) {
				return 0;
			}
			if (this->vetor[k].lin == i) {
				return this->vetor[k].value;
			}
		}
	}
	return 0;
}

template<>
__host__    __device__ cuDoubleComplex cuSparse<cuDoubleComplex>::operator()(
		const unsigned long i, const unsigned long j) {
	if (!this->orderned) {
		this->order();
	}

	switch (this->compress) {
	case CSR:
		for (unsigned int k = this->index[i]; k < this->index[i + 1]; k++) {
			if (this->vetor[k].col > j) {
				return make_cuDoubleComplex(0, 0);
			}
			if (this->vetor[k].col == j) {
				return this->vetor[k].value;
			}
		}
		break;
	case CSC:
		for (unsigned int k = this->index[j]; k < this->index[j + 1]; k++) {
			if (this->vetor[k].lin > i) {
				return make_cuDoubleComplex(0, 0);
			}
			if (this->vetor[k].lin == i) {
				return this->vetor[k].value;
			}
		}
	}
	return make_cuDoubleComplex(0, 0);
}

template<typename type>
__host__ __device__ void cuSparse<type>::operator()(const unsigned long i,
		const unsigned long j, const type v) {
#ifdef __CUDA_ARCH__
//	LOCK(lock);
	unsigned long nnz = (this->compress == CSC) ? this->col : this->lin;
	this->vetor[this->index[nnz]++] = element<type>(i, j, v);
	this->orderned = false;
//	UNLOCK(lock);
#else
	unsigned long nnz = (this->compress == CSC) ? this->col : this->lin;
	this->vetor[this->index[nnz]++] = element<type>(i, j, v);
	this->orderned = false;
#endif

}

template<typename type>
__host__    __device__ cuSparse<type>& cuSparse<type>::operator=(
		const cuSparse<type> &sparse) {
#ifdef __CUDA_ARCH__
	this->lin = sparse.lin;
	this->col = sparse.col;
	this->compress = sparse.compress;
	this->orderned = sparse.orderned;

	if (sparse.vetor != 0) {
		if (this->vetor != 0) {
			CHECK_ERROR(cudaFree(this->vetor));
		}
		int size = this->lin * this->col / 2 + 1;
		CHECK_ERROR(
				cudaMalloc((void**) &this->vetor,
						size * sizeof(element<type> )));

		for (int i = 0; i < size; i++) {
			this->vetor[i] = sparse.vetor[i];
		}
	} else {
		this->vetor = 0;
	}

	if (sparse.index != 0) {
		if (this->index != 0) {
			CHECK_ERROR(cudaFree(this->index));
		}
		int size = max(this->lin, this->col) + 1;
		CHECK_ERROR(
				cudaMalloc((void**) &this->index,
						size * sizeof(unsigned long)));
		for (int i = 0; i < size; i++) {
			this->index[i] = sparse.index[i];
		}
	} else {
		this->index = 0;
	}

#else
	this->lin = sparse.lin;
	this->col = sparse.col;
	if (this->vetor != 0) {
		free(this->vetor);
	}
	this->vetor = (element<type> *) malloc(
			(this->lin * this->col / 2 + 1) * sizeof(element<type> ));
	if (this->index != 0) {
		free(this->index);
	}
	this->index = (unsigned long *) malloc(
			(max(this->lin, this->col) + 1) * sizeof(unsigned long));
	this->compress = sparse.compress;
	this->orderned = sparse.orderned;
	int size = this->lin * this->col / 2 + 1;
	for (int i = 0; i < size; i++) {
		this->vetor[i] = sparse.vetor[i];
	}
	size = max(this->lin, this->col) + 1;
	for (int i = 0; i < size; i++) {
		this->index[i] = sparse.index[i];
	}
#endif
	return *this;
}

template<typename type>
__host__    __device__ cuSparse<type> cuSparse<type>::operator+(
		const cuSparse<type> &sparse) {
	cuSparse<type> B = sparse;
	if (this->compress != CSR) {
		this->compress = CSR;
		this->orderned = false;
		this->order();
	}
	if (B.compress != CSR) {
		B.compress = CSR;
		B.orderned = false;
		B.order();
	}

	cuSparse<type> C(this->lin, this->col);
	for (int i = 0; i < this->lin; i++) {
		int indexA = 0;
		int indexB = 0;
		int sizeA = this->index[i + 1] - this->index[i];
		int sizeB = B.index[i + 1] - B.index[i];
		while (indexA < sizeA || indexB < sizeB) {
			if (indexB == sizeB) {
				C(i, this->vetor[this->index[i] + indexA].col,
						this->vetor[this->index[i] + indexA].value);
				indexA++;
			} else if (indexA == sizeA
					|| this->vetor[this->index[i] + indexA].col
							> B.vetor[B.index[i] + indexB].col) {
				C(i, B.vetor[B.index[i] + indexB].col,
						B.vetor[B.index[i] + indexB].value);
				indexB++;
			} else if (this->vetor[this->index[i] + indexA].col
					< B.vetor[B.index[i] + indexB].col) {
				C(i, this->vetor[this->index[i] + indexA].col,
						this->vetor[this->index[i] + indexA].value);
				indexA++;
			} else {
				C(i, this->vetor[this->index[i] + indexA].col,
						this->vetor[this->index[i] + indexA].value
								+ B.vetor[B.index[i] + indexB].value);
				indexA++;
				indexB++;
			}
		}
	}
	return C;
}

template<>
__host__    __device__ cuSparse<cuDoubleComplex> cuSparse<cuDoubleComplex>::operator+(
		const cuSparse<cuDoubleComplex> &sparse) {
	cuSparse<cuDoubleComplex> B;
	B = sparse;

	if (this->compress != CSR) {
		this->compress = CSR;
		this->orderned = false;
		this->order();
	}
	if (B.compress != CSR) {
		B.compress = CSR;
		B.orderned = false;
		B.order();
	}

	this->order();
	B.order();

	cuSparse<cuDoubleComplex> C(this->lin, this->col);

#ifdef __CUDA_ARCH__

	cuSparse<cuDoubleComplex> *d_A;
	cuSparse<cuDoubleComplex> *d_B;
	cuSparse<cuDoubleComplex> *d_C;

	CHECK_ERROR(cudaMalloc((void**) &d_A, sizeof(cuSparse<cuDoubleComplex> )));
	CHECK_ERROR(cudaMalloc((void**) &d_B, sizeof(cuSparse<cuDoubleComplex> )));
	CHECK_ERROR(cudaMalloc((void**) &d_C, sizeof(cuSparse<cuDoubleComplex> )));

	*d_A = *this;
	*d_B = B;
	*d_C = C;

	int *lock;

	CHECK_ERROR(cudaMalloc((void**) &lock, sizeof(int)));

	*lock = 0;

	cuSadd<<<BLOCKS(this->lin, D_THREADS), D_THREADS>>>(d_A, d_B, d_C, lock);
	CHECK_ERROR(cudaDeviceSynchronize());
	C = *d_C;

	d_A->~cuSparse();
	d_B->~cuSparse();
	d_C->~cuSparse();
	CHECK_ERROR(cudaFree(lock));
	CHECK_ERROR(cudaFree(d_A));
	CHECK_ERROR(cudaFree(d_B));
	CHECK_ERROR(cudaFree(d_C));
#else

	for (int i = 0; i < this->lin; i++) {
		int indexA = 0;
		int indexB = 0;
		int sizeA = this->index[i + 1] - this->index[i];
		int sizeB = B.index[i + 1] - B.index[i];
		while (indexA < sizeA || indexB < sizeB) {
			if (indexB == sizeB) {
				C(i, this->vetor[this->index[i] + indexA].col,
						this->vetor[this->index[i] + indexA].value);
				indexA++;
			} else if (indexA == sizeA
					|| this->vetor[this->index[i] + indexA].col
					> B.vetor[B.index[i] + indexB].col) {
				C(i, B.vetor[B.index[i] + indexB].col,
						B.vetor[B.index[i] + indexB].value);
				indexB++;
			} else if (this->vetor[this->index[i] + indexA].col
					< B.vetor[B.index[i] + indexB].col) {
				C(i, this->vetor[this->index[i] + indexA].col,
						this->vetor[this->index[i] + indexA].value);
				indexA++;
			} else {
				C(i, this->vetor[this->index[i] + indexA].col,
						cuCadd(this->vetor[this->index[i] + indexA].value,
								B.vetor[B.index[i] + indexB].value));
				indexA++;
				indexB++;
			}
		}
	}

#endif
	return C;
}

template<typename type>
__host__    __device__ cuSparse<type> cuSparse<type>::operator*(
		cuSparse<type> &B) {
	if (this->compress != CSR) {
		this->compress = CSR;
		this->orderned = false;
		this->order();
	}
	if (B.compress != CSC) {
		B.compress = CSC;
		B.orderned = false;
		B.order();
	}

	cuSparse<type> C(this->lin, B.col);

	for (int i = 0; i < this->lin; i++) {
		for (int j = 0; j < B.col; j++) {
			double aux = 0.0;
			for (int k = this->index[i]; k < this->index[i + 1]; k++) {
				for (int l = B.index[j]; l < B.index[j + 1]; l++) {
					if (this->vetor[k].col == B.vetor[l].lin) {
						aux += this->vetor[k].value * B.vetor[l].value;
						break;
					}
				}
			}
			if (aux != 0.0) {
				C(i, j, aux);
			}
		}
	}
	return C;
}

template<>
__host__    __device__ cuSparse<cuDoubleComplex> cuSparse<cuDoubleComplex>::operator*(
		cuSparse<cuDoubleComplex> &B) {
	if (this->compress != CSR) {
		this->compress = CSR;
		this->orderned = false;
		this->index[lin] = this->index[col];
		this->order();
	}

	if (B.compress != CSC) {
		B.compress = CSC;
		B.orderned = false;

		B.index[B.col] = B.index[B.lin];

		B.order();
	}

	this->order();
	B.order();

	cuSparse<cuDoubleComplex> C(this->lin, B.col);

#ifdef __CUDA_ARCH__
	cuSparse<cuDoubleComplex> *d_A;
	cuSparse<cuDoubleComplex> *d_B;
	cuSparse<cuDoubleComplex> *d_C;

	CHECK_ERROR(cudaMalloc((void**) &d_A, sizeof(cuSparse<cuDoubleComplex> )));
	CHECK_ERROR(cudaMalloc((void**) &d_B, sizeof(cuSparse<cuDoubleComplex> )));
	CHECK_ERROR(cudaMalloc((void**) &d_C, sizeof(cuSparse<cuDoubleComplex> )));

	*d_A = *this;
	*d_B = B;
	*d_C = C;

	int *lock;

	CHECK_ERROR(cudaMalloc((void**) &lock, sizeof(int)));

	*lock = 0;

	cuSmul<<<BLOCKS(this->lin * B.col, D_THREADS), D_THREADS>>>(d_A, d_B, d_C, lock);
	CHECK_ERROR(cudaDeviceSynchronize());
	C = *d_C;

	d_A->~cuSparse();
	d_B->~cuSparse();
	d_C->~cuSparse();
	CHECK_ERROR(cudaFree(lock));
	CHECK_ERROR(cudaFree(d_A));
	CHECK_ERROR(cudaFree(d_B));
	CHECK_ERROR(cudaFree(d_C));

#else

	for (int i = 0; i < this->lin; i++) {
		for (int j = 0; j < B.col; j++) {
			cuDoubleComplex aux = make_cuDoubleComplex(0, 0);
			for (int k = this->index[i]; k < this->index[i + 1]; k++) {
				for (int l = B.index[j]; l < B.index[j + 1]; l++) {
					if (this->vetor[k].col == B.vetor[l].lin) {
						aux = cuCadd(aux,
								cuCmul(this->vetor[k].value, B.vetor[l].value));
						break;
					}
				}
			}
			if (aux.x != 0.0 || aux.y != 0.0) {
				C(i, j, aux);
			}
		}
	}

#endif
	return C;
}

template<typename type>
__host__ __device__ void cuSparse<type>::order() {

	if (!this->orderned) {

		element<type> max;
		max = element<type>();
#ifdef __CUDA_ARCH__
		switch (this->compress) {
		case CSR:
			bitonic_sort<element<type>, ComparatorCSR<type> >(this->vetor,
					this->index[this->lin], max);
			break;
		case CSC:
			bitonic_sort<element<type>, ComparatorCSC<type> >(this->vetor,
					this->index[this->col] - 1, max);
			break;
		}
#else
		switch (this->compress) {
			case CSR:
			quickSort<element<type>, ComparatorCSR<type> >(this->vetor, 0,
					this->index[this->lin] - 1);
			break;
			case CSC:
			quickSort<element<type>, ComparatorCSC<type> >(this->vetor, 0,
					this->index[this->col] - 1);
			break;
		}
#endif
		int x = 0;
		switch (this->compress) {
		case CSR:
			index[0] = 0;
			x = 0;
			for (int i = 0, z = 0; i < this->index[this->lin]; i++) {
				if (vetor[i].lin != z) {
					while (x != vetor[i].lin) {
						index[++x] = i;
					}
					z = vetor[i].lin;
				}
			}
			for (int i = ++x; i < this->lin; i++) {
				index[i] = this->index[this->lin];
			}
			break;
		case CSC:
			index[0] = 0;
			x = 0;
			for (int i = 0, z = 0; i < this->index[this->col]; i++) {
				if (vetor[i].col != z) {
					while (x != vetor[i].col) {
						index[++x] = i;
					}
					z = vetor[i].col;
				}
			}
			for (int i = ++x; i < this->col; i++) {
				index[i] = this->index[this->col];
			}
			break;
		}
		this->orderned = true;
	}
}

template<>
__host__ __device__ void cuSparse<cuDoubleComplex>::print() {
	if (!this->orderned) {
		order();
	}
	unsigned long nnz = (
			(this->compress == CSR) ?
					this->index[this->lin] : this->index[this->col]);
	double used = (nnz * 100.0) / (this->lin * this->col);
	printf(
			"\tCompressed %s Sparse (rows = %lu, cols = %lu, nnz = %lu [%.2lf\%%])\n",
			((this->compress == CSR) ? "Row" : "Column"), this->lin, this->col,
			nnz, used);
	int s = (this->compress == CSR) ? this->lin : this->col;
	printf("\tindex: [");
	for (int i = 0; i < s; i++) {
		printf(" %lu", this->index[i]);
	}
	printf("]\n");
	for (int i = 0; i < nnz; i++) {
		printf("\t(%lu, %lu)\t->\t%.4e%c%.4ei\n", this->vetor[i].lin,
				this->vetor[i].col, this->vetor[i].value.x,
				((this->vetor[i].value.y < 0.0) ? '-' : '+'),
				((this->vetor[i].value.y < 0.0) ?
						-this->vetor[i].value.y : this->vetor[i].value.y));
	}
}

template<typename type>
void cuSparse<type>::print() {
	unsigned long nnz = (
			(this->compress == CSR) ?
					this->index[this->lin] : this->index[this->col]);
	double used = (nnz * 100.0) / (this->lin * this->col);
	if (!this->orderned) {
		order();
	}
	printf(
			"Compressed %s Sparse (rows = %lu, cols = %lu, nnz = %lu [%.2lf\%%])\n",
			((this->compress == CSR) ? "Row" : "Column"), this->lin, this->col,
			nnz, used);
	int s = (this->compress == CSR) ? this->lin : this->col;
	printf("index: [");
	for (int i = 0; i < s; i++) {
		printf(" %lu", this->index[i]);
	}
	printf("]\n");
	for (int i = 0; i < nnz; i++) {
		printf("(%lu, %lu)\t->\t%.4e\n", this->vetor[i].lin, this->vetor[i].col,
				this->vetor[i].value);
	}
}

#endif //CUSPARSE_CUSPARSE_H
