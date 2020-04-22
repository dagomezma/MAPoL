#ifndef DEFINES_H_
#define DEFINES_H_

/**
 * Usar somente com construtores de funções
 * Define uma função tanto para host como para device
 */
#define def __host__ __device__
#define ID() (threadIdx.x + blockIdx.x * blockDim.x)
#define ABS(a) (((a) < 0.0) ? -(a) : (a))
#define BLOCKS(a,b) ((a) % (b) ? (a) / (b) + 1 : (a) / (b) )
#define LOCK(l) while(atomicCAS(l, 0, ID() + 1))
#define UNLOCK(l) atomicExch(l, 0)
#define CHECK_ERROR( err ) (__CHECK_ERROR( err, __FILE__, __LINE__ ))


__host__ __device__ static void __CHECK_ERROR( cudaError_t err,
                         const char *file,
                         int line ) {
#ifdef __CUDA_ARCH__
    if (err != cudaSuccess) {
    	printf("%s in %s at line %d\n", err, file, line);
    }
#else
    if (err != cudaSuccess) {
           printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                   file, line );
           exit( EXIT_FAILURE );
       }
#endif
}

#endif /* DEFINES_H_ */
