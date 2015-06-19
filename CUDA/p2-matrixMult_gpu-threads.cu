#include <stdio.h>
#include <assert.h>
#include <cuda_runtime.h>

int  XDIM;
int  YDIM;

/*
 * NXN Matrix Multiplication
 */
__global__ void
matMult(const int *A, const int *B, int *C, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements)
    {
    	for(int j = 0; j < numElements; j++){
    		int idx = (i*numElements)+j;
			for(int k = 0; k < numElements; k++){
				C[idx] += A[(i*numElements)+k] * B[(k*numElements)+j];
			}
    	}
        //printf("---→test: %d", i);
    }
}

void printMat(int * M, int XDIM){
    int i;
    int j;
    for(i = 0; i < XDIM; i++){
        for(j = 0; j < XDIM; j++){
            printf(" %d ", M[i*XDIM+j]);
        }
        printf("\n");
    }
}


int main(int argc, char **argv){
	XDIM    = 40;
    YDIM    = 40;
    if(argc > 1){
        XDIM = atoi(argv[1]);
        YDIM = XDIM;
    }
	//STEP 1 : Allocate in host
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;
    // Print the mat size to be used, and compute its size
    size_t size = sizeof(int*)*YDIM*XDIM;
    //printf("[Mat multiplication of %d elements]\n", YDIM);
    // Allocate the host input vector A
    int * h_A = (int *)malloc(size);
    // Allocate the host input vector B
    int * h_B = (int *)malloc(size);
    // Allocate the host output vector C
    int * h_C = (int *)malloc(size);
    // Initialize h_A and h_B with random numbers, h_C with 0's
    for(int i = 0; i < XDIM*XDIM; i++){
		h_A[i] = rand() & 0xF;
		h_B[i] = rand() & 0xF;
		h_C[i] = 0;
    }

    // Verify that allocations succeeded
    if (h_A == NULL || h_B == NULL || h_C == NULL)
    {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }
    //STEP 2: ALLOCATE IN CUDA
    // Allocate device memory
    int *d_A = NULL;
    int *d_B = NULL;
    int *d_C = NULL;

    cudaError_t error;
    error = cudaMalloc((void **) &d_A, size);

    if (error != cudaSuccess)
    {
        printf("cudaMalloc d_A returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }
    error = cudaMalloc((void **) &d_B, size);
    if (error != cudaSuccess)
    {
        printf("cudaMalloc d_B returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }
    error = cudaMalloc((void **) &d_C, size);
    if (error != cudaSuccess)
    {
        printf("cudaMalloc d_C returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }

    // copy host memory to device
    error = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    if (error != cudaSuccess)
    {
        printf("cudaMemcpy (d_A,h_A) returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }
    error = cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    if (error != cudaSuccess)
    {
        printf("cudaMemcpy (d_B,h_B) returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }
    error = cudaMemcpy(d_C, h_C, size, cudaMemcpyHostToDevice);
    if (error != cudaSuccess)
    {
        printf("cudaMemcpy (d_B,h_B) returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }

    // Launch the Mat mult CUDA Kernel
    int threadsPerBlock = 1024;
    int blocksPerGrid =(XDIM + threadsPerBlock - 1) / threadsPerBlock;
    //printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    matMult<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, XDIM);
    err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the device result vector in device memory to the host result vector
    // in host memory.
    //printf("Copy output data from the CUDA device to the host memory\n");
    err = cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Free device global memory
    err = cudaFree(d_A);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_B);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_C);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector C (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Free host memory
    //printMat(h_A,XDIM);
    //printMat(h_B,XDIM);
    //printMat(h_C,XDIM);

    free(h_A);
    free(h_B);
    free(h_C);

    // Reset the device and exit
    err = cudaDeviceReset();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    //printf("Done\n");
    return 0;
}
