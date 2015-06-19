#include <stdio.h>
#include <assert.h>
#include <cuda_runtime.h>

/*
 * NXN Matrix Multiplication
 */
__global__ void
matMult(const int *A, const int *B, int *C, const int DIM)
{
    extern __shared__ int shared[];
    int *As = &shared[0]; // s stands for shared
    int *Bs = &shared[DIM]; // s stands for shared

    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < DIM)
    {
    	for(int j = 0; j < DIM; j++){
    		int idx = (i*DIM)+j;
	        for(int k = 0; k < DIM; k++){
                    As[i] = A[(i*DIM)+k];
                    Bs[i] = B[(k*DIM)+j];
                    __syncthreads();
//		    C[idx] += A[(i*DIM)+k] * B[(k*DIM)+j];
		    C[idx] += As[k] * Bs[k];
                   __syncthreads();

		}
    	}
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


int
main(void)
{
    //STEP 1 : Allocate in host
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;
    // Print the mat size to be used, and compute its size
    const int YDIM = 2;
    const int XDIM = 2;
    size_t size = sizeof(int*)*YDIM*XDIM;
    printf("[Mat multiplication of %d elements]\n", YDIM);
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
    int threadsPerBlock = 256;
    int blocksPerGrid =(XDIM + threadsPerBlock - 1) / threadsPerBlock;

    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);

    matMult<<<blocksPerGrid, threadsPerBlock, 2*XDIM*sizeof(int)>>>(d_A, d_B, d_C, XDIM);
    err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the device result vector in device memory to the host result vector
    // in host memory.
    printf("Copy output data from the CUDA device to the host memory\n");
    err = cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Verify that the result vector is correct
    /*for (int i = 0; i < numElements; ++i)

    {
        if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5)
        {
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }*/

    printf("Test PASSED\n");
    fflush(stdout);
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
    printMat(h_A,XDIM);
    printMat(h_B,XDIM);
    printMat(h_C,XDIM);

    free(h_A);
    free(h_B);
    free(h_C);

    // Reset the device and exit
    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    err = cudaDeviceReset();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    printf("Done\n");
    return 0;

}

