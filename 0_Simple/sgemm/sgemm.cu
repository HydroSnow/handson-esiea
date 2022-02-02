#include <stdio.h>
#include "utils.h"

// CUDA runtime
#include <cuda_runtime.h>

// Helper functions and utilities to work with CUDA
#include <helper_cuda.h>
#include <helper_functions.h>



int main(int argc, char *argv[])
{
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);

  // matrices and other variables
  float *device_A, *device_B, *device_C;
  float *host_A, *host_B, *host_C, *host_C_reference;
  int M, N, K;
  float alpha = 1.0, beta = -1.0;

  size_t mat_size;
  int nstreams;
  cudaStream_t *streams;
  
  int runs = RUNS;
  
  //timing variables
  float elapsed_time;
  cudaEvent_t start_event, stop_event;
  checkCudaErrors(cudaEventCreate(&start_event));
  checkCudaErrors(cudaEventCreate(&stop_event));
  

  if(argc == 5)
  {
    M = atoi(argv[1]);
    N = atoi(argv[2]);
    K = atoi(argv[3]);
    nstreams = atoi(argv[4]);
  }
  else
  {
    M = 8192;
    N = 8192;
    K = 512;
    nstreams = 16;
  }
  mat_size = (M*K+K*N+M*N)*sizeof(float); //size in bytes of all the matrices

  if(mat_size > (deviceProp.totalGlobalMem - (size_t)(64 * 1024 * 1024)))
  {
    printf("matrices won't fit into device memory, exiting...\n"); 
    exit(-1);
  }
  
  /*
   * TODO : allocate A, B and C matrices on device using cudaMalloc()
   */
  device_A = 
  device_B = 
  device_C =

  /*
   * TODO : allocate A and B matrices, and allocate pinned memory for C matrix on the host
   */
  host_A = 
  host_B = 
  host_C = 

  /*
   * TODO : allocate pinned memory for the reference C matrix
   */
  host_C_reference = 

  // then initialize matrices
  initializeHostMatrices(host_A, host_B, host_C, host_C_reference, M, N, K);

  /*
   * TODO : allocate and initialize an array of stream handles
   */
  streams = 
  for(int i = 0; i < nstreams; i++)
    // TODO
   
  /*
   * TODO : modify the grid dimensions according to the streams configuration
   */
  dim3 dimGrid = dim3((M + 63) / 64, N / 16);
  dim3 dimBlock = dim3(16, 4);
  
  /*
   * TODO : write the transfers from host to device for matrices A and B
   */
  checkCudaErrors(cudaMemcpy2D(...));
  checkCudaErrors(cudaMemcpy2D(...));
 
  cudaEventRecord(start_event, 0);
  for(int r = 0; r < runs ; r++)
  {
    for(int i = 0; i < nstreams; i++)
    {
      /*
       * TODO : copy the slice of matrix C for stream i from host to device using
       *        the asynchronous API
       */
      checkCudaErrors(cudaMemcpyAsync());
    }
    
    for(int i = 0; i < nstreams; i++)
    {
      /*
       * TODO : write the kernel call for stream i according to nstreams
       */
      sgemmNN<<<..., ..., ..., ...>>>(...);
      
      cudaError_t err = cudaGetLastError();
      if(err != cudaSuccess)
      {
        fprintf(stderr, "CUDA Error (kernel call): (%d) %s\n",
                err, cudaGetErrorString(err));
        fflush(stderr);
      }   
    }

    for(int i = 0; i < nstreams; i++)
    {
      /*
       * TODO : copy the slice of matrix C for stream i from device to host 
       *        using the asynchronous API
       */
      checkCudaErrors(cudaMemcpyAsync(...));
    }
  }
  
  checkCudaErrors(cudaEventRecord(stop_event, 0));
  checkCudaErrors(cudaEventSynchronize(stop_event));
  checkCudaErrors(cudaEventElapsedTime(&elapsed_time, start_event, stop_event));
  printf("streamed SGEMM, mean time over %d run(s) = %f ms\n", runs, elapsed_time/runs);
    
  // reference version, non-streamed
  dimGrid = dim3(((M+63)/64), ((N/16)));
  dimBlock = dim3(16, 4);
  
  cudaEventRecord(start_event, 0);
  
  for(int r = 0; r < runs ; r++)
  {
    checkCudaErrors(cudaMemcpy2D(device_C, M * sizeof(float),
                                host_C_reference, M * sizeof(float),
                                M * sizeof(float),
                                N, cudaMemcpyHostToDevice));
    
    sgemmNN<<<dimGrid, dimBlock>>>(device_A, M,
                                   device_B, K,
                                   device_C, M,
                                   K, alpha, beta, M);
    
    checkCudaErrors(cudaMemcpy2D(host_C_reference, M * sizeof(float),
                                device_C, M * sizeof(float),
                                M * sizeof(float),
                                N, cudaMemcpyDeviceToHost));
  }
  checkCudaErrors(cudaEventRecord(stop_event, 0));
  checkCudaErrors(cudaEventSynchronize(stop_event));
  checkCudaErrors(cudaEventElapsedTime(&elapsed_time, start_event, stop_event));
  printf("non-streamed SGEMM, mean time over %d run(s) = %f ms\n", runs, elapsed_time/runs);

  // check data
  float rela = 0.0f;
  float difference = diff(M, N, host_C_reference, host_C, &rela);
  fprintf(stderr, "\ndiff(streamed SGEMM VS. non-streamed SGEMM) : absolute error sum = %e (relative error = %e)\n", difference, rela);
  
  /*
   * TODO : deallocate the array of stream handles
   */
  

  /*
   * TODO : deallocate both pinned and standard device memory, and host memory
   */
  
  return 0;
}
