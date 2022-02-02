/*
 * Copyright 1993-2007 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and
 * international Copyright laws.  Users and possessors of this source code
 * are hereby granted a nonexclusive, royalty-free license to use this code
 * in individual and commercial software.
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
 * OF USE, DATA OR PROFITS,  WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
 * OR OTHER TORTIOUS ACTION,  ARISING OUT OF OR IN CONNECTION WITH THE USE
 * OR PERFORMANCE OF THIS SOURCE CODE.
 *
 * U.S. Government End Users.   This source code is a "commercial item" as
 * that term is defined at  48 C.F.R. 2.101 (OCT 1995), consisting  of
 * "commercial computer  software"  and "commercial computer software
 * documentation" as such terms are  used in 48 C.F.R. 12.212 (SEPT 1995)
 * and is provided to the U.S. Government only as a commercial end item.
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
 * source code with only those rights set forth herein.
 *
 * Any use of this source code in individual and commercial software must
 * include, in the user documentation and internal comments to the code,
 * the above Disclaimer and U.S. Government End Users Notice.
 *
 *
 * This sample illustrates the usage of CUDA streams for overlapping
 * kernel execution with device/host memcopies.  The kernel is used to 
 * initialize an array to a specific value, after which the array is 
 * copied to the host (CPU) memory.  To increase performance, multiple 
 * kernel/memcopy pairs are launched asynchronously, each pair in its 
 * own stream.  Devices with Compute Capability 1.1 can overlap a kernel
 * and a memcopy as long as they are issued in different streams.  Kernels
 * are serialized.  Thus, if n pairs are launched, streamed approach
 * can reduce the memcopy cost to the (1/n)th of a single copy of the entire
 * data set.
 *
 * Additionally, this sample uses CUDA events to measure elapsed time for
 * CUDA calls.  Events are a part of CUDA API and provide a system independent
 * way to measure execution times on CUDA devices with approximately 0.5 
 * microsecond precision.
 *
 * Elapsed times are averaged over nreps repetitions (10 by default).
 *
*/

// CUDA runtime
#include <cuda_runtime.h>

// Helper functions and utilities to work with CUDA
#include <helper_cuda.h>
#include <helper_functions.h>

#include <stdio.h>
#include <stdlib.h>


__global__ void init_array(int *g_data, int *factor, int num_iterations)
{ 
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  for(int i=0;i<num_iterations;i++)
    g_data[idx] += *factor;	// non-coalesced on purpose, to burn time
}

int correct_data(int *a, const int n, const int c)
{
  for(int i = 0; i < n; i++)
    if(a[i] != c)
    {
			printf("%d: %d %d\n", i, a[i], c);
      return 0;
		}
  return 1;
}

int main(int argc, char *argv[])
{
  int cuda_device = 0;
  int nstreams = 4;               // number of streams for CUDA calls
  int nreps = 10;                 // number of times each experiment is repeated
  int n = 16 * 1024 * 1024;       // number of ints in the data set
  int nbytes = n * sizeof(int);   // number of data bytes
  dim3 threads, blocks;           // kernel launch configuration
  float elapsed_time;             // timing variables
  int niterations;	              // number of iterations for the loop inside the kernel
  
  if( argc > 1 )
    cuda_device = atoi( argv[1] );
  
#ifdef __DEVICE_EMULATION__
  n = 4096;   // reduced workload for emulation (n should be divisible by 512*nstreams)
#endif
  
  // check the compute capability of the device
  int num_devices=0;
  checkCudaErrors(cudaGetDeviceCount(&num_devices));
  if(0==num_devices)
  {
    printf("your system does not have a CUDA capable device\n");
    return 1;
  }
	
  // check if the command-line chosen device ID is within range, exit if not
  if( cuda_device >= num_devices )
  {
    printf("choose device ID between 0 and %d\n", num_devices-1);
    return 1;
  }
  
  cudaSetDevice( cuda_device );
  
  cudaDeviceProp device_properties;
	
  checkCudaErrors( cudaGetDeviceProperties(&device_properties, cuda_device) );
  if( (1 == device_properties.major) && (device_properties.minor < 1))
    printf("%s does not have compute capability 1.1 or later\n\n", device_properties.name);
  if(device_properties.minor > 1)
    niterations = 5;
  else
    niterations = 2; // reduced workload for compute capability 1.0 and 1.1
  
  printf("running on: %s with compute capability %d.%d\n", device_properties.name, device_properties.major, device_properties.minor);
  
  // allocate host memory
  int c = 5; // value to which the array will be initialized  
  int * a = 0; // pointer to the data array in memory
  
  /*
   * TODO : change allocation of array a to use pinned memory
   */
  a = (int *) malloc(nbytes);

  // allocate device memory
  int *d_a = 0, *d_c = 0;             // pointers to data and init value in the device memory
  checkCudaErrors( cudaMalloc(&d_a, nbytes) );
  checkCudaErrors( cudaMalloc(&d_c, sizeof(int)) );
  checkCudaErrors( cudaMemcpy(d_c, &c, sizeof(int), cudaMemcpyHostToDevice) );
  
  /*
   * TODO :  allocate and initialize an array of stream handles
   */
  cudaStream_t *streams = ...;
  for(int i = 0; i < nstreams; i++)
    // TODO
  
  // create CUDA event handles
  cudaEvent_t start_event, stop_event;
  checkCudaErrors( cudaEventCreate(&start_event) );
  checkCudaErrors( cudaEventCreate(&stop_event) );

  //////////////////////////////////////////////////////////////////////
  // time non-streamed execution for reference
  //////////////////////////////////////////////////////////////////////
  threads=dim3(512, 1);
  blocks=dim3(n / threads.x, 1);
  cudaEventRecord(start_event, 0);
  for(int k = 0; k < nreps; k++)
  {
    init_array<<<blocks, threads>>>(d_a, d_c, niterations);
    cudaMemcpy(a, d_a, nbytes, cudaMemcpyDeviceToHost);
  }
  cudaEventRecord(stop_event, 0);
  cudaEventSynchronize(stop_event);
  checkCudaErrors( cudaEventElapsedTime(&elapsed_time, start_event, stop_event) );
  printf("non-streamed:\t%.2f ms\n", elapsed_time / nreps);
  
  //////////////////////////////////////////////////////////////////////
  // time execution with nstreams streams
  //////////////////////////////////////////////////////////////////////
  threads=dim3(512,1);

  /*
   * TODO : modify the grid dimensions according to the streams configuration
   */
  blocks = dim3(n / threads.x, 1);
   
  memset(a, 255, nbytes);     // set host memory bits to all 1s, for testing correctness
  cudaMemset(d_a, 0, nbytes); // set device memory to all 0s, for testing correctness
  cudaEventRecord(start_event, 0);
  for(int k = 0; k < nreps; k++)
  {
    /*
     * TODO : launch execution of nstreams streams
     */
   
    
    /*
     * TODO : use the asynchronous copy function to retrieve data from the device into nstreams streams
     */
    
  }
  cudaEventRecord(stop_event, 0);
  cudaEventSynchronize(stop_event);
  checkCudaErrors( cudaEventElapsedTime(&elapsed_time, start_event, stop_event) );
  printf("%d streams:\t%.2f ms\n", nstreams, elapsed_time / nreps);

  // check whether the output is correct
  printf("-------------------------------\n");
  if(correct_data(a, n, c*nreps*niterations))	// each element should have been incremented by c a total of npres*niterations times
    printf("Test PASSED\n");
  else
    printf("Test FAILED\n");
  
  // release resources
  /*
   * TODO : destroy the nstreams streams
   */ 
  for(int i = 0; i < nstreams; i++)
    // TODO

  cudaEventDestroy(start_event);
  cudaEventDestroy(stop_event);
  
  /*
   * TODO : free the pinned memory allocated for array a
   */


  cudaFree(d_a);
  cudaFree(d_c);

  return 0;
}
