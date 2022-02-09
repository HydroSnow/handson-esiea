
// System includes
#include <assert.h>
#include <stdio.h>

// CUDA runtime
#include <cuda_runtime.h>

// Helper functions and utilities to work with CUDA
#include <helper_cuda.h>
#include <helper_functions.h>


#include "device_launch_parameters.h"


#define XMAX 2048
#define BLOCK_X 256


void randomInit(float* data, int nb_elements) {

	int i;
	for (i = 0; i < nb_elements; i++) {
		//data[i] = rand();
		//data[i] = (float)((5 + (i % 17))/6.1);
		//data[i] = i;
		data[i] = ((i % 2008) + 1) / 1278.3;
	}
}


__global__ void additionKernel(float* a, float* res)
{
	int index = blockIdx.x * BLOCK_X + threadIdx.x;
	res[index] = a[index] + 11.f;
}


extern "C" void function(float* host_res, float* host_a, int nb_elements, int run, int device)
{
	// set the device to use
	cudaSetDevice(device);

	// timer
	cudaEvent_t start, stop;
	checkCudaErrors(cudaEventCreate(&start));
	checkCudaErrors(cudaEventCreate(&stop));

	// allocation of matrices 'a' and 'res' on device
	float* device_a;
	size_t size = nb_elements * sizeof(float);
	cudaMalloc(&device_a, size);

	float* device_res;
	cudaMalloc(&device_res, size);

	// data transfer from host_a to device_a (from host to device)
	cudaMemcpy(device_a, host_a, size, cudaMemcpyHostToDevice);

	dim3 block(BLOCK_X); // number of threads along the x dimension of each block
	dim3 grid(XMAX / BLOCK_X);

	// Record the start event
	checkCudaErrors(cudaEventRecord(start));

	// kernel execution ('run' times)
	for (int i = 0; i < run; i++)
	{
		additionKernel<<<grid, block>>>(device_a, device_res);
	}

	// Record the stop event
	checkCudaErrors(cudaEventRecord(stop));

	// Wait for the stop event to complete
	checkCudaErrors(cudaEventSynchronize(stop));

	float msecTotal = 0.0f;
	checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));
	printf("kernel execution time : %f (ms)\n", msecTotal);

	// data transfer from device_res to host_res (from device to host)
	cudaMemcpy(host_res, device_res, size, cudaMemcpyDeviceToHost);

	// free memory
	cudaFree(device_a);
	cudaFree(device_res);
}


int main(int argc, char **argv)
{
  printf("[Simple Addtion with CUDA] - Starting...\n");

  if (checkCmdLineFlag(argc, (const char **)argv, "help") ||
      checkCmdLineFlag(argc, (const char **)argv, "?")) {
    printf("Usage -device=n (n >= 0 for deviceID)\n");
    exit(EXIT_SUCCESS);
  }

  int device = findCudaDevice(argc, (const char **)argv);
  int run = 24;

  // sizes
  int nb_elements = (XMAX);
  size_t mem_size = nb_elements * sizeof(float);

  // allocation of matrices on the host
  float* mat_a = (float*)malloc(mem_size);
  float* mat_res = (float*)malloc(mem_size);

  // initialization of matrix 'a' on the host
  randomInit(mat_a, nb_elements);

  function(mat_res, mat_a, nb_elements, run, device);

  // Check result against expected
  int i;
  float temp;
  bool error = FALSE;
  for (i = 0; i < nb_elements; i++) {
	  temp = mat_res[i] - mat_a[i];
	  if (temp > 11.000001f || temp < 10.999999f) {
		  error = TRUE;
	  }
  }

  if (error)
	  printf("Computation failed\n");
  else
	  printf("Passed\n");

  free(mat_a);
  free(mat_res);

  exit(!error ? EXIT_SUCCESS : EXIT_FAILURE);
}
