
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
#define YMAX 2048
#define BLOCK_X 16
#define BLOCK_Y 16



void randomInit(float* data, int nb_elements) {

	int i;
	for (i = 0; i < nb_elements; i++) {
		//data[i] = rand();
		//data[i] = (float)((5 + (i % 17))/6.1);
		//data[i] = i;
		data[i] = ((i % 2008) + 1) / 1278.3;
	}
}


__global__ void additionKernel(float* a, size_t a_pitch, float* res, size_t res_pitch)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	res[x + y * res_pitch / sizeof(float)] = a[x + y * a_pitch / sizeof(float)] + 11.f;
}


extern "C" void function(
	float* host_res, float* host_a,
	int x_nb_elem, int y_nb_elem,
	int run, int device)
{
	// set the device to use
	cudaSetDevice(device);

	// timer
	cudaEvent_t start, stop;
	checkCudaErrors(cudaEventCreate(&start));
	checkCudaErrors(cudaEventCreate(&stop));

	// allocation of matrices 'a' and 'res' on device
	float* device_a;
	size_t device_a_pitch;

	cudaMallocPitch(&device_a, &device_a_pitch, x_nb_elem * sizeof(float), y_nb_elem);

	float* device_res;
	size_t device_res_pitch;

	cudaMallocPitch(&device_res, &device_res_pitch, x_nb_elem * sizeof(float), y_nb_elem);

	// data transfer from host_a to device_a (from host to device)
	size_t host_a_pitch = x_nb_elem * sizeof(float);
	cudaMemcpy2D(device_a, device_a_pitch, host_a, host_a_pitch, x_nb_elem * sizeof(float), y_nb_elem, cudaMemcpyHostToDevice);
	
	// CUDA grid setting
	dim3 block(BLOCK_X, BLOCK_Y);
	dim3 grid(XMAX / BLOCK_X, YMAX / BLOCK_Y);

	// Record the start event
	checkCudaErrors(cudaEventRecord(start));

	// kernel execution ('run' times)
	for (int i = 0; i < run; i++)
	{
		additionKernel<<<grid, block>>>(device_a, device_a_pitch, device_res, device_res_pitch);
	}

	// Record the stop event
	checkCudaErrors(cudaEventRecord(stop));

	// Wait for the stop event to complete
	checkCudaErrors(cudaEventSynchronize(stop));

	float msecTotal = 0.0f;
	checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));
	printf("kernel execution time : %f (ms)\n", msecTotal);

	// data transfer from device_res to host_res (from device to host)
	size_t host_res_pitch = x_nb_elem * sizeof(float);
	cudaMemcpy2D(host_res, host_res_pitch, device_res, device_res_pitch, x_nb_elem * sizeof(float), y_nb_elem, cudaMemcpyDeviceToHost);

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
  unsigned int x_nb_elem = (XMAX);
  unsigned int y_nb_elem = (YMAX);
  unsigned int size = x_nb_elem * y_nb_elem;
  unsigned int mem_size = size * sizeof(float);

  // allocation of matrices on the host
  float* mat_a = (float*)malloc(mem_size);
  float* mat_res = (float*)malloc(mem_size);

  // initialization of matrix 'a' on the host
  randomInit(mat_a, size);

  function(mat_res, mat_a, x_nb_elem, y_nb_elem, run, device);

  // Check result against expected
  int i, j;
  float temp;
  bool error = FALSE;
  for (j = 0; j < y_nb_elem; j++) {
	  for (i = 0; i < x_nb_elem; i++) {
		  temp = mat_res[i + j * x_nb_elem] - mat_a[i + j * x_nb_elem];
		  if (temp > 11.000001f || temp < 10.999999f) {
			  //printf("mat_res : %f | mat_a : %f\n", mat_res[i + j *x_nb_elem], mat_a[i + j *x_nb_elem]);
			  error = TRUE;
		  }
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
