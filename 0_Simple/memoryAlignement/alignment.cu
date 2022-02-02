
// System includes
#include <assert.h>
#include <stdio.h>

// CUDA runtime
#include <cuda_runtime.h>

// Helper functions and utilities to work with CUDA
#include <helper_cuda.h>
#include <helper_functions.h>


#define XMAX 4008
#define YMAX 4008
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


// a_pitch and b_pitch are in number of elements in the kernel
__global__ void kernel(
	float* a, size_t a_pitch,
	float* res, size_t res_pitch,
	int x_offset, int y_offset)
{
	int x = threadIdx.x + blockIdx.x * BLOCK_X + x_offset;
	int y = threadIdx.y + blockIdx.y * BLOCK_Y + y_offset;

	res[x + y * res_pitch] = a[x + y * a_pitch] + 11.f;
}


// device_a_padding and device_res_padding are in number of elements in the kernel
extern "C" void function(
	float* host_res, float* host_a,
	int x_elem, int y_elem,
	int x_offset, int y_offset,
	int device_a_padding, int device_res_padding,
	int run, int device)
{
	cudaSetDevice(device);

	// timer
	cudaEvent_t start, stop;
	checkCudaErrors(cudaEventCreate(&start));
	checkCudaErrors(cudaEventCreate(&stop));

	// allocation of matrices 'a' and 'res' on device
	float* device_a;
	size_t device_a_pitch;
	checkCudaErrors(cudaMallocPitch(
		(void**)& device_a, &device_a_pitch,
		(x_elem) * sizeof(float), y_elem));  // TODO?

	float* device_res;
	size_t device_res_pitch;
	checkCudaErrors(cudaMallocPitch(
		(void**)& device_res, &device_res_pitch,
		(x_elem) * sizeof(float), y_elem));  // TODO?


	// data transfer from host_a to device_a (from host to device)
	checkCudaErrors(cudaMemcpy2D(
		device_a, device_a_pitch,
		host_a, x_elem * sizeof(float),
		x_elem * sizeof(float), (y_elem),
		cudaMemcpyHostToDevice));  // TODO?

	// CUDA grid setting
	dim3 block(BLOCK_X, BLOCK_Y);
	dim3 grid(...);

	// Record the start event
	checkCudaErrors(cudaEventRecord(start));

	// kernel execution (launched 'run' times)
	for (int i = 0; i < run; i++)
	{
		kernel<<<grid, block>>>(
			device_a,
			device_a_pitch / sizeof(float),
			device_res,
			device_res_pitch / sizeof(float),
			x_offset, y_offset);  // TODO?
	}

	// Record the stop event
	checkCudaErrors(cudaEventRecord(stop));

	// Wait for the stop event to complete
	checkCudaErrors(cudaEventSynchronize(stop));

	float msecTotal = 0.0f;
	checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));
	printf("kernel execution time : %f (ms)\n", msecTotal);

	// data transfer from device_res to host_res (from device to host)
	checkCudaErrors(cudaMemcpy2D(
		host_res, x_elem * sizeof(float),
		device_res, device_res_pitch,
		x_elem * sizeof(float), (y_elem),
		cudaMemcpyDeviceToHost));  // TODO?

	// free memory
	cudaFree(device_a);
	cudaFree(device_res);
}


int main(int argc, char **argv)
{
  printf("[Memory alignment on CUDA devices] - Starting...\n");

  if (checkCmdLineFlag(argc, (const char **)argv, "help") ||
      checkCmdLineFlag(argc, (const char **)argv, "?")) {
    printf("Usage -device=n (n >= 0 for deviceID)\n");
    exit(EXIT_SUCCESS);
  }

  int device = findCudaDevice(argc, (const char **)argv);
  int run = 24;

  int device_a_padding = 0;
  int device_res_padding = 0;
  int s;
  for (s = 1; s < argc; s++) {
	if (argv[s][0] == '-') {
		switch (argv[s][1]) {
		case 'h': printf("Usage : %s [OPTIONS]\n", argv[0]);
			printf("\n");
			printf("  OPTIONS :\n");
			printf("    -i : padding for device_a   (number of elements)\n");
			printf("    -o : padding for device_res (number of elements)\n");
			printf("\n");
			return 0;
		case 'i': device_a_padding = atoi(argv[++s]);
			break;
		case 'o': device_res_padding = atoi(argv[++s]);
			break;
		default: printf("Unrecognized option : %c \n", argv[s][1]);
			return 0;
		}
	}
	else {
		printf("try -h for help\n");
		printf("\n");
		return 0;
	}
  }

  printf("device_a   padding: %d\n", device_a_padding);
  printf("device_res padding: %d\n", device_res_padding);

  // sizes
  unsigned int x_nb_elem = (XMAX);
  unsigned int y_nb_elem = (YMAX);
  unsigned int size = x_nb_elem * y_nb_elem;
  unsigned int mem_size = size * sizeof(float);

  // allocation of matrices on the host
  float* mat_a = (float*) malloc(mem_size);
  float* mat_res = (float*) malloc(mem_size);

  // initialization of matrix 'a' on the host
  randomInit(mat_a, size);

  int x_offset = 4; // must not be changed
  int y_offset = 4; // must not be changed

  function(
	  mat_res, mat_a,
	  x_nb_elem, y_nb_elem,
	  x_offset, y_offset,
	  device_a_padding, device_res_padding,
	  run, device);

  // Check result against expected
  int i, j;
  float temp;
  bool error = FALSE;
  for (j = y_offset; j < y_nb_elem - y_offset; j++) {
	  for (i = x_offset; i < x_nb_elem - x_offset; i++) {
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
