
// System includes
#include <assert.h>
#include <stdio.h>

// CUDA runtime
#include <cuda_runtime.h>

// Helper functions and utilities to work with CUDA
#include <helper_cuda.h>
#include <helper_functions.h>


#define XMAX 2008
#define YMAX 2008
#define BLOCK_X 16  // 64
#define BLOCK_Y 16  // 8


void randomInit(float* data, int nb_elements)
{
	int i;
	for (i = 0; i < nb_elements; i++)
		data[i] = ((i % 2008) + 1) / 1278.3;
}


__constant__ float constantes [5];


__global__ void kernel_convolution(
	float* a, size_t a_pitch,
	float* b, size_t b_pitch,
	int x_offset, int y_offset,
	int device_a_padding, int device_res_padding)
{ 
  int index_x_entree = threadIdx.x + blockIdx.x * BLOCK_X + x_offset;  // TODO padding?
  int index_x_sortie = threadIdx.x + blockIdx.x * BLOCK_X + x_offset;  // TODO padding?

  int index_y = threadIdx.y + blockIdx.y * BLOCK_Y + y_offset;

  __shared__ float shared_mem[...][...];

  int x = threadIdx.x + 4;
  int y = threadIdx.y + 4;

  shared_mem[...][...] = a[...];

  if((x-4) < 4)
    shared_mem[...][...] = a[...];

  if((x-4) > (blockDim.x-5))
    shared_mem[...][...] = a[...];

  if((y-4) < 4)
    shared_mem[...][...] = a[...];

  if((y-4) > (blockDim.y-5))
    shared_mem[...][...] = a[...];

  __syncthreads();

  float coeff =0.0f;

  coeff = constantes[0] * shared_mem[...][...];
  for(int i=1; i<=4; i++)
    coeff += constantes[i] * (shared_mem[...][...] + shared_mem[...][...]);

  coeff += constantes[0] * shared_mem[...][...];
  for(int i=1; i<=4; i++)
    coeff += constantes[i] * (shared_mem[...][...] + shared_mem[...][...]);

  b[...] = shared_mem[...][...] * (1.0f + coeff);
}


extern "C" void convolution(
	float* host_res, float *host_a, 
	int x_nb_elem, int y_nb_elem, 
	int x_offset, int y_offset,
	int device_a_padding, int device_res_padding,
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

	//TO DO : 2D allocation of matrix a
	checkCudaErrors(cudaMallocPitch(
		(void**)& device_a,
		&device_a_pitch,
		x_nb_elem * sizeof(float),
		y_nb_elem));

	float* device_res;
	size_t device_res_pitch;

	//TO DO : 2D allocation of matrix res
	checkCudaErrors(cudaMallocPitch(
		(void**)& device_res,
		&device_res_pitch,
		x_nb_elem * sizeof(float),
		y_nb_elem));

	// data transfer from host_a to device_a (from host to device)
	// TO DO : cudaMemcpy2D();
	checkCudaErrors(cudaMemcpy2D(
		device_a, device_a_pitch,
		host_a, x_nb_elem * sizeof(float),
		x_nb_elem * sizeof(float), y_nb_elem,
		cudaMemcpyHostToDevice));

	float constantes_cpu[5];
	constantes_cpu[0]=  5.321f;
	constantes_cpu[1]= -4.683f;
	constantes_cpu[2]=  3.857f;
	constantes_cpu[3]= -2.764f;
	constantes_cpu[4]=  1.953f;
	checkCudaErrors(cudaMemcpyToSymbol(constantes, constantes_cpu, sizeof(constantes_cpu), 0));

	// CUDA grid setting
	dim3 block(BLOCK_X, BLOCK_Y);
	dim3 grid(
		((x_nb_elem - 2 * x_offset) + block.x - 1) / block.x,
		((y_nb_elem - 2 * y_offset) + block.y - 1) / block.y);

	// Record the start event
	checkCudaErrors(cudaEventRecord(start));

	// kernel execution ('run' times)
	for (int i = 0; i < run; i++)
	{
		kernel_convolution <<< grid, block >>>(
			device_a, device_a_pitch / sizeof(float),
			device_res, device_res_pitch / sizeof(float),
			x_offset, y_offset,
			device_a_padding, device_res_padding);
	}

	// Record the stop event
	checkCudaErrors(cudaEventRecord(stop));

	// Wait for the stop event to complete
	checkCudaErrors(cudaEventSynchronize(stop));

	float msecTotal = 0.0f;
	checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));
	printf("kernel execution time : %f (ms)\n", msecTotal);

	// data transfer from device_res to host_res (from device to host)
	// TO DO : cudaMemcpy
	checkCudaErrors(cudaMemcpy2D(
		host_res, x_nb_elem * sizeof(float),
		device_res, device_res_pitch,
		x_nb_elem * sizeof(float), y_nb_elem,
		cudaMemcpyDeviceToHost));

	// free memory
	cudaFree(device_a);
	cudaFree(device_res);
}



int main(int argc, char **argv)
{
  printf("[Convolution on CUDA devices] - Starting...\n");

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
  unsigned int x_elem = (XMAX);
  unsigned int y_elem = (YMAX);
  unsigned int size = x_elem * y_elem;
  unsigned int mem_size = size * sizeof(float);

  // allocation of matrices on the host
  float * host_a = (float*) malloc(mem_size);
  float * host_res = (float*) malloc(mem_size);
  float * expected_res = (float*) malloc(mem_size);

  // initialization of matrix 'a' on the host
  randomInit(host_a, size);

  int x_offset = 4; // must not be changed
  int y_offset = 4; // must not be changed

  convolution(host_res, host_a, x_elem, y_elem, x_offset, y_offset, device_a_padding, device_res_padding, run, device);
  
  FLOAT coeff;
  FLOAT constantes [5];
  constantes[0] = 5.321f;
  constantes[1] = -4.683f;
  constantes[2] = 3.857f;
  constantes[3] = -2.764f;
  constantes[4] = 1.953f;
  int i,j;
  for(j = y_offset ; j < (y_elem - y_offset) ; j++) {
    for(i = x_offset ; i < (x_elem - x_offset) ; i++) {
      int k;
      coeff = constantes[0] * host_a[i + j * x_elem];
      for(k=1; k<=4; k++)
		coeff += constantes[k] * (host_a[(i+k) + j * x_elem] + host_a[(i-k) + j * x_elem]);
      coeff += constantes[0] * host_a[i + j * x_elem];
      for(k=1; k<=4; k++)
		coeff += constantes[k] * (host_a[i + (j+k) * x_elem] + host_a[i + (j-k) * x_elem]);
      expected_res[i + j * x_elem]= host_a[i + j * x_elem]* (1.0f+coeff);
    }
  }

  FLOAT temp;
  int error = 0;
  for(j = y_offset ; j < (y_elem - y_offset) ; j++){
    for(i = x_offset ; i < (x_elem - x_offset) ; i++){
      temp = host_res[i + j *x_elem] - expected_res[i + j *x_elem]; // difference absolue
      if(temp > 0.000001f || temp < -0.000001f){
		//difference relative
		temp = temp / expected_res[i + j *x_elem];
		if(temp > 0.0001f || temp < -0.0001f){
			if (error < 20) fprintf(stderr, "%d,%d: %f <-> %f\n", i, j, host_res[i + j *x_elem] , expected_res[i + j *x_elem]);
			error++;
		}
      }
    }
  }

  if (error)
	  printf("Computation failed\n");
  else
	  printf("Passed\n");

  free(host_a);
  free(host_res);

  exit(!error ? EXIT_SUCCESS : EXIT_FAILURE);
}
