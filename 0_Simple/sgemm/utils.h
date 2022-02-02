#include <stdio.h>
#include <driver_types.h>

typedef cudaDeviceProp cudaDeviceProp_t;
#define RUNS 10

// error checking macro
#define CUDA_SAFE_CALL(call, location)                                                                              \
 do                                                                                                                 \
 {                                                                                                                  \
  cudaError_t err = call;                                                                                           \
  if(cudaSuccess != err)                                                                                            \
  {                                                                                                                 \
    fprintf(stderr, "%s: CUDA Error (%s): (%d) %s\n", __PRETTY_FUNCTION__, location, err, cudaGetErrorString(err)); \
    fflush(stderr);                                                                                                 \
  }                                                                                                                 \
}while (0)

//
// kernel function
//
__device__ void saxpy(float a, float *b, float *c)
{
	c[0] += a*b[0];
	c[1] += a*b[1];
	c[2] += a*b[2];
	c[3] += a*b[3];
	c[4] += a*b[4];
	c[5] += a*b[5];
	c[6] += a*b[6];
	c[7] += a*b[7];
	c[8] += a*b[8];
	c[9] += a*b[9];
	c[10] += a*b[10];
	c[11] += a*b[11];
	c[12] += a*b[12];
	c[13] += a*b[13];
	c[14] += a*b[14];
	c[15] += a*b[15];
}

__global__ void sgemmNN(const float *A, int lda,
                        const float *B, int ldb,
                        float* C, int ldc,
                        int k, float alpha, float beta, int M)
{
  const int inx = threadIdx.x;
  const int iny = threadIdx.y;
  const int ibx = blockIdx.x * 64;
  const int iby = blockIdx.y * 16;
  const int id = inx + iny*16;
	
  A += ibx + id;
  B += inx + __mul24(iby + iny, ldb);
  C += ibx + id  + __mul24(iby, ldc);
	
  const float *Blast = B + k;
	
  float c[16] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
    
  do
  {
    float a[4];
    __shared__ float bs[16][17];

    if ((ibx + id) < M) {
      a[0] = A[0*lda];
      a[1] = A[1*lda];
      a[2] = A[2*lda];
      a[3] = A[3*lda];
    }

    bs[inx][iny]    = B[0*ldb];
    bs[inx][iny+4]  = B[4*ldb];
    bs[inx][iny+8]  = B[8*ldb];
    bs[inx][iny+12] = B[12*ldb];
    
    __syncthreads();

    if ((ibx + id) < M) {
      A += 4*lda;
      saxpy(a[0], &bs[0][0], c);		a[0] = A[0*lda];
      saxpy(a[1], &bs[1][0], c);		a[1] = A[1*lda];
      saxpy(a[2], &bs[2][0], c);		a[2] = A[2*lda];
      saxpy(a[3], &bs[3][0], c);		a[3] = A[3*lda];	

      A += 4*lda;
      saxpy(a[0], &bs[4][0], c);		a[0] = A[0*lda];
      saxpy(a[1], &bs[5][0], c);		a[1] = A[1*lda];
      saxpy(a[2], &bs[6][0], c);		a[2] = A[2*lda];
      saxpy(a[3], &bs[7][0], c);		a[3] = A[3*lda];
		
      A += 4*lda;
      saxpy(a[0], &bs[8][0], c);		a[0] = A[0*lda];
      saxpy(a[1], &bs[9][0], c);		a[1] = A[1*lda];
      saxpy(a[2], &bs[10][0], c);		a[2] = A[2*lda];
      saxpy(a[3], &bs[11][0], c);		a[3] = A[3*lda];
		
      A += 4*lda;
      saxpy(a[0], &bs[12][0], c);
      saxpy(a[1], &bs[13][0], c);
      saxpy(a[2], &bs[14][0], c);
      saxpy(a[3], &bs[15][0], c);
    }
		
    B += 16;
    __syncthreads();
  } while(B < Blast);
	
  if ((ibx + id) < M) 
  {
    for(int i = 0; i < 16; i++, C += ldc)
      C[0] = alpha*c[i] + beta*C[0]; 
  }
}	

void checkDeviceProperties(cudaDeviceProp_t * deviceProp)
{
  int num_devices, cuda_device = 0;
  
#ifdef __DEVICE_EMULATION__
  n = 4096;   // reduced workload for emulation (n should be divisible by 512*nstreams)
#endif
  
  // check the compute capability of the device
  CUDA_SAFE_CALL(cudaGetDeviceCount(&num_devices), "cudaGetDeviceCount()");
  if(num_devices == 0)
  {
    printf("your system does not have a CUDA capable device\n");
    exit(-1);
  }
	
  // check if the command-line chosen device ID is within range, exit if not
  if(cuda_device >= num_devices)
  {
    printf("choose device ID between 0 and %d\n", num_devices-1);
    exit(-1);
  }
  
  cudaSetDevice(cuda_device);
  CUDA_SAFE_CALL(cudaGetDeviceProperties(deviceProp, cuda_device), "cudaGetDeviceProperties()");

  if((1 == (*deviceProp).major) && ((*deviceProp).minor < 1))
    printf("%s does not have compute capability 1.1 or later\n\n", (*deviceProp).name);
  
  printf("running on: %s with compute capability %d.%d\n\n", (*deviceProp).name, (*deviceProp).major, (*deviceProp).minor);
}

void initializeHostMatrices(float * host_A, float * host_B, float * host_C, float * host_C_reference,
                            int M, int N, int K)
{
  int seed = 12345;
  int i, j;

  for (i = 0 ; i < M ; i++)
  {
    for (j = 0 ; j < K ; j ++) 
    {
      if (j > 0)
        host_A[i*K+j] = ((float)(i%j)+seed)/((float)K);
      else
        host_A[i*K+j] = ((float)i+seed)/((float)M);
    }
  }
  for (i = 0 ; i < N ; i++) 
  {
    for (j = 0 ; j < K ; j ++) 
    {
      if (i > 0)
        host_B[i*K+j] = ((float)(j%i)-seed)/((float)N);
      else
        host_B[i*K+j] = ((float)j-seed)/((float)K);
    }
  }
  for (i = 0 ; i < M ; i++) 
  {
    for (j = 0 ; j < N ; j ++) 
    {
      host_C[i*N+j] = ((float)(j-i)*(seed+1))/((float)(M*N));
      host_C_reference[i*N+j] =  host_C[i*N+j];
    }
  }
}

static inline float diff(int M, int N, float * host_C_reference, float * host_C, float * rel)
{
  int counterror = 0;
  float err = 0.0f;
  int j, i;
  *rel = 0.0f;

  for(j = 0; j < N; j++)
  {
    for(i = 0; i < M; i++)
    {
      float lrel = 0.f;
      err = fmaxf(err, fabsf(host_C_reference[i+j*M] - host_C[i+j*M]));
      if (fmaxf(fabs(host_C_reference[i+j*M]), fabs(host_C[i+j*M])) > 0.)
        lrel = fabsf(host_C_reference[i+j*M] - host_C[i+j*M]) / fmaxf(fabs(host_C_reference[i+j*M]), fabs(host_C[i+j*M]));
      *rel = fmaxf(*rel, lrel);
        
      if ((lrel > 0.001) || isnan(host_C_reference[i+j*M]) || isnan(host_C[i+j*M]))
      {
        fprintf(stderr, "host_C_reference[%d,%d = %d] = %.15f/ host_C[%d,%d = %d] = %.15f (%.10f)\n", i,j,i+j*M, host_C_reference[i+j*M], i,j,i+j*M, host_C[i+j*M], lrel);
        counterror ++;
        if (counterror > 10)
          return err;
      }
    }
  }
  return err;
}
