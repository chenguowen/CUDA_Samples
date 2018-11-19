
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <malloc.h>
#include <random>
#include <time.h>

const int threadPerBlock = 16;

texture<int> texA;
texture<int> texB;

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

cudaError_t mulWithCuda(const int *a, const int *b, int *result, const int M, const int N, const int S);

cudaError_t mulWithCudaTex(const int *a, const int *b, int *result, const int M, const int N, const int S);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

/* MatMultiply：CPU下矩阵乘法
*  a:第一个矩阵指针，表示a[M][N];
*  b:第二个矩阵指针，表示b[N][S];
*  result:结果矩阵，表示为result[M][S];
*/
void CPUMatMultiply(const int * a,const int * b, int *result,const int M,const int N,const int S)
{
	for (int i = 0; i < M; i++)
	{
		for (int j = 0; j < S; j++)
		{
			int index = i * S + j;
			result[index] = 0;

			//计算每一个元素的结果
			for (int k = 0; k < N; k++)
			{
				result[index] += a[i * N + k] * b[k * S + j];
			}
		}
	}
}

/* gpuMatMultKernel：GPU下矩阵乘法核函数
*  a:第一个矩阵指针，表示a[M][N]
*  b:第二个矩阵指针，表示b[N][S]
*  result:结果矩阵，表示result[M][S]
*/
__global__ void gpuMatMultKernel(const int *a, const int *b, int *result, const int M, const int N, const int S)
{
	//int threadId = threadIdx.x + blockIdx.x * blockDim.x;

	int threadId = (blockIdx.y * blockDim.y + threadIdx.y) * gridDim.x * blockDim.x + blockIdx.x * blockDim.x + threadIdx.x;
	if (threadId < M * S)
	{
		int row = threadId / S;
		int column = threadId % S;

		result[threadId] = 0;
		for (int i = 0; i < N; i++)
		{
			result[threadId] += a[row * N + i] * b[i * S + column];
		}
	}
}

/* gpuMatMultWithSharedKernel：GPU下使用shared内存的矩阵乘法
*  a:第一个矩阵指针，表示a[height_A][width_A]
*  b:第二个矩阵指针，表示b[width_A][width_B]
*  result:结果矩阵，表示result[height_A][width_B]
*/
template<int BLOCK_SIZE>
__global__ void gpuMatMultWithSharedKernel(const int *a, const int *b, int *result, const int height_A, const int width_A, const int width_B)
{
	int block_x = blockIdx.x;
	int block_y = blockIdx.y;
	int thread_x = threadIdx.x;
	int thread_y = threadIdx.y;

	if ((thread_y + block_y * blockDim.y) * width_B + block_x * blockDim.x + thread_x >= height_A * width_B)
	{
		return;
	}

	const int begin_a = block_y * blockDim.y * width_A;
	const int end_a = begin_a + width_A - 1;
	const int step_a = blockDim.x;

	const int begin_b = block_x * blockDim.x;
	const int step_b = blockDim.y * width_B;

	int result_temp = 0;

	for (int index_a = begin_a, int index_b = begin_b;
		index_a < end_a; index_a += step_a, index_b += step_b)
	{
		__shared__ int SubMat_A[BLOCK_SIZE][BLOCK_SIZE];
		__shared__ int SubMat_B[BLOCK_SIZE][BLOCK_SIZE];

		SubMat_A[thread_y][thread_x] = a[index_a + thread_y * width_A + thread_x];
		SubMat_B[thread_y][thread_x] = b[index_b + thread_y * width_B + thread_x];

		__syncthreads();

		for (int i = 0; i < BLOCK_SIZE; i++)
		{
			result_temp += SubMat_A[thread_y][i] * SubMat_B[i][thread_x];
		}

		__syncthreads();
	}

	int begin_result = block_y * blockDim.y * width_B + begin_b;
	result[begin_result + thread_y * width_B + thread_x] = result_temp;
}

/* gpuMatMultWithTextureKernel：GPU下使用texture内存的矩阵乘法
*  result：结果矩阵，表示为result[M][S];
*  M：表示为矩阵A与矩阵result的行数
*  N：表示矩阵A的列数，矩阵B的行数
*  S：表示矩阵B和矩阵result的列数
*/
__global__ void gpuMatMultWithTextureKernel(int * result, const int M, const int N, const int S)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;

	if (offset < M * S)
	{
		int a = 0, b = 0;
		int temp_result = 0;
		for (int i = 0; i < N; i++)
		{
			a = tex1Dfetch(texA, y * N + i);
			b = tex1Dfetch(texB, i * S + x);
			temp_result += a * b;
		}
		result[offset] = temp_result;
	}
}


// main主函数，分别运行CPU和GPU矩阵乘法函数，比较二者的运行时间
int main()
{

	//确定矩阵的大小
	int M = 0, N = 0, S = 0;
	printf("please input the value of M (Mat a's row):");
	scanf("%d", &M);
	printf("please input the value of N (Mat a's column and Mat b's row):");
	scanf("%d", &N);
	printf("please input the value of S (Mat b's column):");
	scanf("%d", &S);

	//分配矩阵空间
	int * a = (int *)malloc(M * N * sizeof(int));
	if (NULL == a)
	{
		printf("the malloc of Mat a is failed!\n");
		return 0;
	}
	int * b = (int *)malloc(N * S * sizeof(int));
	if (NULL == b)
	{
		printf("the malloc of Mat b is failed!\n");
		return 0;
	}
	//cpu与gpu的结果矩阵分别存放
	int * cpuResult = (int *)malloc(M * S * sizeof(int));
	if (NULL == cpuResult)
	{
		printf("the malloc of Mat cpuResult is failed!\n");
		return 0;
	}
	int * gpuResult = (int *)malloc(M * S * sizeof(int));
	if (NULL == cpuResult)
	{
		printf("the malloc of Mat gpuResult is failed!\n");
		return 0;
	}

	//生成矩阵数据
	printf("\nstart random the Mat a...\n");
	for (int i = 0; i < M; i++)
	{
		for (int j = 0; j < N; j++)
		{
			a[i * N + j] = rand() % 5;
		}
	}

	printf("\nstart random the Mat b...\n");
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < S; j++)
		{
			b[i * S + j] = rand() % 5;
		}
	}

	//统计CPU运行乘法的时间
	clock_t start, finish;
	double totalTime = 0.0;
	start = clock();

	//调用CPU矩阵乘法函数
	CPUMatMultiply(a, b, cpuResult, M, N, S);

	finish = clock();
	totalTime = (double)(finish - start) / CLOCKS_PER_SEC;
	printf("\nThe total time is %lf seconds!\n", totalTime);

	//调用GPU矩阵乘法函数
	cudaError_t cudaStatus = mulWithCuda(a, b, gpuResult, M, N, S);
	//cudaError_t cudaStatus = mulWithCudaTex(a, b, gpuResult, M, N, S);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "mulWithCuda failed!");
		return 0;
	}
	//打印结果矩阵result
	/*printf("\nthe result of CPU :\n");
	for (int i = 0; i < M; i++)
	{
		for (int j = 0; j < S; j++)
		{
			printf("%d\t", cpuResult[i * M + j]);
		}
		printf("\n");
	}

	printf("\nthe result of GPU :\n");
	for (int i = 0; i < M; i++)
	{
		for (int j = 0; j < S; j++)
		{
			printf("%d\t", gpuResult[i * M + j]);
		}
		printf("\n");
	}*/

	//确认CPU和GPU矩阵乘法结果是否相同，从而说明结果是否正确
	for (int i = 0; i < M; i++)
	{
		for (int j = 0; j < S; j++)
		{
			if (cpuResult[i * M + j] != gpuResult[i * M + j])
			{
				printf("the Result isn't equal!\n");
				return 0;
			}
		}
	}

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}

// 调用CUDA运行GPU矩阵乘法核函数
cudaError_t mulWithCuda(const int *a, const int *b, int *result, const int M, const int N, const int S)
{
	int *dev_a = 0;
	int *dev_b = 0;
	int *dev_result = 0;

	cudaError_t cudaStatus;

	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaSetDevice failed! Do you have a CUDA-capable GPU installed?\n");
		goto Error;
	}

	cudaStatus = cudaMalloc((void **)&dev_a, M * N * sizeof(int));
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc dev_a failed!\n");
		goto Error;
	}

	cudaStatus = cudaMalloc((void **)&dev_b, N * S * sizeof(int));
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc dev_b failed!\n");
		goto Error;
	}

	cudaStatus = cudaMalloc((void **)&dev_result, M * S * sizeof(int));
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc dev_result failed!\n");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_a, a, M * N * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudamemcpy dev_a failed!\n");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_b, b, N * S * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMemcpy dev_b failed!\n");
		goto Error;
	}

	cudaEvent_t gpuStart, gpuFinish;
	float elapsedTime;
	cudaEventCreate(&gpuStart);
	cudaEventCreate(&gpuFinish);
	cudaEventRecord(gpuStart, 0);

	/*const int THREADNUM = 256;
	const int BLOCKNUM = (M * S + 255) / 256;*/

	const int BLOCK_SIZE = 16;
	dim3 block(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid((S + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);
	gpuMatMultKernel << <grid, block >> >(dev_a, dev_b, dev_result, M, N, S);
	//gpuMatMultWithSharedKernel<16> << <grid, block >> >(dev_a, dev_b, dev_result, M, N, S);

	cudaEventRecord(gpuFinish, 0);
	cudaEventSynchronize(gpuFinish);
	cudaEventElapsedTime(&elapsedTime, gpuStart, gpuFinish);
	printf("\nThe runing time of GPU on Mat Multiply is %f seconds.\n", elapsedTime / 1000.0);

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "MulKernel launch failed: %s!\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaDeviceSynchronize return Error code %d after Kernel launched!\n", cudaStatus);
		goto Error;
	}

	cudaStatus = cudaMemcpy(result, dev_result, M * S * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMemcpy result failed!\n");
		goto Error;
	}

Error:
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_result);

	return cudaStatus;
}

//调用CUDA运行GPU矩阵乘法核函数
//将矩阵A与矩阵B绑定到纹理内存中
cudaError_t mulWithCudaTex(const int *a, const int *b, int *result, const int M, const int N, const int S)
{
	int * dev_a = 0;
	int * dev_b = 0;
	int * dev_result = 0;

	cudaError_t cudaStatus;

	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaSetDevice failed! Do you have a CUDA_capable GPU installed?\n");
		goto Error;
	}

	cudaStatus = cudaMalloc((void **)&dev_a, M * N * sizeof(int));
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc dev_a failed!\n");
		goto Error;
	}

	cudaStatus = cudaMalloc((void **)&dev_b, N * S * sizeof(int));
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc dev_b failed!\n");
		goto Error;
	}

	cudaStatus = cudaMalloc((void **)&dev_result, M * S * sizeof(int));
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc dev_result failed!\n");
		goto Error;
	}

	cudaChannelFormatDesc desc = cudaCreateChannelDesc<int>();
	cudaStatus = cudaBindTexture(NULL, texA, dev_a, desc, M * N * sizeof(int));
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaBindTexture texA failed!\n");
		goto Error;
	}

	cudaStatus = cudaBindTexture(NULL, texB, dev_b, desc, N * S * sizeof(int));
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaBindTexture texB failed!\n");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_a, a, M * N * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudamemcpy dev_a failed!\n");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_b, b, N * S * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMemcpy dev_b failed!\n");
		goto Error;
	}

	cudaEvent_t gpuStart, gpuFinish;
	float elapsedTime;
	cudaEventCreate(&gpuStart);
	cudaEventCreate(&gpuFinish);
	cudaEventRecord(gpuStart, 0);

	const int BLOCK_SIZE = 16;
	if ((M % BLOCK_SIZE != 0) && (S % BLOCK_SIZE != 0))
	{
		fprintf(stderr, "M or S can't be dividen by 16!\n");
		goto Error;
	}

	dim3 block(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid(S / BLOCK_SIZE, M / BLOCK_SIZE);
	gpuMatMultWithTextureKernel << <grid, block >> >(dev_result, M, N, S);

	cudaEventRecord(gpuFinish, 0);
	cudaEventSynchronize(gpuFinish);
	cudaEventElapsedTime(&elapsedTime, gpuStart, gpuFinish);
	printf("\nThe runing time of GPU on Mat Multiply is %f seconds.\n", elapsedTime / 1000.0);

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "MulKernel launch failed: %s!\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaDeviceSynchronize return Error code %d after Kernel launched!\n", cudaStatus);
		goto Error;
	}

	cudaStatus = cudaMemcpy(result, dev_result, M * S * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMemcpy result failed!\n");
		goto Error;
	}

Error:
	cudaUnbindTexture(texA);
	cudaUnbindTexture(texB);
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_result);

	return cudaStatus;

}

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <stdio.h>
#include <malloc.h>
#include <random>
#include <time.h>

#define threadPerBlock 4

//typedef int test_type;
typedef float test_type;
//typedef double test_type;
//typedef long long test_type;
//typedef unsigned short test_type;
//typedef unsigned char test_type;

texture<int> texA;
texture<int> texB;

cudaError_t addWithCuda(int *c, const int *mul_a, const int *mul_b, unsigned int size);
cudaError_t mulWithCuda(const int *mul_a, const int *mul_b, int *result, const int M, const int N, const int S);
template< class T_type > cudaError_t mulWithCuda_Shared_ATA(const T_type *mul_a, T_type *result, const int M, const int N);
template< class T_type > cudaError_t mulWithCuda_Shared(const T_type *mul_a, const T_type *mul_b, T_type *result, const int M, const int N, const int S) ; 
// cudaError_t mulWithCuda_Shared(const int *mul_a, const int *mul_b, int *result, const int M, const int N, const int S);
cudaError_t mulWithCudaTex(const int *mul_a, const int *mul_b, int *result, const int M, const int N, const int S);

__global__ void addKernel(int *c, const int *mul_a, const int *mul_b)
{
	int i = threadIdx.x;
	c[i] = mul_a[i] + mul_b[i];
}

/* MatMultiply：CPU下矩阵乘法
*  mul_a:第一个矩阵指针，表示mul_a[M][N];
*  mul_b:第二个矩阵指针，表示mul_b[N][S];
*  result:结果矩阵，表示为result[M][S];
*/
template< class T_type > void CPUMatMultiply(const T_type * mul_a, const T_type * mul_b, T_type *result, const int M, const int N, const int S)
{
	for (int i = 0; i < M; i++)
	{
		for (int j = 0; j < S; j++)
		{
			int index = i * S + j;
			result[index] = 0; 
			//计算每一个元素的结果
			for (int k = 0; k < N; k++)
			{
				result[index] += mul_a[i * N + k] * mul_b[k * S + j];
			}
		}
	}
}

/* gpuMatMultKernel：GPU下矩阵乘法核函数
*  mul_a:第一个矩阵指针，表示mul_a[M][N]
*  mul_b:第二个矩阵指针，表示mul_b[N][S]
*  result:结果矩阵，表示result[M][S]
*/
__global__ void gpuMatMultKernel(const int *mul_a, const int *mul_b, int *result, const int M, const int N, const int S)
{
	//int threadId = threadIdx.x + blockIdx.x * blockDim.x;

	int threadId = (blockIdx.y * blockDim.y + threadIdx.y) * gridDim.x * blockDim.x + blockIdx.x * blockDim.x + threadIdx.x;
	if (threadId < M * S)
	{
		int row = threadId / S;
		int column = threadId % S;

		result[threadId] = 0;
		for (int i = 0; i < N; i++)
		{
			result[threadId] += mul_a[row * N + i] * mul_b[i * S + column];
		}
	}
}
// 
template<int BLOCK_SIZE, class T_type > 
__global__ void gpuMatMultWithSharedKernel(const T_type *mul_a, const T_type *mul_b, T_type *result, const int height_A, const int width_A, const int width_B)
{
	int block_x = blockIdx.x;
	int block_y = blockIdx.y;
	int thread_x = threadIdx.x;
	int thread_y = threadIdx.y;

	if ((thread_y + block_y * blockDim.y) * width_B + block_x * blockDim.x + thread_x >= height_A * width_B)
	{
		return;
	}

	const int begin_a = block_y * blockDim.y * width_A;
	const int end_a = begin_a + width_A ;
	const int step_a = blockDim.x;

	const int begin_b = block_x * blockDim.x;
	const int step_b = blockDim.y * width_B;

	T_type result_temp = 0;

	for (int index_a = begin_a, int index_b = begin_b; index_a < end_a; index_a += step_a, index_b += step_b)
	{
		__shared__ T_type SubMat_A[BLOCK_SIZE][BLOCK_SIZE];
		__shared__ T_type SubMat_B[BLOCK_SIZE][BLOCK_SIZE];

		SubMat_A[thread_y][thread_x] = mul_a[index_a + thread_y * width_A + thread_x];
		SubMat_B[thread_y][thread_x] = mul_b[index_b + thread_y * width_B + thread_x];

		//SubMat_A[thread_y][thread_x] = tex1Dfetch(texA, index_a + thread_y * width_A + thread_x);
		//SubMat_B[thread_y][thread_x] = tex1Dfetch(texB, index_b + thread_y * width_B + thread_x);

		__syncthreads();

		for (int i = 0; i < BLOCK_SIZE; i++)
		{
			result_temp += SubMat_A[thread_y][i] * SubMat_B[i][thread_x];
		}

		__syncthreads();
	}

	int begin_result = block_y * blockDim.y * width_B + begin_b;
	result[begin_result + thread_y * width_B + thread_x] = result_temp;
} 
 
// 
template<int BLOCK_SIZE, class T_type > __global__ void gpuMatMultWithSharedKernel_ATA(const T_type *mul_a, T_type *result, const int height_A, const int width_A)
{
	int block_x = blockIdx.x;
	int block_y = blockIdx.y;
	int thread_x = threadIdx.x;
	int thread_y = threadIdx.y;
	// 
	if ( thread_y + block_y * blockDim.y >= height_A || block_x * blockDim.x + thread_x >= height_A * height_A )
	{
		return;
	}
	// 
	//if ((thread_y + block_y * blockDim.y) * width_B + block_x * blockDim.x + thread_x >= height_A * width_B)
	//{
	//	return;
	//}

	const int begin_a = block_y * blockDim.y * width_A; // 
	const int end_a   = begin_a + width_A - 1;
	const int step_a  = blockDim.x;
	// 
	const int begin_b = block_x * blockDim.y * width_A; // 
	const int step_b  = blockDim.x ;
	//
	T_type result_temp = 0; // 
	for (int index_a = begin_a, int index_b = begin_b; index_a < end_a; index_a += step_a, index_b += step_b)
	{
		__shared__ T_type SubMat_A[BLOCK_SIZE][BLOCK_SIZE];//
		__shared__ T_type SubMat_B[BLOCK_SIZE][BLOCK_SIZE];//
		SubMat_A[thread_y][thread_x] = mul_a[index_a + thread_y * width_A + thread_x];
		SubMat_B[thread_y][thread_x] = mul_a[index_b + thread_y * width_A + thread_x];
		//SubMat_A[thread_y][thread_x] = tex1Dfetch(texA, index_a + thread_y * width_A + thread_x);
		//SubMat_B[thread_y][thread_x] = tex1Dfetch(texB, index_b + thread_y * width_B + thread_x);
		__syncthreads(); //
		//SubMat_A[thread_y][i] * SubMat_B[i][thread_x];
		for (int i = 0; i < BLOCK_SIZE; i++)
		{
			result_temp += SubMat_A[i][thread_y] * SubMat_B[i][thread_x];
		}
		__syncthreads();
	}
	int begin_result = block_y * blockDim.y * height_A + begin_b;
	result[begin_result + thread_y * height_A + thread_x] = result_temp;
	// result[(thread_y + block_y * blockDim.y) * height_A + block_x * blockDim.x + thread_x] = result_temp;
}


/* gpuMatMultWithSharedKernel：GPU下使用shared内存的矩阵乘法
*  mul_a:第一个矩阵指针，表示mul_a[height_A][width_A]
*  mul_b:第二个矩阵指针，表示mul_b[width_A][width_B]
*  result:结果矩阵，表示result[height_A][width_B]
*/
//template<int BLOCK_SIZE, class T >
//__global__ void gpuMatMultWithSharedKernel(const int *mul_a, const int *mul_b, int *result, const int height_A, const int width_A, const int width_B)
//{
//	int block_x = blockIdx.x;
//	int block_y = blockIdx.y;
//	int thread_x = threadIdx.x;
//	int thread_y = threadIdx.y;
//
//	if ((thread_y + block_y * blockDim.y) * width_B + block_x * blockDim.x + thread_x >= height_A * width_B)
//	{
//		return;
//	}
//
//	const int begin_a = block_y * blockDim.y * width_A;
//	const int end_a   = begin_a + width_A - 1;
//	const int step_a  = blockDim.x;
//
//	const int begin_b = block_x * blockDim.x;
//	const int step_b  = blockDim.y * width_B;
//
//	int result_temp = 0;
//
//	for (int index_a = begin_a, int index_b = begin_b;	index_a < end_a; index_a += step_a, index_b += step_b)
//	{
//		__shared__ int SubMat_A[BLOCK_SIZE][BLOCK_SIZE];
//		__shared__ int SubMat_B[BLOCK_SIZE][BLOCK_SIZE];
//
//		SubMat_A[thread_y][thread_x] = mul_a[index_a + thread_y * width_A + thread_x];
//		SubMat_B[thread_y][thread_x] = mul_b[index_b + thread_y * width_B + thread_x];
//
//		//SubMat_A[thread_y][thread_x] = tex1Dfetch(texA, index_a + thread_y * width_A + thread_x);
//		//SubMat_B[thread_y][thread_x] = tex1Dfetch(texB, index_b + thread_y * width_B + thread_x);
//
//		__syncthreads();
//
//		for (int i = 0; i < BLOCK_SIZE; i++)
//		{
//			result_temp += SubMat_A[thread_y][i] * SubMat_B[i][thread_x];
//		}
//
//		__syncthreads();
//	}
//
//	int begin_result = block_y * blockDim.y * width_B + begin_b;
//	result[begin_result + thread_y * width_B + thread_x] = result_temp;
//}

/* gpuMatMultWithTextureKernel：GPU下使用texture内存的矩阵乘法
*  result：结果矩阵，表示为result[M][S];
*  M：表示为矩阵A与矩阵result的行数
*  N：表示矩阵A的列数，矩阵B的行数
*  S：表示矩阵B和矩阵result的列数
*/
__global__ void gpuMatMultWithTextureKernel(int * result, const int M, const int N, const int S)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;

	if (offset < M * S)
	{
		int mul_a = 0, mul_b = 0;
		int temp_result = 0;
		for (int i = 0; i < N; i++)
		{
			mul_a = tex1Dfetch(texA, y * N + i);
			mul_b = tex1Dfetch(texB, i * S + x);
			temp_result += mul_a * mul_b;
		}
		result[offset] = temp_result;
	}
}


// main主函数，分别运行CPU和GPU矩阵乘法函数，比较二者的运行时间
int main()
{ 
	//确定矩阵的大小
	int M = 0, N = 0, S = 0;
	printf("please input the value of M (Mat mul_a's row):");
	scanf("%d", &M);
	printf("please input the value of N (Mat mul_a's column and Mat mul_b's row):");
	scanf("%d", &N);
	printf("please input the value of S (Mat mul_b's column):");
	scanf("%d", &S);

	//分配矩阵空间
	test_type * mul_a = (test_type *)malloc(M * N * sizeof(test_type)); if (NULL == mul_a){ printf("the malloc of Mat mul_a is failed!\n");	return 0; }
	test_type * mul_b = (test_type *)malloc(N * S * sizeof(test_type)); if (NULL == mul_b){ printf("the malloc of Mat mul_b is failed!\n");	return 0; }
	//cpu与gpu的结果矩阵分别存放
	test_type * cpuResult = (test_type *)malloc(M * S * sizeof(test_type)); if (NULL == cpuResult){ printf("the malloc of Mat cpuResult is failed!\n");	return 0; }
	test_type * gpuResult = (test_type *)malloc(M * S * sizeof(test_type)); if (NULL == cpuResult){ printf("the malloc of Mat gpuResult is failed!\n");	return 0; }

	//生成矩阵数据
	printf("\nstart random the Mat mul_a...\n");
	for (int i = 0; i < M; i++)
	{
		for (int j = 0; j < N; j++)
		{
			mul_a[i * N + j] = rand() % 5; // 
			mul_b[j * M + i] = mul_a[i * N + j]; 
		}
	}

	//printf("\nstart random the Mat mul_b...\n");
	//for (int i = 0; i < N; i++)
	//{
	//	for (int j = 0; j < S; j++)
	//	{
	//		mul_b[i * S + j] = mul_a[i + j * M];// rand() % 5;
	//	}
	//}

	double result = 0; 
	printf("\n  ..\n");
	for (int i = 0; i < M; i++)
	{
		for (int j = 0; j < N; j++)
		{
			result += abs(mul_a[i * N + j] - mul_b[i + j * M]);
		}
	}

	std::cout << "result =" << result << std::endl;

	//统计CPU运行乘法的时间
	clock_t start, finish;
	double totalTime = 0.0;

	start = clock(); //调用CPU矩阵乘法函数
	CPUMatMultiply<test_type>(mul_a, mul_b, cpuResult, M, N, S);	finish = clock();
	totalTime = (double)(finish - start) / CLOCKS_PER_SEC;
	printf("\nThe total time is %lf seconds!\n", totalTime);



	cudaError_t cudaStatus5 = mulWithCuda_Shared_ATA<test_type>(mul_a, gpuResult, M, N); 
	printf("\n GPU-ATA result!\n", totalTime);
	for (int i = 0; i < M; i++)
	{
		for (int j = 0; j < S; j++)
		{
			std::cout << gpuResult[i * M + j] << " ";
		}
		std::cout << std::endl;
	}
	cudaError_t cudaStatus1 = mulWithCuda_Shared<test_type>(mul_a, mul_b, gpuResult, M, N, S);
	printf("\n GPU result!\n", totalTime);
	for (int i = 0; i < M; i++)
	{
		for (int j = 0; j < S; j++)
		{
			std::cout << gpuResult[i * M + j] << " "; 
		}
		std::cout << std::endl;
	}
	printf("\n CPU result!\n", totalTime);
	for (int i = 0; i < M; i++)
	{
		for (int j = 0; j < S; j++)
		{
			std::cout << cpuResult[i * M + j] << " "; 
		}
		std::cout << std::endl;
	}

	printf("\n A matrix !\n", totalTime);
	for (int i = 0; i < M; i++)
	{
		for (int j = 0; j < N; j++)
		{
			std::cout << mul_a[i * N + j] << " ";
		}
		std::cout << std::endl;
	}
	
	printf("\n B matrix !\n", totalTime);
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < S; j++)
		{
			std::cout << mul_b[i * S + j] << " ";
		}
		std::cout << std::endl;
	}
	////调用GPU矩阵乘法函数
	//cudaError_t cudaStatus0 = mulWithCuda(mul_a, mul_b, gpuResult, M, N, S); 
	//printf(" \nComparing mulWithCuda amd mulWithCpu \n");
	//for (int i = 0; i < M; i++)
	//{
	//	for (int j = 0; j < S; j++)
	//	{
	//		if (cpuResult[i * M + j] != gpuResult[i * M + j])
	//		{
	//			printf("the Result isn't equal!\n");
	//			return 0;
	//		}
	//	}
	//}
	//printf("the Result is equal!\n\n");


	//cudaError_t cudaStatus4 = mulWithCudaTex(mul_a, mul_b, gpuResult, M, N, S);  
	//printf(" \nComparing mulWithCudaTex amd mulWithCpu \n");
	//for (int i = 0; i < M; i++)
	//{
	//	for (int j = 0; j < S; j++)
	//	{
	//		if (cpuResult[i * M + j] != gpuResult[i * M + j])
	//		{
	//			printf("the Result isn't equal!\n");
	//			return 0;
	//		}
	//	}
	//}
	//printf("the Result is equal!\n\n");



	//cudaError_t cudaStatus1 = mulWithCuda_Shared<test_type>(mul_a, mul_b, gpuResult, M, N, S);
	printf(" \nComparing mulWithCuda_Shared amd mulWithCpu \n");
	for (int i = 0; i < M; i++)
	{
		for (int j = 0; j < S; j++)
		{
			if (cpuResult[i * M + j] != gpuResult[i * M + j])
			{
				printf("the Result isn't equal!\n");
				return 0;
			}
		}
	}
	printf("the Result is equal!\n\n");


	//cudaError_t cudaStatus5 = mulWithCuda_Shared_ATA<test_type>(mul_a, gpuResult, M, N);

	//printf(" \nComparing mulWithCuda_Shared amd mulWithCpu \n");
	//for (int i = 0; i < M; i++)
	//{
	//	for (int j = 0; j < S; j++)
	//	{
	//		if (cpuResult[i * M + j] != gpuResult[i * M + j])
	//		{
	//			printf("the Result isn't equal!\n");
	//			return 0;
	//		}
	//	}
	//}
	//printf("the Result is equal!\n\n");



	//打印结果矩阵result
	/*printf("\nthe result of CPU :\n");
	for (int i = 0; i < M; i++)
	{
	for (int j = 0; j < S; j++)
	{
	printf("%d\t", cpuResult[i * M + j]);
	}
	printf("\n");
	}
	printf("\nthe result of GPU :\n");
	for (int i = 0; i < M; i++)
	{
	for (int j = 0; j < S; j++)
	{
	printf("%d\t", gpuResult[i * M + j]);
	}
	printf("\n");
	}*/

	//确认CPU和GPU矩阵乘法结果是否相同，从而说明结果是否正确


	return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *mul_a, const int *mul_b, unsigned int size)
{
	int *dev_a = 0;
	int *dev_b = 0;
	int *dev_c = 0;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on mul_a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have mul_a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_a, mul_a, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_b, mul_b, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	// Launch mul_a kernel on the GPU with one thread for each element.
	addKernel << <1, size >> >(dev_c, dev_a, dev_b);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(dev_c);
	cudaFree(dev_a);
	cudaFree(dev_b);

	return cudaStatus;
}

// 调用CUDA运行GPU矩阵乘法核函数
cudaError_t mulWithCuda(const int *mul_a, const int *mul_b, int *result, const int M, const int N, const int S)
{
	int *dev_a = 0;
	int *dev_b = 0;
	int *dev_result = 0;

	cudaError_t cudaStatus;

	cudaStatus = cudaSetDevice(0);

	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaSetDevice failed! Do you have mul_a CUDA-capable GPU installed?\n");
		goto Error;
	}

	cudaStatus = cudaMalloc((void **)&dev_a, M * N * sizeof(int));
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc dev_a failed!\n");
		goto Error;
	}

	cudaStatus = cudaMalloc((void **)&dev_b, N * S * sizeof(int));
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc dev_b failed!\n");
		goto Error;
	}

	cudaStatus = cudaMalloc((void **)&dev_result, M * S * sizeof(int));
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc dev_result failed!\n");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_a, mul_a, M * N * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudamemcpy dev_a failed!\n");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_b, mul_b, N * S * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMemcpy dev_b failed!\n");
		goto Error;
	}

	cudaEvent_t gpuStart, gpuFinish;
	float elapsedTime;
	cudaEventCreate(&gpuStart);
	cudaEventCreate(&gpuFinish);
	cudaEventRecord(gpuStart, 0);

	/*const int THREADNUM = 256;
	const int BLOCKNUM = (M * S + 255) / 256;*/

	const int BLOCK_SIZE = 16;
	dim3 block(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid((S + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);
	// gpuMatMultKernel << <grid, block >> >(dev_a, dev_b, dev_result, M, N, S);
	gpuMatMultKernel << <grid, block >> >(dev_a, dev_b, dev_result, M, N, S);

	cudaEventRecord(gpuFinish, 0);
	cudaEventSynchronize(gpuFinish);
	cudaEventElapsedTime(&elapsedTime, gpuStart, gpuFinish);

	printf("\nThe runing time of GPU on Mat Multiply using no texture is %f seconds.\n", elapsedTime / 1000.0);

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "MulKernel launch failed: %s!\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaDeviceSynchronize return Error code %d after Kernel launched!\n", cudaStatus);
		goto Error;
	}

	cudaStatus = cudaMemcpy(result, dev_result, M * S * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMemcpy result failed!\n");
		goto Error;
	}

Error:
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_result);
	return cudaStatus;
}
template< class T_type > cudaError_t mulWithCuda_Shared_ATA(const T_type *mul_a, T_type *result, const int M, const int N)
{
	T_type *dev_a = 0;
	T_type *dev_result = 0;  
	cudaError_t cudaStatus;

	cudaStatus = cudaSetDevice(0); if (cudaStatus != cudaSuccess){ fprintf(stderr, "cudaSetDevice failed! Do you have mul_a CUDA-capable GPU installed?\n");	goto Error; }
	cudaStatus = cudaMalloc((void **)&dev_a, M * N * sizeof(T_type)); if (cudaStatus != cudaSuccess){ fprintf(stderr, "cudaMalloc dev_a failed!\n");	goto Error; }
	cudaStatus = cudaMalloc((void **)&dev_result, M * M * sizeof(T_type));	if (cudaStatus != cudaSuccess){ fprintf(stderr, "cudaMalloc dev_result failed!\n");	goto Error; }
	cudaStatus = cudaMemcpy(dev_a, mul_a, M * N * sizeof(T_type), cudaMemcpyHostToDevice);	if (cudaStatus != cudaSuccess){ fprintf(stderr, "cudamemcpy dev_a failed!\n");	goto Error; }
	cudaEvent_t gpuStart, gpuFinish;  float elapsedTime;	cudaEventCreate(&gpuStart);	cudaEventCreate(&gpuFinish); cudaEventRecord(gpuStart, 0);

	const int BLOCK_SIZE = threadPerBlock;
	dim3 block(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid((M + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);
	gpuMatMultWithSharedKernel_ATA<threadPerBlock, T_type> << <grid, block >> >(dev_a, dev_result, M, N);

	cudaEventRecord(gpuFinish, 0);
	cudaEventSynchronize(gpuFinish);
	cudaEventElapsedTime(&elapsedTime, gpuStart, gpuFinish);
	printf("\nThe runing time of GPU on Mat Multiply ATA using sharing memory is %f seconds.\n", elapsedTime / 1000.0);
	cudaStatus = cudaGetLastError(); if (cudaStatus != cudaSuccess)	{ fprintf(stderr, "MulKernel launch failed: %s!\n", cudaGetErrorString(cudaStatus));	goto Error; }
	cudaStatus = cudaMemcpy(result, dev_result, M * M * sizeof(T_type), cudaMemcpyDeviceToHost); if (cudaStatus != cudaSuccess){ fprintf(stderr, "cudaMemcpy result failed!\n"); goto Error; }

Error:
	cudaFree(dev_a);
	cudaFree(dev_result); 
	return cudaStatus; 
}
// 调用CUDA运行GPU矩阵乘法核函数
template< class T_type > cudaError_t mulWithCuda_Shared(const T_type *mul_a, const T_type *mul_b, T_type *result, const int M, const int N, const int S)
{
	T_type *dev_a = 0;
	T_type *dev_b = 0;
	T_type *dev_result = 0;

	cudaError_t cudaStatus;

	cudaStatus = cudaSetDevice(0); if (cudaStatus != cudaSuccess){fprintf(stderr, "cudaSetDevice failed! Do you have mul_a CUDA-capable GPU installed?\n");	goto Error;}
	cudaStatus = cudaMalloc((void **)&dev_a, M * N * sizeof(T_type)); if (cudaStatus != cudaSuccess){	fprintf(stderr, "cudaMalloc dev_a failed!\n");	goto Error;}
	cudaStatus = cudaMalloc((void **)&dev_b, N * S * sizeof(T_type)); if (cudaStatus != cudaSuccess){fprintf(stderr, "cudaMalloc dev_b failed!\n");	goto Error;	}
	cudaStatus = cudaMalloc((void **)&dev_result, M * S * sizeof(T_type));	if (cudaStatus != cudaSuccess){	fprintf(stderr, "cudaMalloc dev_result failed!\n");	goto Error;}

	//cudaChannelFormatDesc desc = cudaCreateChannelDesc<T_type>();
	//cudaStatus = cudaBindTexture(NULL, texA, dev_a, desc, M * N * sizeof(T_type));	if (cudaStatus != cudaSuccess){ fprintf(stderr, "cudaBindTexture texA failed!\n");	goto Error; }
	//cudaStatus = cudaBindTexture(NULL, texB, dev_b, desc, N * S * sizeof(T_type));	if (cudaStatus != cudaSuccess){ fprintf(stderr, "cudaBindTexture texB failed!\n");	goto Error; }
	cudaStatus = cudaMemcpy(dev_a, mul_a, M * N * sizeof(T_type), cudaMemcpyHostToDevice);	if (cudaStatus != cudaSuccess){ fprintf(stderr, "cudamemcpy dev_a failed!\n");	goto Error; }
	cudaStatus = cudaMemcpy(dev_b, mul_b, N * S * sizeof(T_type), cudaMemcpyHostToDevice);	if (cudaStatus != cudaSuccess){ fprintf(stderr, "cudaMemcpy dev_b failed!\n"); goto Error; }
	cudaEvent_t gpuStart, gpuFinish;  float elapsedTime;	cudaEventCreate(&gpuStart);	cudaEventCreate(&gpuFinish);cudaEventRecord(gpuStart, 0);

	/*const int THREADNUM = 256; const int BLOCKNUM = (M * S + 255) / 256;*/ 

	const int BLOCK_SIZE = threadPerBlock;
	dim3 block(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid((S + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);
	// gpuMatMultKernel << <grid, block >> >(dev_a, dev_b, dev_result, M, N, S);
	gpuMatMultWithSharedKernel<threadPerBlock, T_type> << <grid, block >> >(dev_a, dev_b, dev_result, M, N, S);

	cudaEventRecord(gpuFinish, 0);
	cudaEventSynchronize(gpuFinish);
	cudaEventElapsedTime(&elapsedTime, gpuStart, gpuFinish);
	printf("\nThe runing time of GPU on Mat Multiply using sharing memory is %f seconds.\n", elapsedTime / 1000.0);
	cudaStatus = cudaGetLastError(); if (cudaStatus != cudaSuccess)	{fprintf(stderr, "MulKernel launch failed: %s!\n", cudaGetErrorString(cudaStatus));	goto Error;	}
	// cudaStatus = cudaDeviceSynchronize(); if (cudaStatus != cudaSuccess){fprintf(stderr, "cudaDeviceSynchronize return Error code %d after Kernel launched!\n", cudaStatus);goto Error; }
	cudaStatus = cudaMemcpy(result, dev_result, M * S * sizeof(T_type), cudaMemcpyDeviceToHost); if (cudaStatus != cudaSuccess){ fprintf(stderr, "cudaMemcpy result failed!\n"); goto Error; }
	
Error:
	cudaUnbindTexture(texA);
	cudaUnbindTexture(texB);
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_result);

	return cudaStatus;
}

//调用CUDA运行GPU矩阵乘法核函数
//将矩阵A与矩阵B绑定到纹理内存中
cudaError_t mulWithCudaTex(const int *mul_a, const int *mul_b, int *result, const int M, const int N, const int S)
{
	int * dev_a = 0;
	int * dev_b = 0;
	int * dev_result = 0;

	cudaError_t cudaStatus;

	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaSetDevice failed! Do you have mul_a CUDA_capable GPU installed?\n");
		goto Error;
	}

	cudaStatus = cudaMalloc((void **)&dev_a, M * N * sizeof(int));
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc dev_a failed!\n");
		goto Error;
	}

	cudaStatus = cudaMalloc((void **)&dev_b, N * S * sizeof(int));
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc dev_b failed!\n");
		goto Error;
	}

	cudaStatus = cudaMalloc((void **)&dev_result, M * S * sizeof(int));
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc dev_result failed!\n");
		goto Error;
	}

	cudaChannelFormatDesc desc = cudaCreateChannelDesc<int>();
	cudaStatus = cudaBindTexture(NULL, texA, dev_a, desc, M * N * sizeof(int));
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaBindTexture texA failed!\n");
		goto Error;
	}

	cudaStatus = cudaBindTexture(NULL, texB, dev_b, desc, N * S * sizeof(int));
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaBindTexture texB failed!\n");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_a, mul_a, M * N * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudamemcpy dev_a failed!\n");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_b, mul_b, N * S * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMemcpy dev_b failed!\n");
		goto Error;
	}

	cudaEvent_t gpuStart, gpuFinish;
	float elapsedTime;
	cudaEventCreate(&gpuStart);
	cudaEventCreate(&gpuFinish);
	cudaEventRecord(gpuStart, 0);

	const int BLOCK_SIZE = 16;
	if ((M % BLOCK_SIZE != 0) && (S % BLOCK_SIZE != 0))
	{
		fprintf(stderr, "M or S can't be dividen by 16!\n");
		goto Error;
	}

	dim3 block(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid(S / BLOCK_SIZE, M / BLOCK_SIZE);
	gpuMatMultWithTextureKernel << <grid, block >> >(dev_result, M, N, S);

	cudaEventRecord(gpuFinish, 0);
	cudaEventSynchronize(gpuFinish);
	cudaEventElapsedTime(&elapsedTime, gpuStart, gpuFinish);
	printf("\nThe runing time of GPU on Mat Multiply using texture is %f seconds.\n", elapsedTime / 1000.0);

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "MulKernel launch failed: %s!\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaDeviceSynchronize return Error code %d after Kernel launched!\n", cudaStatus);
		goto Error;
	}

	cudaStatus = cudaMemcpy(result, dev_result, M * S * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMemcpy result failed!\n");
		goto Error;
	}

Error:
	cudaUnbindTexture(texA);
	cudaUnbindTexture(texB);
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_result);

	return cudaStatus;

}
