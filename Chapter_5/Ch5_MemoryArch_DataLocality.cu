/*  Programming Massively Parallel Processors 4th Edition Solution
    Chapter 5:
*/
#include<cuda.h>
#include<stdio.h>

//----------------------------------------------------------------------------------------------------------------
/*
1. Consider matrix addition. Can one use shared memory to reduce the
global memory bandwidth consumption? Hint: Analyze the elements that
are accessed by each thread and see whether there is any commonality
between threads.
    Ans. There wont be any significant performance boost.
         As no two thread shares data among each other. 
         Each thread just computes once C[i][j] = A[i][j] + B[i][j]
         Also since each thread accesses unique elements, 
         global memory accesses are already coalesced 

2. Draw the equivalent of Fig. 5.7 for a 8 x 8 matrix multiplication with 2 x 2
tiling and 4 x 4 tiling. Verify that the reduction in global memory bandwidth
is indeed proportional to the dimension size of the tiles.

3. What type of incorrect execution behavior can happen if one forgot to use
one or both __syncthreads() in the kernel of Fig. 5.9?
    Ans.If threads start reading shared memory before all threads have finished writing to it, you can get partially updated data.
    Some threads might read correct values, others might read old or uninitialized values.

4.Assuming that capacity is not an issue for registers or shared memory, give
one important reason why it would be valuable to use shared memory
instead of registers to hold values fetched from global memory? Explain
your answer.
    Ans. shared memory allows inter-thread communication within a thread block, 
    whereas registers are private to individual threads and cannot be accessed by other threads.

5. For our tiled matrix-matrix multiplication kernel, if we use a 32 x 32 tile,
what is the reduction of memory bandwidth usage for input matrices M
and N?
    Ans. There will be a reduction by 32

6. Assume that a CUDA kernel is launched with 1000 thread blocks, each of
which has 512 threads. If a variable is declared as a local variable in the
kernel, how many versions of the variable will be created through the
lifetime of the execution of the kernel?
    Ans. If variable declared as __shared__ then 1000 variables will be created for 1000 blocks
         If variable declared as __device__ ( global access to entire device ) 1 will be created.
         If variable is declared locally, private to each thread then there will be 512000.
7. In the previous question, if a variable is declared as a shared memory
variable, how many versions of the variable will be created through the
lifetime of the execution of the kernel?
    Ans. If variable declared as __shared__ then 1000 variables will be created for 1000 blocks

8.Consider performing a matrix multiplication of two input matrices with
dimensions N x N. How many times is each element in the input matrices
requested from global memory when:
a. There is no tiling?               Ans. Each element is requested N times
b. Tiles of size T x T are used?     Ans. Each element is requested N/T times

9. A kernel performs 36 floating-point operations and seven 32-bit global
memory accesses per thread. For each of the following device
properties, indicate whether this kernel is compute-bound or memory-
bound.
a. Peak FLOPS=200 GFLOPS, peak memory bandwidth=100 GB/second
b. Peak FLOPS=300 GFLOPS, peak memory bandwidth=250 GB/second

    Ans. a. Operational Intensity = 36/(4*7) = 1.29 FLOPS/byte
            Machine Balance Point = 200 GFLOPS/100 GBps = 2
            1.29 < 2  -> memory bound
        b.  Machine Balance Point  = 300 GLOPS/250 GBps =  1.2
            1.29 > 1.2 -> compute bound

*/
dim3 blockDim(BLOCK_WIDTH, BLOCK_WIDTH);
dim3 gridDim(A_width / blockDim.x, A_height / blockDim.y);
BlockTranspose<<<gridDim, blockDim>>>(A, A_width, A_height);

__global__ void 
BlockTranspose(float* A_elements, int A_width, int A_height){
    __shared__ float blockA[BLOCK_WIDTH][BLOCK_WIDTH];

    int baseIdx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    baseIdx += (blockIdx.y * BLOCK_SIZE + threadIdx.y) * A_width;

    blockA[threadIdx.y][threadIdx.x] = A_elements[baseIdx];
    A_elements[baseIdx] = blockA[threadIdx.x][threadIdx.y];
}
/*
10. To manipulate tiles, a new CUDA programmer has written a device kernel
that will transpose each tile in a matrix. The tiles are of size
BLOCK_WIDTH by BLOCK_WIDTH, and each of the dimensions of
matrix A is known to be a multiple of BLOCK_WIDTH. The kernel
invocation and code are shown below. BLOCK_WIDTH is known at
compile time and could be set anywhere from 1 to 20.
a. Out of the possible range of values for BLOCK_SIZE, for what values
of BLOCK_SIZE will this kernel function execute correctly on the
device?    
    Ans. BLOCK_WIDTH x BLOCKWIDTH <= 1024 for the kernel to work
         Therefor BLOCK_WIDTH <= 32 is sufficient
b. If the code does not execute correctly for all BLOCK_SIZE values, what
is the root cause of this incorrect execution behavior? Suggest a fix to the
code to make it work for all BLOCK_SIZE values.
    Ans. __synchthreads is required after blockA[threadIdx.y][threadIdx.x] = A_elements[baseIdx];
*/


//11.Consider the following CUDA kernel and the corresponding host function
that calls it:
__global__ void foo_kernel(float* a, float* b){
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    float x[4];
    __shared__ float y_s;
    __shared__ float b_s[128];
    for (unsigned int j = 0; j < 4; j++){
        x[j] = a[j * blockDim.x * gridDim.x + i];
    }
    if (threadIdx.x == 0){
        y_s = 7.4f;
    }
    b_s[threadIdx.x] = b[i];
    __syncthreads();
    b[i] = 2.5f * x[0] + 3.7f * x[1] + 6.3f * x[2] + 8.5f * x[3] 
            + ys * b_s[threadIdx.x] + b_s[(threadIdx.x + 3) % 128];
}
void foo(int* a_d, int* b_d){
    unsigned int N = 1024;
    foo_kernel <<< (N + 128 - 1) / 128, 128 >>>(a_d, b_d);
}
/*
a. How many versions of the variable i are there?
    Ans. i is declared locally, so it will be created for each thread launched (8*128)
b. How many versions of the array x[] are there?
    Ans. x[4] is declared locally, so equal to number of threads launched (8*128)
c. How many versions of the variable y_s are there?
    Ans. y_s is a shared variable, so it will be created for each block (8)
d. How many versions of the array b_s[] are there?
    Ans. b_s is a shared variable, so it will be created for each block (8)
e. What is the amount of shared memory used per block (in bytes)?
    Ans. float y_s will take 4 bytes and b_s will take 128*4 bytes for each block
f. What is the floating-point to global memory access ratio of the kernel (in OP/B)?
    Ans. 4 (loads from a) + 1 (load from b) + 1 (store to b) = 6 accesses.
         Total 10 operations are carried out. 5 Multiplication and 5 Addition.
         So 10/(4+1+1)= 10/6 OP/B 


12. Consider a GPU with the following hardware limits: 2048 threads/SM, 32
blocks/SM, 64K (65,536) registers/SM, and 96 KB of shared memory/SM.
For each of the following kernel characteristics, specify whether the kernel
can achieve full occupancy. If not, specify the limiting factor.
a. The kernel uses 64 threads/block, 27 registers/thread, and 4 KB of shared
memory/SM.
    Ans. Kernel uses :
         Number of blocks that can be launched = 2048 / 64 = 32 blocks
         Registers - 64 * 27 * 32 = 55296 < 64K regs. -> Allowed
         Shared Memory usage = 32 * 4 KB  > 96KB      -> Not Allowed
         Hence shared memory will put a limit on number of block to launch
         that will be 96KB/(Number of Block) = 4KB
         i.e Number of block that can be launched = 24 blocks

b. The kernel uses 256 threads/block, 31 registers/thread, and 8 KB of
shared memory/SM.
    Ans. Kernel uses :
         Number of blocks that can be launcehd = 2048/256 = 8
         Registers - 256 * 31 * 8 = 63488 < 64K regs. -> Allowed
         Shared Memory usage = 8 KB * 8 = 64KB < 96KB      -> Allowed
*/


//----------------------------------------------------------------------------------------------------------------
