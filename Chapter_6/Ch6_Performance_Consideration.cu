/*  Programming Massively Parallel Processors 4th Edition Solution
    Chapter 6: Performance Considerations

    Exercise Solution

*/
/*
Q6.1    Write a matrix multiplication kernel function that corresponds to the design
        illustrated in Fig. 6.4

    Solution:
    A, B- NxN input Matrix
    C   - NxN output Matrix
    TILE_WIDTH = BlockSize
*/

#define N 32
#define TILE_WIDTH 4 

__global__ void matMul_kernel ( float *A, float *B, float *C ){
    
    int row= blockIdx.y * blockDim.y + threadIdx.y;
    int col= blockIdx.x * blockDim.x + threadIdx.x;
    
    __shared__ float A_s[TILE_WIDTH][TILE_WIDTH];
    __shared__ float B_s[TILE_WIDTH][TILE_WIDTH];
    
    float acc=0.0f;

    for(int i=0; i< ((N+TILE_WIDTH-1)/TILE_WIDTH); i++){
        if( row < N && i*TILE_WIDTH+threadIdx.x < N ){
            A_s[threadIdx.y][threadIdx.x]= A[ row*N + i*TILE_WIDTH + threadIdx.x ]
        }else{
            A_s[threadIdx.y][threadIdx.x]= 0.0f;
        }
        if( col < N && i*TILE_WIDTH+threadIdx.y < N ){
            B_s[threadIdx.y][threadIdx.x]= B[ col*N + i*TILE_WIDTH + threadIdx.y ]
        }else{
            B_s[threadIdx.y][threadIdx.x]= 0.0f;
        }
        __syncthreads();

        for (int i = 0; i < TILE_WIDTH; ++i)
        acc += A_s[threadIdx.y][i] * B_s[i][threadIdx.x];

        __syncthreads();
    }
}

/*
Q6.2    For tiled matrix multiplication, of the possible range of values for
        BLOCK_SIZE, for what values of BLOCK_SIZE will the kernel completely
        avoid uncoalesced accesses to global memory? (You need to consider only
        square blocks.)
    Ans:
        A block dimension of 32 will give coalesced access.

        Lets consider case where block dimension is 16
        and N = 100
        The first block(0,0) will access data as
        
        warp lane:    0  1  2  3  4  5  6  .. 15     16   17   18 .. 31
        threadIdx.x   0  1  2  3  4  5  6     15      0    1    2    15
        threadIdx.y   0  0  0  0  0  0  0      0      1    1    1     1
        index:        0  1  2  3  4  5  6     15    100    101 102  115

        Here 16th and 17th thread, although consecutive are accessing different index
        Hence uncoalesced access.

        -- Reference taken from https://stackoverflow.com/questions/54003653/how-to-avoid-un-coalesced-accesses-in-matrix-multiplication-cuda-kernel

Q6.3 Consider the following CUDA kernel:
*/

__global__ void foo_kernel(float* a, float* b, float* c, float* d, float* e) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float a_s[256];
    __shared__ float bc_s[4 * 256];

    a_s[threadIdx.x] = a[i];

    for(unsigned int j = 0; j < 4; ++j) {
        bc_s[j * 256 + threadIdx.x] = b[j * blockDim.x * gridDim.x + i] + c[i * 4 + j];
    }

    __syncthreads();

    d[i + 8] = a_s[threadIdx.x];
    e[i * 8] = bc_s[threadIdx.x * 4];
}
/*
    For each of the following memory accesses, specify whether they are
    coalesced or uncoalesced or coalescing is not applicable:
    a. The access to array a of line 05
    Ans: Access is coalesced, as each consecutive thread in a warp are accessing consecutive elements.
    b. The access to array a_s of line 05
    Ans: Shared memory is SRAM, so coalescing is not 
    c. The access to array b of line 07
    Ans: Uncoalesced, as (j * blockDim.x * gridDim.x) makes access of nonconsecutive elements
    d. The access to array c of line 07
    Ans: Uncoalesced, as c[i*4+j] access are far off for each thread
    e. The access to array bc_s of line 07
    Ans. Memory coalescing doesn't apply for Shared memory access.
    f. The access to array a_s of line 10
    Ans. Memory coalescing doesn't apply for Shared memory access.
    g. The access to array d of line 10
    Ans. Coalesced. Here each thread access consecutive elements, shifted by 8
    h. The access to array bc_s of line 11
    Ans. Memory coalescing doesn't apply for Shared memory access.
    i. The access to array e of line 11
    Ans. Uncoalesced. Here each thread access non consecutive elements



4. What is the floating point to global memory access ratio (in OP/B) of each of
    the following matrix-matrix multiplication kernels?    
    a. The simple kernel described in Chapter 3, Multidimensional Grids and
        Data, without any optimizations applied.
    Ans. For Dim(M) - AxK and Dim(N) - KxB elements
        Number of global memory access for 1 op element(read)  = K (reading M) + K (Reading N)
        Numbef of global memory access for 1 op element(store) = 1 (P)
        Numbef of global memory access for AxB output P matrix = A * B * ( K+K+1 )
        Number of Operation: K multiplication + (K-1) addition

        Therefore, Floating point Operation / Global memory access = (2*K+1)/(A*B*(2*K+1)) 


    b. The kernel described in Chapter 5, Memory Architecture and Data
        Locality, with shared memory tiling applied using a tile size of 32 x 32.
    Ans. Tiled Matrix multiplication reduces access by TILE_WIDTH
        Number of global memory access for 1 op Tile(read)      = K/32 (reading M) + K/32 (Reading N)
        Numbef of global memory access inside 1 op Tile(read)   = 32*32*(K/32 + K/32)
        Numbef of global memory access for store = A * B 
        Numbef of global memory access for AxB/(32*32) output tile of P matrix = A/32 * B/32 * 32*32*( K/32 + K/32 ) + A*B
        Number of Operation: K multiplication + (K-1) addition

        Therefore, Floating point Operation / Global memory access = (2*K+1)/( (A)*(B)*(2*K/32 + 1))


    c. The kernel described in this chapter with shared memory tiling applied
        using a tile size of 32 x 32 and thread coarsening applied using a
        coarsening factor of 4.
    Ans. In thread coarsening, a thread does more work, reducing total number of thread 
         Here we are loading A once and calculating for 4 tiles of B
         which decreases access of B by 4 ( or increase reuse of A )
        Therefore, Floating point Operation / Global memory access = (2*K+1)/( (A)*(B)*(K/32 + K/(4*32) + 1))
*/



