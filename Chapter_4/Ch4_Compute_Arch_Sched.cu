
/*  Programming Massively Parallel Processors 4th Edition Solution
    Chapter 4:
*/
//------------------------------------------------------------------------------------------------------------------------------------------------------------------
/*
4.1 Consider the following CUDA kernel and the corresponding host function that
calls it:
*/
__global__ void foo_kernel(int* a, int * b){
    unsigned int i= blockIdx.x*blockDim.x+threadIdx.x;

    if(threadIdx.x < 40 || threadIdx.x >= 104){
        b[i]=a[i]+1;
    }
    if(i%2==0){
        a[i]= b[i]*2;
    }
    for(unsigned int j=0; j< 5-(i%3) ; ++j){
        b[i] += j;
    }
}
void foo(int *a_d, int *b_d){
    unsigned int N=1024;
    foo_kernel <<< ( (N+128-1)/128 ) , 128 >>> (a_d,b_d);
}

/*
a. What is the number of warps per block?
Ans:    Number of threads per Block=128
        with warp size of 32 threads
        we can have maximum of 4 warps per block

b. What is the number of warps in the grid?
Ans:    N=1024, which makes Number of blocks= 8
        Total Threads   = 8*128
        Total Warps     = (8*128)/32 = 32 warps

c. For the statement on line 04:
    i. How many warps in the grid are active?
    Ans. Total 1024 threads are launched
         out of which [0,39] and [104,1024] are active
         Warp [0,31] and [128,159] [160,191] ... [992,1023]
         29 Warps are active
    ii. How many warps in the grid are divergent?
    Ans. Warp [32,63] and [96, 127] are divergent
         [64,95] is inactive
    iii. What is the SIMD efficiency (in %) of warp 0 of block 0?
    Ans. All threads are executing -> 100% efficiency    
    iv. What is the SIMD efficiency (in %) of warp 1 of block 0?
    Ans. [32,39] are active out of 32 threads.
         Efficiency = 8/32 = 25%
    v. What is the SIMD efficiency (in %) of warp 3 of block 0?
    Ans. [104,127] are active, i.e 24 threads
         Efficiency = 24/32 = 75%

d. For the statement on line 07:
    i. How many warps in the grid are active?
    Ans- 32 Warps are active
    ii. How many warps in the grid are divergent?
    Ans- 32 warps are divergent as only even id threads execute
    iii. What is the SIMD efficiency (in %) of warp 0 of block 0?
    Ans- 16 out of 32 executes. Hence 50%

e. For the loop on line 09:
    i. How many iterations have no divergence?
    Ans- (j < 5 - (i%3)) will have values 5, 4, 3
        Threads will go together till 3 
        So 3 iterations have no divergence
    ii. How many iterations have divergence?
        Threads which go till 4 and 5 will cause divergence



4.2. For a vector addition, assume that the vector length is 2000, each thread
calculates one output element, and the thread block size is 512 threads. How
many threads will be in the grid?
    Ans. Block size = 512
        For 2000 element, ceil(2000/512) = 4 blocks will be active
        Therefore 4*512 =2048 Threads will be active 
4.3 For the previous question, how many warps do you expect to have divergence
due to the boundary check on vector length?
    Ans.Warps [0,31][32,63]...[1952,1983] will have no divergence
        Warp [1984, 2015] will have divergence
        Warp [2016, 2047] will be inactive


4.4. Consider a hypothetical block with 8 threads executing a section of code
before reaching a barrier. The threads require the following amount of time
(in microseconds) to execute the sections: 2.0, 2.3, 3.0, 2.8, 2.4, 1.9, 2.6, and
2.9; they spend the rest of their time waiting for the barrier. What percentage
of the threads’ total execution time is spent waiting for the barrier?
    Ans. Threads march together has to wait till 3 ms ( maximum of all )
        th 0 :  2.0 -> 3.0 = 1.0 ms
        th 1 :  2.3 -> 3.0 = 0.7 ms
        th 2 :  3.0 -> 3.0 = 0.0 ms         
        th 3 :  2.8 -> 3.0 = 0.2 ms          
        th 4 :  2.4 -> 3.0 = 0.6 ms  
        th 5 :  1.9 -> 3.0 = 1.1 ms 
        th 6 :  2.6 -> 3.0 = 0.4 ms 
        th 7 :  2.9 -> 3.0 = 0.1 ms 
    Total wait time     : 4.1 ms
    Total Execution Time: 19.9 ms

4.5 A CUDA programmer says that if they launch a kernel with only 32 threads
in each block, they can leave out the __syncthreads() instruction wherever
barrier synchronization is needed. Do you think this is a good idea? Explain

Ans. No
    -If threads in the block read and write shared memory, there is no 
    guarantee that one thread's update is visible to another without __syncthreads()
    -Even though current GPUs may execute a single warp in sync, future GPUs 
    or different execution models may not.

4.6 If a CUDA device’s SM can take up to 1536 threads and up to 4 thread
blocks, which of the following block configurations would result in the most
number of threads in the SM?
    a. 128 threads per block
    Ans. 1536/128  =  12 blocks. But Max Possible Block= 4
         i.e 4*128 =  512 threads.
    b. 256 threads per block
         1536/256  =  6 blocks. But Max Possible Block= 4
         i.e 4*256 =  1024 threads.
    c. 512 threads per block
         1536/512  =  3 blocks. But Max Possible Block= 4
         i.e 4*512 =  2048 threads. ( not possible )
             3*512 =  1536 threads.
    d. 1024 threads per block
         1536/1024 =  2 blocks
         but 2*1024=  2048 threads ( not possible )
         Hence only 1 block will run in 1 SM
    Option C is most efficient with 1536 threads/SM

4.7. Assume a device that allows up to 64 blocks per SM and 2048 threads per
SM. Indicate which of the following assignments per SM are possible. In the
cases in which it is possible, indicate the occupancy level.
    a. 8 blocks with 128 threads each
        8*128 = 1024 threads ( possible )
        Occupancy Level = 1024/2048 = 50%
    b. 16 blocks with 64 threads each
        16*64 = 1024 threads ( possible )
        Occupancy Level = 1024/2048 = 50%
    c. 32 blocks with 32 threads each
        32*32 = 1024 threads ( possible )
        Occupancy Level = 1024/2048 = 50%
    d. 64 blocks with 32 threads each
        64*32 = 2048 threads ( possible )
        Occupancy Level = 2048/2048 = 100%
    e. 32 blocks with 64 threads each    
        32*64 = 2048 threads ( possible )
        Occupancy Level = 2048/2048 = 100%

4.8. Consider a GPU with the following hardware limits: 2048 threads per SM, 32
blocks per SM, and 64K (65,536) registers per SM. For each of the following
kernel characteristics, specify whether the kernel can achieve full occupancy.
If not, specify the limiting factor.
    a. The kernel uses 128 threads per block and 30 registers per thread.
        128 Threads for 32 blocks = 4096 threads > 2048 threads
        2048 / 128 = 16 blocks per SM
        Also 16*128*30 reg = 61440 reg
        Occupancy = 61440/65536 = 93.75 % 
    b. The kernel uses 32 threads per block and 29 registers per thread.
        2048 / 32 = 64 blocks > max 32 block per SM
        Thefore 32*32 = 1024 threads are being used per SM
        32*32*29 = 29696 reg
        Occupancy = 29696/65536 = 45.31%
    c. The kernel uses 256 threads per block and 34 registers per thread.  
        2048/256 = 8 block per SM
        8*256*34 = 69632
        We cannot use more registers than available 
        Therefore 65536/(34*256)= 7 block per SM 
        Occupancy (7*256)/2048 = 87.5%      


4.9. A student mentions that they were able to multiply two 1024 x 1024 matrices
using a matrix multiplication kernel with 32 x 32 thread blocks. The student is
using a CUDA device that allows up to 512 threads per block and up to 8 blocks
per SM. The student further mentions that each thread in a thread block calculates
one element of the result matrix. What would be your reaction and why?

    A thread block of 32 x 32 = 1024 threads exceeds the GPU's limit of 512 threads per block.
*/
//------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
