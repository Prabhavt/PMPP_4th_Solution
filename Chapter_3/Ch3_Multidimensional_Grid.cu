/*  Programming Massively Parallel Processors 4th Edition Solution
    Chapter 3:
*/
#include<cuda.h>
#include<stdlib.h>
#include<stdio.h>

#define N 600           //Total number of elements
#define THREADS 32      //Number of Threads in each block

//------------------------------Matrix Multiplication--------------------------------------------------------------------------
//  One Thread to calculate one P element 
__global__ void matMul_ex_k( float *A, float *B, float *C){
    unsigned row = blockIdx.y*blockDim.y + threadIdx.y;
    unsigned col = blockIdx.x*blockDim.x + threadIdx.x;
    
    if ( ( row < N ) && ( col < N )){
        float val=0;
        for( unsigned i=0; i< N; i++){
            val+= A[ row*N + i ] * B[ i*N + col]; 
        }
        C[ row*N + col ] = val; 
    } 

}
//------------------------------Exercise 3.1--------------------------------------------------------------------------
/*
In this chapter we implemented a matrix multiplication kernel that has each
thread produce one output matrix element. In this question, you will
implement different matrix-matrix multiplication kernels and compare them.
a. Write a kernel that has each thread produce one output matrix row. Fill in
the execution configuration parameters for the design.
b. Write a kernel that has each thread produce one output matrix column. Fill
in the execution configuration parameters for the design.
c. Analyze the pros and cons of each of the two kernel designs.
*/
//a.
__global__ void matMul_1a_k( float *A, float *B, float *C){
    unsigned th = blockIdx.x*blockDim.x + threadIdx.x;
    
    if ( th < N ){                         
        for( unsigned col=0; col< N; col++ ){
            float val=0;
            for( unsigned i=0; i< N; i++){
                val+= A[ th*N + i ] * B[ i*N + col ]; 
            }
            C[ th*N + col ] = val;
        } 
    } 

}
//b. To compute one complete colomn -> C[ row*N + th ] = val
__global__ void matMul_1b_k( float *A, float *B, float *C){
    unsigned th = blockIdx.x*blockDim.x + threadIdx.x;
    
    if ( th < N ){                        
        for( unsigned row=0; row< N; row++ ){
            float val=0;
            for( unsigned i=0; i< N; i++){
                val+= A[ row*N + i ] * B[ i*N + th ]; 
            }
            C[ row*N + th ] = val;
        } 
    } 

}
//-------------------------------Exercise 3.2----------------------------------------------------------------------------------------------------
/* A matrix-vector multiplication takes an input matrix B and a vector C and
produces one output vector A. Each element of the output vectorPA is the dot
product of one row of the input matrix B and C.
For simplicity we will handle only square matrices whose elements are single-
precision floating-point numbers. Write a matrix-vector multiplication kernel and
the host stub function that can be called with four parameters: pointer to the output
matrix, pointer to the input matrix, pointer to the input vector, and the number of
elements in each dimension. Use one thread to calculate an output vector element. 
*/

__global__ void matMultvec1_k(float *A, float *B, float *C, int n){
    unsigned th= blockIdx.x * blockDim.x + threadIdx.x;
    if(th < n){
        float val= 0.0f;
        for(int i=0; i<n; i++){
            val += (float) B[th*n + i]*C[i];
        }
        A[th]= val;
    }
}
//The solution below uses atomicAdd to avoid Race Around condition 
__global__ void matMultvec2_k(float *A, float *B, float *C, int n){
    unsigned th= blockIdx.x * blockDim.x + threadIdx.x;
    if(th < n){
        for(int i=0; i<n; i++){
            atomicAdd(&A[i], B[i*n + th] * C[th]);
        }
    }
} 
//------------------------------Exercise 3.3--------------------------------------------------------------------------------------------------------
/*
a. What is the number of threads per block
Ans.    dim3 bd(16,32) -> 16*32 = 512 threads per block
b. What is the number of threads in the grid
Ans.    dim3 gd((N-1)/16+1, (M-1)/32+1)   Where M=150, N=300
             gd(19,5);    
        Number of threads= 19*5*512= 48640
c. What is the number of blocks in the grid?
Ans.    gd(19,5) = 95    
d. What is the number of threads that execute the code on line 05?
Ans.    150*300 = 45000

//------------------------------Exercise 3.4--------------------------------------------------------------------------------------------------------
4. Consider a 2D matrix with a width of 400 and a height of 500. The matrix is
   stored as a one-dimensional array. Specify the array index of the matrix
   element at row 20 and column 10:
a. If the matrix is stored in row-major order.
Ans.  Matrix [ 400 * 20 + 10 ]
b. If the matrix is stored in column-major order.
Ans.  Matrix [ 20 + 10 * 500 ]

//------------------------------Exercise 3.5--------------------------------------------------------------------------------------------------------
5. Consider a 3D tensor with a width of 400, a height of 500, and a depth of
300. The tensor is stored as a one-dimensional array in row-major order.
Specify the array index of the tensor element at x = 10, y = 20, and z = 5.

Ans.  Matrix [ z*W*H + y*W + x ]
      Matrix [ 5*400*500 + 20*400 + 10 ]


*/

//---------------------------------------------Verify Output-------------------------------------------------------------------------------------
int matMul_verify(float *A, float *B, float *C){
    float ans[N*N]={0};
    for(int i=0; i<N; i++){
        for(int j=0; j<N; j++){
            for(int k=0; k<N; k++){
                ans[i*N+j]+=A[i*N+k]*B[k*N+j];
            }
        }
    }
    for(int i=0;i<N*N;i++){
        if(ans[i]!=C[i]){
            printf("\nResult: Wrong Output\n");
            return 1;
        }
    }
    printf("\nResult: Correct Output\n");
    return 0;
}

int matMulvec_verify( float *A, float *B, float *C){
    float ans[N]={0};
    printf("\nCorrect Output\n");
    for(int i=0; i<N; i++){
        for(int j=0; j<N; j++){
                ans[i]+=B[i*N+j]*C[j];
        }
        printf("%2.3f,", ans[i]);
    }
    printf("\n");
    for(int i=0;i<N;i++){
        if( std::abs(ans[i]-A[i])> 2){
            printf("\nResult:  Wrong Output\n %f %f",ans[i],A[i]);
            return 1;
        }
    }
    printf("\nResult: Correct Output\n");
    return 0;
}

void print_Matrix(float *mat, int n, int m){
    //n-number of nows, m-number of colomns
    for(int i=0; i< n*m; i++){
        if(i%m==0){
            printf("\n");
        }
        printf("%2.3f,", mat[i]);
    }
    printf("\n");
}
void print_Vec(float *vec, int n){
    for(int i=0; i<n; i++){
        printf("%2.3f,", vec[i]);
    }
}

//------------------------------------------------MAIN--------------------------------------------------------------------------------------
int main(){
    float *A_h, *B_h, *C_h; float *vecA_h, *vecC_h;
    float *A_d, *B_d, *C_d; float *vecA_d, *vecC_d;
    A_h = (float*) malloc(N*N*sizeof(float));
    B_h = (float*) malloc(N*N*sizeof(float));
    C_h = (float*) malloc(N*N*sizeof(float));
    vecA_h=(float*) malloc(N*sizeof(float));
    vecC_h=(float*) malloc(N*sizeof(float));
    
    cudaMalloc(&A_d, N*N*sizeof(float));
    cudaMalloc(&B_d, N*N*sizeof(float));
    cudaMalloc(&C_d, N*N*sizeof(float));
    cudaMalloc(&vecA_d, N*sizeof(float));
    cudaMalloc(&vecC_d, N*sizeof(float));

    for( int i=0; i<N*N; i++){
        A_h[i]=(float)(rand()) / (float)(rand()) ;
        B_h[i]=(float)(rand()) / (float)(rand()) ;
        if(i<N){
            vecC_h[i]=(float)(rand()) / (float)(rand()) ;
        }
    }

    //cudaMemcpy( A_d, A_h, N*N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy( B_d, B_h, N*N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy( vecC_d, vecC_h, N*sizeof(float), cudaMemcpyHostToDevice);
    
    //dim3 numBlocks(1,1,1);
    //dim3 numThreads(N,N,1);
    //dim3 numOneDim( (float)(( N*N + THREADS - 1)/THREADS), 1, 1);
    //matMul_ex_k << numBlocks, numThreads >>> (A_d, B_d, C_d );
    //matMul_1a_k <<< 1, N >>> ( A_d, B_d, C_d );
    //matMul_1b_k <<< numOneDim , THREADS >>> ( A_d, B_d, C_d );
    matMultvec2_k <<< (N+THREADS-1)/THREADS , THREADS >>> (vecA_d, B_d, vecC_d, N); 
    cudaDeviceSynchronize();
    
    cudaMemcpy( vecA_h, vecA_d, N*sizeof(float), cudaMemcpyDeviceToHost);
    //cudaMemcpy( C_h, C_d, N*N*sizeof(float), cudaMemcpyDeviceToHost);
    
    //printf("Input Matrix %dx%d",N,N);
    //print_Matrix(B_h,N,N);
    
    //printf("\nVector \n");
    //print_Vec(vecC_h,N);

    
    //printf("\n\nGenerated Output\n");
    //print_Vec(vecA_h, N);
    // print_Matrix(C_h,N,N);


    // matMul_verify(A_h,B_h,C_h);
    matMulvec_verify(vecA_h, B_h, vecC_h);
    
    free(A_h);
    free(B_h);
    free(C_h);
    free(vecA_h);
    free(vecC_h);
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
    cudaFree(vecA_d);
    cudaFree(vecC_d);
    return 0;
}