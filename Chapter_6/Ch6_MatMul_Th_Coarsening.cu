
/*  Programming Massively Parallel Processors 4th Edition Solution
    Chapter 6: Performance Considerations

    Example Code: Thread Coaresening

*/


#include<cuda.h>
#include<iostream>
#include<stdlib.h>
#include<random>
#include<iomanip>
#include<math.h>
using namespace std;

#define width 32
#define TILE_WIDTH 8
#define COARSE_FACTOR 2

__global__ void mat_gpu(int *M, int *N, int *P){
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    int row = blockIdx.y*TILE_WIDTH + threadIdx.y;
    int colStart = blockIdx.x*TILE_WIDTH*COARSE_FACTOR + threadIdx.x;

    float Pvalue[COARSE_FACTOR];
    for(int c=0; c<COARSE_FACTOR; c++){
        Pvalue[c] = 0.0f;
    }

    for(int ph=0; ph < width/TILE_WIDTH; ph++){
        Mds[threadIdx.y][threadIdx.x] = M[row*width + ph*TILE_WIDTH + threadIdx.x];
        for(int c=0; c < COARSE_FACTOR; c++){
            int col= colStart + c*TILE_WIDTH;
            Nds[threadIdx.y][threadIdx.x] = N[(ph*TILE_WIDTH+threadIdx.y)*width + col]; 
            __syncthreads();

            for(int k=0; k<TILE_WIDTH; k++){
                Pvalue[c] += Mds[threadIdx.y][k]*Nds[k][threadIdx.x];
            }
            __syncthreads();
        }
    }
    for(int c=0; c<COARSE_FACTOR; c++){
        int col=colStart+c*TILE_WIDTH;
        P[row*width+col]=Pvalue[c];
    }
}




void print_arrays(int *A, int N, int M)
{
    for(int i=0; i<N; i++){
       for(int j=0; j<M; j++){
            cout<<setprecision(3)<<A[i*M+j]<<" ";
       } 
       cout<<endl;
    }
}

int matMul_verify(int *A, int *B, int *C){
    int ans[width*width]={0};
    for(int i=0; i<width; i++){
        for(int j=0; j<width; j++){
            for(int k=0; k<width; k++){
                ans[i*width+j]+=A[i*width+k]*B[k*width+j];
            }
        }
    }
    cout<<endl<<"Expected Output: "<<endl;
    print_arrays(ans,width,width);

    cout<<endl<<"Kernel Output: "<<endl;
    print_arrays(C,width,width);
    for(int i=0;i<width*width;i++){
        if(ans[i]!=C[i]){
            return 1;
        }
    }
    return 0;
}

int main()
{
    int *a,*b,*c;
    int *gpu_a, *gpu_b, *gpu_c;

    a =(int*)malloc(width*width*sizeof(int));
    b =(int*)malloc(width*width*sizeof(int));
    c =(int*)malloc(width*width*sizeof(int));
        
    cudaMalloc(&gpu_a, width*width*sizeof(int));
    cudaMalloc(&gpu_b, width*width*sizeof(int));
    cudaMalloc(&gpu_c, width*width*sizeof(int));
    
    for(int i=0;i<width*width;i++){
        a[i]=i%10; //rand()%10;
        b[i]=i%10; //rand()%10;
        c[i]=-1;
    }
    cudaMemcpy(gpu_a, a, width*width*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_b, b, width*width*sizeof(int), cudaMemcpyHostToDevice);

    int gridSize= (int)ceil(width/TILE_WIDTH);
    int THREADS = TILE_WIDTH;

    dim3 threads(THREADS, THREADS, 1);
    //Due to coarse-graining, number of blocks launched is reduced
    //as we are doing more work with threads.
    dim3 grid(gridSize/COARSE_FACTOR, gridSize, 1);
    
    
    mat_gpu<<<grid,threads>>>(gpu_a, gpu_b, gpu_c);

    cudaMemcpy(c, gpu_c, width*width*sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    if(matMul_verify(a,b,c)==1){
        printf("\nWrong Calculation\n");
    }
    else{
        printf("\nCorrect\n");
    }
    free(a);
    free(b);
    free(c);
    cudaFree(gpu_a);
    cudaFree(gpu_b);
    cudaFree(gpu_c);

    return 0;
}
