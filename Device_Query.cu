/* To install 
    Cuda:       sudo apt install nvidia-cuda-toolkit
    Driver:     sudo apt install --reinstall nvidia-driver-535
                sudo reboot
    To Compile: nvcc -arch=sm_50 file.cu -o out             - for gtx 940m
*/
#include<iostream>

int main(){
    
    //Device Querying 
    int dev_count;
    cudaDeviceProp dev_prop;

    cudaError_t err = cudaGetDeviceCount(&dev_count);     // Get number of cuda enabled devices
    
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        return 0;
    }

    printf("Number of Cuda enabled devices: %d \n", dev_count);
    for(int i=0; i< dev_count; i++){
        cudaGetDeviceProperties(&dev_prop, i);
        std::cout<<"Device: "<<i<<" "<<dev_prop.name<<std::endl;
        printf("Maximum Threads Per Block: %d\n", dev_prop.maxThreadsPerBlock);
        printf("Number of SM: %d\n", dev_prop.multiProcessorCount);
        printf("Warp Size: %d\n", dev_prop.warpSize);
        printf("Maximum Threads in a Block (x-dir): %d\n", dev_prop.maxThreadsDim[0]);
        printf("Maximum Threads in a Block (y-dir): %d\n", dev_prop.maxThreadsDim[1]);
        printf("Maximum Threads in a Block (z-dir): %d\n", dev_prop.maxThreadsDim[2]);
        printf("Maximum Blocks in a Grid (x-dir): %d\n", dev_prop.maxGridSize[0]);
        printf("Maximum Blocks in a Grid (y-dir): %d\n", dev_prop.maxGridSize[1]);
        printf("Maximum Blocks in a Grid (z-dir): %d\n", dev_prop.maxGridSize[2]);

    }
    return 0;
}