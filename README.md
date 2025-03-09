# Programming Massively Parallel Processors 4th Edition- Exercise Solutions 

This repository contains my solutions to the exercises from the book **"Programming Massively Parallel Processors"** by **David B. Kirk and Wen-mei W. Hwu**. The book provides a comprehensive introduction to GPU computing using CUDA.  

## About the Book  
*"Programming Massively Parallel Processors"* covers:  
- GPU architecture and parallel programming models  
- CUDA programming fundamentals  
- Memory hierarchy and optimization techniques  
- Parallel algorithms and performance considerations  

## 📁 Repository Structure  
-  Chapter_3  - Completed 

## 🛠 Prerequisites  
Before running the code, ensure you have:  
- **NVIDIA GPU** with CUDA support  
- **CUDA Toolkit** installed ([Download here](https://developer.nvidia.com/cuda-downloads))  
- **Compiler** (e.g., `nvcc`)  

## ▶️ Running the Solutions  
1. Clone this repository:  
   ```bash
   git clone https://github.com/Prabhavt/PMPP_4th_Solution.git

2. Navigate to the desired exercise folder:
   ```bash
   cd PMPP_4th_Solution
   cd Chapter_3/
3. Modify the main function according the kernel you want to launch
   ```bash
   nvcc Ch3_Multidimensional_Grid.cu -o ex1
   ./ex1
