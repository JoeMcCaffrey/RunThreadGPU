#include <stdio.h>
#include <cuda.h>

/*

This program moves data from main memory to the GPU memory and prints
using threads by exploiting the CPU's synconization
*/

// Compile: nvcc Hello.cu -o Hello



// this is the global gpu function that will run on the threads
__global__ void hello(float *a, float *r) {
  int thread=blockIdx.x * blockDim.x + threadIdx.x;

  r[thread]=a[thread];

  printf("TID=%d:  Hello World! a=%f\n", thread, a[thread]);
}

main() {
// we have some data stored in cpu memory, adata, bdata,rdata
// and two defined variables that will live in gpu data
   float aData[32], bData[32], rData[32], *aGPU, *rGPU;
   int index;

// we init the array that is in cpu memory
   for(index=0;index<32;index++) {
     aData[index]=5*index;
   }
// we init the gpu memory of some size
   cudaMalloc((void **)&aGPU, sizeof(float)*32);
   cudaMalloc((void **)&rGPU, sizeof(float)*32);
// copy the memory from cpu land to gpu land so we can run it on the gpu
   cudaMemcpy(aGPU, aData, sizeof(float)*32, cudaMemcpyHostToDevice);
// this is the call to threads to run on the gpu, see we have defined the agpu and rgpu
// we want to call the cuda device sycronize to wait for the threads to execute
   hello<<<1, 32>>>(aGPU, rGPU);
// the above have to equal the size of the array (multiplication) 

   cudaDeviceSynchronize();
// we want to copy the results back from gpu land to cpu land
   cudaMemcpy(rData, rGPU, sizeof(float)*32, cudaMemcpyDeviceToHost);
   int i ;
// print the results 
   for(i = 0; i < 32 ; i++){
    printf("result: r[%d]=%f\n",i, rData[i]);
   }
}


