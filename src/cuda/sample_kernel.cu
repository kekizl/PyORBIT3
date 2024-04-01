// sample_kernel.cu

#include <cuda_runtime.h>
#include <sample_kernel.cuh>
#ifndef SAMPLE_KERNEL_CUH
#define SAMPLE_KERNEL_CUH

__global__ void cuda_kernel(int* array, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        array[idx] *= 2; // Example operation: double each element of the array
    }
}

#endif // SAMPLE_KERNEL_CUH

