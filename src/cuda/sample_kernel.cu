#include <cuda_runtime.h>
#include "sample_kernel.cuh"

__global__ void cuda_kernel(int* array, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        array[idx] *= 2; // Example operation: double each element of the array
    }
}

void sampleRun(int* array, int size) {
    // Launch the CUDA kernel
    cuda_kernel<<<(size + 255) / 256, 256>>>(array, size);
    cudaDeviceSynchronize();
}

void sampleAllocate(int** array_device, int size) {
    // Allocate memory on the device
    cudaMalloc((void**)array_device, size * sizeof(int));
}

void sampleFree(int* array_device) {
    // Free memory on the device
    cudaFree(array_device);
}

void sampleCopy(int* array_device, int* array_host, int size) {
    // Copy result from host to device
    cudaMemcpy(array_device, array_host, size * sizeof(int), cudaMemcpyHostToDevice);
}
