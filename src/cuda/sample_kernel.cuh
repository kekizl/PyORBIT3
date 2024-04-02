#ifndef SAMPLE_KERNEL_CUH
#define SAMPLE_KERNEL_CUH

void runKernel(int* array, int size);

void allocateMemory(int** array_device, int size);

void freeMemory(int* array_device);

void copyResultToDevice(int* array_device, int* array_host, int size);

#endif // SAMPLE_KERNEL_CUH

