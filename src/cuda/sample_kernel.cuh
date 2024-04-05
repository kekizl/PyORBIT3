#ifndef SAMPLE_KERNEL_CUH
#define SAMPLE_KERNEL_CUH

void sampleRun(int* array, int size);

void sampleAllocate(int** array_device, int size);

void sampleFree(int* array_device);

void sampleCopy(int* array_device, int* array_host, int size);

#endif // SAMPLE_KERNEL_CUH

