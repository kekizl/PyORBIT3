#include <cuda_runtime.h>
#include "MatrixVec_kernel.cuh"

__global__ void matrixVectorMul(float *deviceMatrix, float *deviceVector, float *deviceResult, int matrixRows, int matrixCols, int vectorSize) {
    int idx = threadIdx.x;
    float sum = 0.0f;
    for (int i = 0; i < matrixCols; ++i) {
        sum += deviceMatrix[idx * matrixCols + i] * deviceVector[i];
    }
    deviceResult[idx] = sum;
}

void MatrixVecRun(float *deviceMatrix, float *deviceVector, float *deviceResult, int matrixRows, int matrixCols, int vectorSize) {
    matrixVectorMul<<<1, vectorSize>>>(deviceMatrix, deviceVector, deviceResult, matrixRows, matrixCols, vectorSize);
    cudaDeviceSynchronize();
}

void MatrixVecCopy(float *deviceResult, float *hostResult, int vectorSize) {
    cudaMemcpy(hostResult, deviceResult, vectorSize * sizeof(float), cudaMemcpyDeviceToHost);
}

void MatrixVecInit(float **deviceMatrix, float **deviceVector, float **deviceResult, int matrixRows, int matrixCols, int vectorSize) {
    cudaMalloc((void **)deviceMatrix, matrixRows * matrixCols * sizeof(float));
    cudaMalloc((void **)deviceVector, vectorSize * sizeof(float));
    cudaMalloc((void **)deviceResult, vectorSize * sizeof(float));

    // Copy data from host to device
    float hostMatrix[matrixRows * matrixCols] = {1.0f, 2.0f, 3.0f, 4.0f};
    float hostVector[vectorSize] = {5.0f, 6.0f};
    cudaMemcpy(*deviceMatrix, hostMatrix, matrixRows * matrixCols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(*deviceVector, hostVector, vectorSize * sizeof(float), cudaMemcpyHostToDevice);
}

void MatrixVecFree(float *deviceMatrix, float *deviceVector, float *deviceResult) {
    cudaFree(deviceMatrix);
    cudaFree(deviceVector);
    cudaFree(deviceResult);
}

