#include "VectorPerThread_kernel.cuh"
__global__ void matrixVectorMultiply(double *vectors, double *matrix, double *results, int numVectors, int vectorSize) {

//perform matrix-vector multiplication
    int vectorIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if (vectorIdx < numVectors) {
        double result[6] = {0.0}; 
        // Perform matrix-vector multiplication
        for (int i = 0; i < vectorSize; ++i) { // Iterate over rows of the matrix
            double dotProduct = 0.0;

            // Compute dot product of matrix row with vector
            for (int j = 0; j < vectorSize; ++j) {
                dotProduct += matrix[i * vectorSize + j] * vectors[vectorIdx * vectorSize + j];
            }

            // Accumulate dot product to result
            result[i] = dotProduct;
        }

        // Store the result back to global memory
        for (int i = 0; i < vectorSize; ++i) {
            results[vectorIdx * vectorSize + i] = result[i];
        }
    }
}

void runMatrixVectorMultiplyKernel(double *deviceVectors, double *deviceMatrix, double *deviceResults, int numVectors, int vectorSize, int matrixSize) {
    // Define thread block and grid dimensions
    int blockSize = 256;
    int numBlocks = (numVectors + blockSize - 1) / blockSize;

    // Launch kernel to perform matrix-vector multiplication
    matrixVectorMultiply<<<numBlocks, blockSize>>>(deviceVectors, deviceMatrix, deviceResults, numVectors, vectorSize);
    cudaDeviceSynchronize();
}

void cudaAllocateMemory(double **d_vectors, double **d_matrix, double **d_results, int numVectors, int vectorSize, int matrixSize) {
    cudaMalloc(d_vectors, sizeof(double) * numVectors * vectorSize);
    cudaMalloc(d_matrix, sizeof(double) * matrixSize * matrixSize);
    cudaMalloc(d_results, sizeof(double) * numVectors * vectorSize);
}

//host to device
void cudaSet(double *hostVectors, double *hostMatrix, double *deviceVectors, double *deviceMatrix, int numVectors, int vectorSize, int matrixSize) {
    cudaMemcpy(deviceVectors, hostVectors, sizeof(double) * numVectors * vectorSize, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceMatrix, hostMatrix, sizeof(double) * matrixSize * matrixSize, cudaMemcpyHostToDevice);
}

//device to host
void cudaCopyDeviceToHost(double *deviceVectors, double *deviceResults, double *hostResults, int numVectors, int vectorSize) {
    cudaMemcpy(hostResults, deviceResults, sizeof(double) * numVectors * vectorSize, cudaMemcpyDeviceToHost);
}

// Function to free device memory
void cudaFreeAll(double *d_matrix, double *d_vectors, double *d_results) {
    cudaFree(d_vectors);
    cudaFree(d_matrix);
    cudaFree(d_results);
}

void runCompleteMatrixVectorMultiplyKernel(double *deviceVectors, double *deviceMatrix, double *deviceResults, int numVectors, int vectorSize, int matrixSize) {

}

