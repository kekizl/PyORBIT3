#include "batch_kernel.cuh"
#include <cuda_runtime.h>
__global__ void matrixVectorMultiplyBatch(double *vectors, double *matrix, double *results, int numVectors, int vectorSize) {

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
void runMatrixVectorBatchKernel(double *h_vectors, double *h_results, double *d_batch, double *d_matrix, double *d_results, int batchSize, int vectorSize, int startIdx){
  
   cudaStream_t stream1, stream2;
   cudaStreamCreate(&stream1);
   cudaStreamCreate(&stream2);
   
   if (startIdx == 0){
	   cudaMemcpy(d_batch, h_vectors + startIdx, sizeof(double) * batchSize * vectorSize, cudaMemcpyHostToDevice);
   }
    int blockSize = 256;
    int numBlocks = (batchSize + blockSize - 1) / blockSize;

    // Execute kernel on batch
    matrixVectorMultiplyBatch<<<numBlocks, blockSize>>>(d_batch, d_matrix, d_results, batchSize, vectorSize);
    int nextBatchStartIdx = (startIdx + batchSize * vectorSize);
    cudaMemcpyAsync(d_batch, h_vectors + nextBatchStartIdx, sizeof(double) * batchSize * vectorSize, cudaMemcpyHostToDevice, stream1);
 
    // Copy back the results of the current batch into h_results
    cudaMemcpyAsync(h_results + startIdx, d_results, sizeof(double) * batchSize * vectorSize, cudaMemcpyDeviceToHost, stream2);

    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);
    // Destroy the CUDA streams
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
}

void cudaAllocateMemoryBatch(double **d_batch, double **d_matrix, double **d_results, int numVectors, int vectorSize, int matrixSize) {
    cudaMalloc(d_batch, sizeof(double) * numVectors * vectorSize);
    cudaMalloc(d_matrix, sizeof(double) * matrixSize * matrixSize);
    cudaMalloc(d_results, sizeof(double) * numVectors * vectorSize);
}

void cudaSetMatrix(double *hostMatrix, double *deviceMatrix, int matrixSize) {
      cudaMemcpy(deviceMatrix, hostMatrix, sizeof(double) * matrixSize * matrixSize, cudaMemcpyHostToDevice);
}

void cudaFreeBatch(double *d_matrix, double *d_batch, double *d_results) {
    cudaFree(d_matrix);
    cudaFree(d_batch);
    cudaFree(d_results);
}

