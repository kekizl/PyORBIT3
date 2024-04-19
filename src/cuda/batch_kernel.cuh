void runMatrixVectorBatchKernel(double *h_vectors, double *h_results, double *d_batch, double *d_matrix, double *d_results, int batchSize, int vectorSize, int startIdx);

void cudaAllocateMemoryBatch(double **d_batch, double **d_matrix, double **d_results, int numVectors, int vectorSize, int matrixSize);

void cudaSetMatrix(double *hostMatrix, double *deviceMatrix, int matrixSize);

void cudaFreeBatch(double *d_matrix, double *d_batch, double *d_results);

