void runMatrixVectorMultiplyKernel(double *deviceVectors, double *deviceMatrix, double *deviceResults, int numVectors, int vectorSize, int matrixSize);

void cudaAllocateMemory(double **d_vectors, double **d_matrix, double **d_results, int numVectors, int vectorSize, int matrixSize);

void cudaSet(double *hostVectors, double *hostMatrix, double *deviceVectors, double *deviceMatrix, int numVectors, int vectorSize, int matrixSize);

void cudaCopyDeviceToHost(double *deviceVectors, double *deviceResults, double *hostResults, int numVectors, int vectorSize);

void cudaFreeAll(double *d_matrix, double *d_vectors, double *d_results);

