#ifndef A_KERNEL_CUH
#define A_KERNEL_CUH

void MatrixVecRun(float *deviceMatrix, float *deviceVector, float *deviceResult, int matrixRows, int matrixCols, int vectorSize);

void MatrixVecCopy(float *deviceResult, float *hostResult, int vectorSize);

void MatrixVecInit(float **deviceMatrix, float **deviceVector, float **deviceResult, int matrixRows, int matrixCols, int vectorSize);

void MatrixVecFree(float *deviceMatrix, float *deviceVector, float *deviceResult);

#endif
