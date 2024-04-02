#ifndef WRAP_SAMPLE_CUDA_HH
#define WRAP_SAMPLE_CUDA_HH

#ifdef __cplusplus
extern "C" {
#endif
  namespace wrap_orbit_cuda{

    PyMODINIT_FUNC initorbit_cuda(void);

  }

#ifdef __cplusplus
}
#endif
#endif // WRAP_SAMPLE_CUDA_HH

