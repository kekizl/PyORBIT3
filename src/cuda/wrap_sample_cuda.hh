// wrap_sample_cuda.hh

#ifndef WRAP_SAMPLE_CUDA_HH
#define WRAP_SAMPLE_CUDA_HH

#include <Python.h>

namespace wrap_sample_cuda {
    PyObject* cuda_kernel_wrapper(PyObject* self, PyObject* args);
}

#endif // WRAP_SAMPLE_CUDA_HH

