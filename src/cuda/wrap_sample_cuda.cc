// wrap_sample_cuda.cc

#include "Python.h"
#include "wrap_sample_cuda.hh"
#include "sample_kernel.cuh"

namespace wrap_sample_cuda {

    PyObject* cuda_kernel_wrapper(PyObject* self, PyObject* args) {
        PyObject* py_array;
        int size;

        // Parse arguments
        if (!PyArg_ParseTuple(args, "Oi", &py_array, &size)) {
            return NULL;
        }

        // Convert PyObject to int* array
        int* array;
        Py_buffer buf;
        if (PyObject_GetBuffer(py_array, &buf, PyBUF_SIMPLE) == -1) {
            PyErr_SetString(PyExc_TypeError, "Expected a buffer object");
            return NULL;
        }
        array = (int*)buf.buf;

        // Launch CUDA kernel
        cuda_kernel<<<(size + 255) / 256, 256>>>(array, size);
        cudaDeviceSynchronize(); // Ensure kernel execution is complete

        // Release the buffer
        PyBuffer_Release(&buf);

        // Return None
        Py_INCREF(Py_None);
        return Py_None;
    }

    // Method definitions
    static PyMethodDef sample_cudaMethods[] = {
        {"cuda_kernel", cuda_kernel_wrapper, METH_VARARGS, "Execute a simple CUDA kernel"},
        {NULL, NULL, 0, NULL}
    };

    // Module definition
    static struct PyModuleDef sample_cudaModule = {
        PyModuleDef_HEAD_INIT,
        "sample_cuda",
        "Python wrappers for sample CUDA functions",
        -1,
        sample_cudaMethods
    };

    // Module initialization function
    PyMODINIT_FUNC PyInit_sample_cuda(void) {
        return PyModule_Create(&sample_cudaModule);
    }
}

