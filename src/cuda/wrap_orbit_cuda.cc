#include "Python.h"
#include <iostream>

#include "wrap_orbit_cuda.hh"

//wrappers for different kernels
#include "wrap_sample_cuda.hh"

extern "C" {
    // Prototype of the CUDA kernel function
    __global__ void cuda_kernel(int* array, int size);

    // Python wrapper for the CUDA kernel
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

        // Release the buffer
        PyBuffer_Release(&buf);

        // Return None
        Py_INCREF(Py_None);
        return Py_None;
    }

    // Method definitions
    static PyMethodDef orbit_cudaMethods[] = {
        {"cuda_kernel", cuda_kernel_wrapper, METH_VARARGS, "Execute a simple CUDA kernel"},
        {NULL, NULL, 0, NULL}
    };

    // Module definition
    static struct PyModuleDef orbit_cudaModule = {
        PyModuleDef_HEAD_INIT,
        "orbit_cuda",
        "Python wrappers for CUDA functions",
        -1,
        orbit_cudaMethods
    };

    // Module initialization function
    PyMODINIT_FUNC PyInit_orbit_cuda(void) {
        return PyModule_Create(&orbit_cudaModule);
    }
}
