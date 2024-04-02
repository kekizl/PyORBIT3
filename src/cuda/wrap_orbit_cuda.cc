#include "Python.h"
#include "wrap_orbit_cuda.hh"
#include "sample_kernel.cuh"

namespace wrap_orbit_cuda {

    PyObject* runKernel_wrapper(PyObject* self, PyObject* args) {
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

        // Run the kernel
        runKernel(array, size);

        // Release the buffer
        PyBuffer_Release(&buf);

        // Return None
        Py_INCREF(Py_None);
        return Py_None;
    }

    PyObject* allocateMemory_wrapper(PyObject* self, PyObject* args) {
        int size;

        // Parse arguments
        if (!PyArg_ParseTuple(args, "i", &size)) {
            return NULL;
        }

        // Allocate memory on the device
        int* array_device;
        allocateMemory(&array_device, size);

        // Return the device memory pointer
        return PyLong_FromVoidPtr((void*)array_device);
    }

    PyObject* freeMemory_wrapper(PyObject* self, PyObject* args) {
        long device_pointer;

        // Parse arguments
        if (!PyArg_ParseTuple(args, "l", &device_pointer)) {
            return NULL;
        }

        // Free memory on the device
        freeMemory((int*)device_pointer);

        // Return None
        Py_INCREF(Py_None);
        return Py_None;
    }

    PyObject* copyResultToDevice_wrapper(PyObject* self, PyObject* args) {
        PyObject* py_array_device;
        PyObject* py_array_host;
        int size;

        // Parse arguments
        if (!PyArg_ParseTuple(args, "OOi", &py_array_device, &py_array_host, &size)) {
            return NULL;
        }

        // Convert PyObject to int* arrays
        int* array_device;
        Py_buffer buf_device;
        if (PyObject_GetBuffer(py_array_device, &buf_device, PyBUF_SIMPLE) == -1) {
            PyErr_SetString(PyExc_TypeError, "Expected a buffer object for device array");
            return NULL;
        }
        array_device = (int*)buf_device.buf;

        int* array_host;
        Py_buffer buf_host;
        if (PyObject_GetBuffer(py_array_host, &buf_host, PyBUF_SIMPLE) == -1) {
            PyErr_SetString(PyExc_TypeError, "Expected a buffer object for host array");
            return NULL;
        }
        array_host = (int*)buf_host.buf;

        // Copy result from host to device
        copyResultToDevice(array_device, array_host, size);

        // Release the buffers
        PyBuffer_Release(&buf_device);
        PyBuffer_Release(&buf_host);

        // Return None
        Py_INCREF(Py_None);
        return Py_None;
    }

    // Method definitions
    static PyMethodDef sample_cudaMethods[] = {
        {"runKernel", runKernel_wrapper, METH_VARARGS, "Run the CUDA kernel"},
        {"allocateMemory", allocateMemory_wrapper, METH_VARARGS, "Allocate memory on the device"},
        {"freeMemory", freeMemory_wrapper, METH_VARARGS, "Free memory on the device"},
        {"copyResultToDevice", copyResultToDevice_wrapper, METH_VARARGS, "Copy result from host to device"},
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

