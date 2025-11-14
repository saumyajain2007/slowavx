#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "slow_simd.h"

static PyObject* py_slow_vector(PyObject* self, PyObject* args, PyObject* kwargs) {
    Py_buffer dst_buf, src_buf, idx_buf;
    size_t n;
    int slow_factor = 1;
    static char *kwlist[] = {"dst", "src", "idx", "n", "slow_factor", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "y*y*y#n|i", kwlist,
                                     &dst_buf, &src_buf, &idx_buf, &n, &slow_factor)) {
        return NULL;
    }

    // basic checks
    if (dst_buf.len < (ssize_t)(n * sizeof(float)) || src_buf.len < (ssize_t)(n * sizeof(float))) {
        PyErr_SetString(PyExc_ValueError, "buffers smaller than n * sizeof(float)");
        PyBuffer_Release(&dst_buf);
        PyBuffer_Release(&src_buf);
        PyBuffer_Release(&idx_buf);
        return NULL;
    }

    // choose best implementation at runtime
    if (slow_simd_runtime_has_avx2()) {
        slow_vector_work_avx2((float*)dst_buf.buf, (const float*)src_buf.buf, (const int*)idx_buf.buf, n, slow_factor);
    } else {
        slow_vector_work_scalar((float*)dst_buf.buf, (const float*)src_buf.buf, (const int*)idx_buf.buf, n, slow_factor);
    }

    PyBuffer_Release(&dst_buf);
    PyBuffer_Release(&src_buf);
    PyBuffer_Release(&idx_buf);
    Py_RETURN_NONE;
}

static PyMethodDef SlowMethods[] = {
    {"slow_vector", (PyCFunction)py_slow_vector, METH_VARARGS | METH_KEYWORDS, "Run slow AVX function"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef slowmodule = {
    PyModuleDef_HEAD_INIT,
    "slowavx",
    NULL,
    -1,
    SlowMethods
};

PyMODINIT_FUNC PyInit_slowavx(void) {
    return PyModule_Create(&slowmodule);
}
