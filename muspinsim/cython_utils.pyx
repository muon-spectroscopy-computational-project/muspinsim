import numpy as np

cimport numpy as np
np.import_array()

def calc_time_evolve_basic(times, other_dimension, A, W):
    cdef int num_times = times.shape[0]
    cdef int mat_dim = A.shape[0]

    # Avoid using append as assignment should be faster
    result = np.zeros((num_times, 1), dtype=np.float64)
    for i in range(num_times):
        # k <= j
        for j in range(mat_dim):
            for k in range(0, j + 1):
                result[i, 0] += A[j, k] * np.cos(W[j, k] * times[i])
        result[i, 0] /= other_dimension

    return result

cimport cython

""""
# Use C versions to avoid linking back to python
cdef extern from "math.h":
    double cos(double theta)
    double sin(double theta)

@cython.boundscheck(False) # Disable bounds-checking
@cython.wraparound(False)  # Disable negative index wrapping
def calc_time_evolve(np.ndarray[np.float64_t, ndim=1] times, int other_dimension, np.ndarray[np.float64_t, ndim=2] A, np.ndarray[np.float64_t, ndim=2] W):
    cdef Py_ssize_t num_times = times.shape[0]
    cdef Py_ssize_t mat_dim = A.shape[0]

    # Avoid using append as assignment should be faster
    cdef np.ndarray[np.float64_t, ndim=1] result = np.zeros((num_times), dtype=np.float64)
    cdef Py_ssize_t  i, j, k
    for i in range(num_times):
        # k <= j
        for j in range(mat_dim):
            for k in range(0, j):
                result[i] += A[j, k] * cos(W[j, k] * times[i])
        result[i] /= other_dimension

    return result
"""

# Use C versions to avoid linking back to python
cdef extern from "<math.h>" nogil:
    double cos(double theta)
    double sin(double theta)

from cython.parallel import prange
@cython.boundscheck(False) # Disable bounds-checking
@cython.wraparound(False)  # Disable negative index wrapping
def calc_time_evolve_parallel(double [:] times, int other_dimension, double [:, :] A, double [:, :] W):
    cdef Py_ssize_t num_times = times.shape[0]
    cdef Py_ssize_t mat_dim = A.shape[0]

    # Avoid using append as assignment should be faster
    cdef np.ndarray[np.float64_t, ndim=1] result = np.zeros((num_times), dtype=np.float64)
    cdef double [:] result_view = result
    cdef Py_ssize_t  i, j, k

    for i in prange(num_times, nogil=True):
        # k <= j
        for j in range(mat_dim):
            for k in range(0, j):
                result_view[i] += A[j, k] * cos(W[j, k] * times[i])
        result_view[i] /= other_dimension

    return result