import numpy as np
from cython.parallel import prange

cimport numpy as np
cimport cython

np.import_array()

# Use C versions to avoid linking back to python
cdef extern from "<math.h>" nogil:
    double cos(double theta)
    double sin(double theta)

"""
def fast_time_evolve_basic(times, other_dimension, A, W):
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
"""

"""
@cython.boundscheck(False) # Disable bounds-checking
@cython.wraparound(False)  # Disable negative index wrapping
def fast_time_evolve(np.ndarray[np.float64_t, ndim=1] times, int other_dimension, np.ndarray[np.float64_t, ndim=2] A, np.ndarray[np.float64_t, ndim=2] W):
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

@cython.boundscheck(False) # Disable bounds-checking
@cython.wraparound(False)  # Disable negative index wrapping
def fast_time_evolve_parallel(double [:] times, double other_dimension, double [:, ::1] A, double [:, ::1] W):
    """Computes the result of time evolution of a muon polarisation

    Fast method for computing the result of time evolution of a
    muon polarisation. It requires the T -> inf approximation.
    Calculated as:
    \frac{1}{d}\sum_{\alpha, \beta}|<\alpha|\sigma_{\mu}^{\hat{n}}|\beta>|^2
        \cos[(E_{\alpha} - E{\alpha})t]

    Arguments:
        other_dimension {double} -- Value of the dimension labelled
                                    as d in the equation above.
        A {ndarray} -- Matrix giving the amplites of the cosines
                       |<\alpha|\sigma_{\mu}^{\hat{n}}|\beta>|^2
        W {ndarray} -- Matrix containing the differences of eigenvalues
                       computed with np.outer

    Returns:
        [DensityOperator | ndarray] -- DensityOperators or expectation values

    Raises:
        TypeError -- Invalid operators
        ValueError -- Invalid values of times or operators
        RuntimeError -- Hamiltonian is not hermitian
    """

    cdef Py_ssize_t num_times = times.shape[0]
    cdef Py_ssize_t mat_dim = A.shape[0]

    # Avoid using append as assignment should be faster
    cdef np.ndarray[np.float64_t, ndim=1] result = np.zeros((num_times), dtype=np.float64)
    cdef double [::1] result_view = result
    cdef Py_ssize_t i, j, k

    # Run times in parallel
    for i in prange(num_times, nogil=True):
        # k <= j
        for j in range(mat_dim):
            for k in range(mat_dim):
            #for k in range(0, j + 1):
                result_view[i] += A[j, k] * cos(W[j, k] * times[i])
        result_view[i] /= other_dimension

    return result