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
def cy_parallel_fast_time_evolve(double [:] times, int other_dimension, double [:, :] A, double [:, :] W):
    """Computes the result of time evolution of a muon polarisation

    Fast method for computing the result of time evolution of a
    muon polarisation. It requires the T -> inf approximation.
    Calculated as:
    \frac{1}{2d}\sum_{\alpha, \beta}|<\alpha|\sigma_{\mu}^{\hat{n}}|\beta>|^2
    +
    \frac{1}{d}\sum_{\alpha < \beta}|<\alpha|\sigma_{\mu}^{\hat{n}}|\beta>|^2
        \cos[(E_{\alpha} - E{\alpha})t]

    Arguments:
        other_dimension {int} -- Value of the dimension labelled as d in the
                                 equation above. Equal to half the total
                                 dimension of the system.
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

    # Use a memory view to the results array to make access faster
    cdef np.ndarray[np.float64_t, ndim=1] results = np.zeros((num_times), dtype=np.float64)
    cdef double [:] results_view = results
    
    cdef Py_ssize_t i, j, k

    cdef double one_over_d = 1.0 / other_dimension

    # Run times in parallel
    for i in prange(num_times, nogil=True):
        for j in range(mat_dim):
            # k <= j
            for k in range(0, j + 1):
                if k == j:
                    # Multiply by 0.5 here as want 1/2d here, and 1/d below
                    results_view[i] += 0.5 * A[j, k]
                else:
                    results_view[i] += A[j, k] * cos(W[j, k] * times[i])
        results_view[i] *= one_over_d

    return results