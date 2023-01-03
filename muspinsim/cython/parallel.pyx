import numpy as np
from cython.parallel import prange

cimport numpy as np
cimport cython

np.import_array()

# Use C versions to avoid linking back to python
cdef extern from "<math.h>" nogil:
    double cos(double theta)
    double sin(double theta)

@cython.boundscheck(False) # Disable bounds-checking
@cython.wraparound(False)  # Disable negative index wrapping
def parallel_fast_time_evolve(double [:] times, int other_dimension, double [:, :] A, double [:, :] W):
    """Computes the result of time evolution of a muon polarisation

    Fast method for computing the result of time evolution of a
    muon polarisation. To be accurate it requires that
    \frac{e \hbar^2 B}{2 m_p k T} -> 0 and so is most suitable when T is
    \inf or B is 0. 

    Calculated as:
    \frac{1}{2d}\sum_{\alpha, \beta}|<\alpha|\sigma_{\mu}^{\hat{n}}|\beta>|^2
    +
    \frac{1}{d}\sum_{\alpha < \beta}|<\alpha|\sigma_{\mu}^{\hat{n}}|\beta>|^2
        \cos[(E_{\alpha} - E{\alpha})t]

    Arguments:
        times {ndarray} -- Times to compute the evolution for, in microseconds
        other_dimension {int} -- Value of the dimension labelled as d in the
                                 equation above. Equal to half the total
                                 dimension of the system.
        A {ndarray} -- Matrix indexed by \alpha and \beta that
                       gives the amplitude
                       |<\alpha|\sigma_{\mu}^{\hat{n}}|\beta>|^2
                       of each cosine term in the sum
        W {ndarray} -- Matrix indexed by \alpha and \beta that contains the
                       differences of their eigenvalues, expressed as angular
                       frequency and computed with np.outer

    Returns:
        [ndarray] -- Expectation values
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
            # Term where j = k
            # Multiply by 0.5 here as want 1/2d here, and 1/d below
            results_view[i] += 0.5 * A[j, j]
            # Terms where k < j
            for k in range(0, j):
                results_view[i] += A[j, k] * cos(W[j, k] * times[i])
        results_view[i] *= one_over_d

    return results