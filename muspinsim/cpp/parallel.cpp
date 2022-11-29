#include "base.h"

/*
 * Computes V^\dagger (M \otimes 1_{d}) V between a complex vector V and a
 * matrix M where 1_{d} an identity matrix of size d
 *
 * For speed, no checks are performed here. The matrix M should be Hermitian
 * with size N and the shape of V should be (N*d, 1).
 *
 * @param V The vector
 * @param M The matrix
 * @param d The dimension of the identity matrix
 *
 * @return The result of the combined product. Will be a single value.
 */
double parallel::fast_measure(np_array_complex_t V, np_array_complex_t M, size_t d) {
    py::buffer_info V_info = V.request();
    py::buffer_info M_info = M.request();

    return parallel::fast_measure_ptr(
        static_cast<std::complex<double>*>(V_info.ptr),
        V_info.shape[0],
        static_cast<std::complex<double>*>(M_info.ptr),
        M_info.shape[0],
        d);
}

/* Same as above, but only for use in C++ - uses pointers instead of numpy
 * arrays
 *
 * @param V_ptr Pointer to the vector
 * @param V_dim Dimension of V
 * @param M_ptr Pointer to the matrix
 * @param M_dim Dimension of M
 * @param d The dimension of the identity matrix
 *
 * @return The result of the combined product. Will be a single value.
 */
double parallel::fast_measure_ptr(std::complex<double>* V_ptr, size_t V_dim, std::complex<double>* M_ptr, size_t M_dim, size_t d) {
    double result = 0;

// Sum results from separate threads into result
#pragma omp parallel reduction(+ \
                               : result)
    {
        // Used to store the result of (M \otimes 1_d) * V
        std::complex<double> term;

        // Avoid unecessary multiply ops
        size_t mat_col_offset;

// Parallelise over d - likely to be a large value
#pragma omp for
        for (size_t k = 0; k < d; ++k) {
            for (size_t i = 0; i < M_dim; ++i) {
                mat_col_offset = i * M_dim;
                term = {0, 0};
                for (size_t j = 0; j < M_dim; ++j)
                    // M[i, j] -> M[i * operator_dim + j] as column major
                    term += M_ptr[mat_col_offset + j] * V_ptr[j * d + k];
                // Sum current result of full product
                result += (std::conj(V_ptr[i * d + k]) * term).real();
            }
        }
    }

    return result;
}

/*
 * Computes V^\dagger (M \otimes 1_{d}) V between a complex vector V and a
 * Hermitian matrix M where 1_{d} an identity matrix of size d
 *
 * For speed, no checks are performed here. The matrix M should be Hermitian
 * with size N and the shape of V should be (N*d, 1).
 *
 * @param V The vector
 * @param M The matrix
 * @param d The dimension of the identity matrix
 *
 * @return The result of the combined product. Will be a single value.
 */
double parallel::fast_measure_h(np_array_complex_t V, np_array_complex_t M, size_t d) {
    py::buffer_info V_info = V.request();
    py::buffer_info M_info = M.request();

    return parallel::fast_measure_h_ptr(
        static_cast<std::complex<double>*>(V_info.ptr),
        V_info.shape[0],
        static_cast<std::complex<double>*>(M_info.ptr),
        M_info.shape[0],
        d);
}

/* Same as above, but only for use in C++ - uses pointers instead of numpy
 * arrays
 *
 * @param V_ptr Pointer to the vector
 * @param V_dim Dimension of V
 * @param M_ptr Pointer to the matrix
 * @param M_dim Dimension of M
 * @param d The dimension of the identity matrix
 *
 * @return The result of the combined product. Will be a single value.
 */
double parallel::fast_measure_h_ptr(std::complex<double>* V_ptr, size_t V_dim, std::complex<double>* M_ptr, size_t M_dim, size_t d) {
    double result = 0;

// Sum results from separate threads into result
#pragma omp parallel reduction(+ \
                               : result)
    {
        // Used to store the result of (M \otimes 1_d) * V
        std::complex<double> term;

        // Avoid unecessary multiply ops
        size_t mat_col_offset;

// Parallelise over d - likely to be a large value
#pragma omp for
        for (size_t k = 0; k < d; ++k) {
            for (size_t i = 0; i < M_dim; ++i) {
                mat_col_offset = i * M_dim;
                // Initialise with term on diagonal
                term = M_ptr[mat_col_offset + i] * V_ptr[i * d + k];

                // Use the fact the matrix is symmetric - then we can take
                // only the values of j > i and double (since we are taking
                // the real part only in the end also due to commutativity
                // can do here before multiplying by the conjugate)
                for (size_t j = i + 1; j < M_dim; ++j)
                    // M[i, j] -> M[i * operator_dim + j] as column major
                    term += M_ptr[mat_col_offset + j] * V_ptr[j * d + k] * 2.0;

                // Sum current result of full product
                result += (std::conj(V_ptr[i * d + k]) * term).real();
            }
        }
    }

    return result;
}

/* Init function for defining python bindings */
void parallel::init(py::module_& m) {
    m.def("parallel_fast_measure", &parallel::fast_measure, "Computes V^\\dagger (M \\otimes 1_{d}) V between a complex vector V and a matrix M where 1_{d} an identity matrix of size d");
    m.def("parallel_fast_measure_h", &parallel::fast_measure_h, "Computes V^\\dagger (M \\otimes 1_{d}) V between a complex vector V and a Hermitian matrix M where 1_{d} an identity matrix of size d");
}