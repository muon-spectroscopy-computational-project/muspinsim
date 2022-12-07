#include "base.h"

/*
 * Computes V^\dagger (M \otimes 1_{d}) V between a complex vector V and a
 * matrix M where 1_{d} an identity matrix of size d
 *
 * For speed, no checks are performed here. The matrix M should be Hermitian
 * with size N and the shape of V should be (N*d, 1).
 *
 * No swapping is done here as it is assumed the thing being measured is
 * first in the system.
 *
 * @param V The vector
 * @param M The matrix
 * @param d The dimension of the identity matrix
 *
 * @return The result of the combined product. Will be a single value.
 */
double parallel::fast_measure(np_array_complex_t& V, np_array_complex_t& M, long int d) {
    py::buffer_info V_info = V.request();
    py::buffer_info M_info = M.request();

    return parallel::fast_measure_ptr(
        static_cast<std::complex<double>*>(V_info.ptr),
        static_cast<std::complex<double>*>(M_info.ptr),
        M_info.shape[0],
        d);
}

/*
 * Computes V^\dagger (M \otimes 1_{d}) V between a complex vector V and a
 * matrix M where 1_{d} an identity matrix of size d
 *
 * For speed, no checks are performed here. The matrix M should be Hermitian
 * with size N and the shape of V should be (N*d, 1).
 *
 * No swapping is done here as it is assumed the thing being measured is
 * first in the system.
 *
 * (Same as above, but only for use in C++ - uses pointers instead of numpy
 * arrays)
 *
 * @param V_ptr Pointer to the vector
 * @param M_ptr Pointer to the matrix
 * @param M_dim Dimension of M
 * @param d The dimension of the identity matrix
 *
 * @return The result of the combined product. Will be a single value.
 */
double parallel::fast_measure_ptr(std::complex<double>* V_ptr, std::complex<double>* M_ptr, unsigned int M_dim, long int d) {
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
        // OpenMP on Windows needs to use signed int for the index
        // Use long int as muon and 9 spin 7/2 will use max integer value
        for (long int k = 0; k < d; ++k) {
            for (unsigned int i = 0; i < M_dim; ++i) {
                mat_col_offset = i * M_dim;
                term = {0, 0};
                for (unsigned int j = 0; j < M_dim; ++j)
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
 * No swapping is done here as it is assumed the thing being measured is
 * first in the system.
 *
 * @param V The vector
 * @param M The matrix
 * @param d The dimension of the identity matrix
 *
 * @return The result of the combined product. Will be a single value.
 */
double parallel::fast_measure_h(np_array_complex_t& V, np_array_complex_t& M, long int d) {
    py::buffer_info V_info = V.request();
    py::buffer_info M_info = M.request();

    return parallel::fast_measure_h_ptr(
        static_cast<std::complex<double>*>(V_info.ptr),
        static_cast<std::complex<double>*>(M_info.ptr),
        M_info.shape[0],
        d);
}

/*
 * Computes V^\dagger (M \otimes 1_{d}) V between a complex vector V and a
 * Hermitian matrix M where 1_{d} an identity matrix of size d
 *
 * For speed, no checks are performed here. The matrix M should be Hermitian
 * with size N and the shape of V should be (N*d, 1).
 *
 * No swapping is done here as it is assumed the thing being measured is
 * first in the system.
 *
 * (Same as above, but only for use in C++ - uses pointers instead of numpy
 * arrays)
 *
 * @param V_ptr Pointer to the vector
 * @param M_ptr Pointer to the matrix
 * @param M_dim Dimension of M
 * @param d The dimension of the identity matrix
 *
 * @return The result of the combined product. Will be a single value.
 */
double parallel::fast_measure_h_ptr(std::complex<double>* V_ptr, std::complex<double>* M_ptr, unsigned int M_dim, long int d) {
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
        // OpenMP on Windows needs to use signed int for the index
        // Use long int as muon and 9 spin 7/2 will use max integer value
        for (long int k = 0; k < d; ++k) {
            for (unsigned int i = 0; i < M_dim; ++i) {
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

/*
 * Computes (M \otimes 1_{d}) V, modifying V inplace, where V is a complex
 * vector, M is a complex square matrix and 1_{d} an identity matrix of
 * size d. The indices can be used to modify the rows/columns accessed and
 * when chosen appropriately can be used to act as a swap gate.
 * (assuming 1_{d} is the result of multiple kronecker products of smaller
 * identities)
 *
 * For speed, no checks are performed here. The matrix M should be square
 * with size N and the shape of V should be (N*d, 1). The number of indices
 * supplied should be equal to N*d.
 *
 * @param V The vector
 * @param M The matrix
 * @param d The total dimension M should increase by after kronecker
 *          products. E.g. a 2x2 matrix with d = 8 would become
 *          a 16x16 matrix.
 * @param indices Indices used to offset the accessed rows of V.
 *
 *                To compute M \otimes 1_d you would need to supply N*d
 *                indices in order from 0 to N*d.
 *
 *                To generate these to act like a swap gate you can use
 *                np.transpose e.g.
 *                  np.transpose(
 *                      np.arange(np.product(dimensions)).reshape(
 *                          dimensions
 *                      ),
 *                      axes=new_order,
 *                  ).flatten()
 *                For a list of dimensions and a new order of kronecker
 *                products.
 */
void parallel::fast_evolve(np_array_complex_t& V, np_array_complex_t& M, int d, np_array_size_t& indices) {
    py::buffer_info V_info = V.request();
    py::buffer_info M_info = M.request();
    py::buffer_info indices_info = indices.request();

    parallel::fast_evolve_ptr(
        static_cast<std::complex<double>*>(V_info.ptr),
        static_cast<std::complex<double>*>(M_info.ptr),
        M_info.shape[0],
        d,
        static_cast<size_t*>(indices_info.ptr));
}

/*
 * Computes (M \otimes 1_{d}) V, modifying V inplace, where V is a complex
 * vector, M is a complex square matrix and 1_{d} an identity matrix of
 * size d. The indices can be used to modify the rows/columns accessed and
 * when chosen appropriately can be used to act as a swap gate.
 * (assuming 1_{d} is the result of multiple kronecker products of smaller
 * identities)
 *
 * For speed, no checks are performed here. The matrix M should be square
 * with size N and the shape of V should be (N*d, 1). The number of indices
 * supplied should be equal to N*d.
 *
 * (Same as above, but only for use in C++ - uses pointers instead of numpy)
 *
 * @param V_ptr Pointer to the vector
 * @param M_ptr Pointer to the matrix
 * @param M_dim Dimension of M
 * @param d The total dimension M should increase by
 * @param indices Indices used to offset the accessed rows of V
 */
void parallel::fast_evolve_ptr(std::complex<double>* V_ptr, std::complex<double>* M_ptr, unsigned int M_dim, long int d, size_t* indices_ptr) {
#pragma omp parallel
    {
        // Avoid unecessary multiply ops
        size_t mat_col_offset;

        // Temp array to ensure we can modify psi in place without modifying
        // the result
        std::complex<double>* tmp = new std::complex<double>[M_dim];

// Parallelise over d - likely to be a large value
#pragma omp for
        // OpenMP on Windows needs to use signed int for the index
        // Use long int as muon and 9 spin 7/2 will use max integer value
        for (long int k = 0; k < d; ++k) {
            for (size_t i = 0; i < M_dim; ++i) {
                mat_col_offset = i * M_dim;
                tmp[i] = {0.0, 0.0};

                for (size_t j = 0; j < M_dim; ++j)
                    // M[i, j] -> M[i * d + j] as column major
                    tmp[i] += M_ptr[mat_col_offset + j] * V_ptr[indices_ptr[j * d + k]];
            }
            // Must do this separately as V_ptr used in above calculation
            for (size_t i = 0; i < M_dim; ++i)
                V_ptr[indices_ptr[i * d + k]] = tmp[i];
        }
        // Allocated with new, so delete
        delete[] tmp;
    }
}

/* Init function for defining python bindings */
void parallel::init(py::module_& m) {
    m.def("parallel_fast_measure", &parallel::fast_measure, "Computes V^\\dagger (M \\otimes 1_{d}) V between a complex vector V and a matrix M where 1_{d} an identity matrix of size d");
    m.def("parallel_fast_measure_h", &parallel::fast_measure_h, "Computes V^\\dagger (M \\otimes 1_{d}) V between a complex vector V and a Hermitian matrix M where 1_{d} an identity matrix of size d");
    m.def("parallel_fast_evolve", &parallel::fast_evolve, "Modifies a vector, V to have a value equal to its product with a matrix equal to M with some amount of kronecker products with identity matrices");
}