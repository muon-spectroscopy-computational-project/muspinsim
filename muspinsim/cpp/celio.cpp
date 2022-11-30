#include "base.h"

/* Structure for storing information about a Hamiltonian contribution */
struct celio::EvolveContrib {
    np_array_complex_t matrix;
    size_t other_dim;
    np_array_size_t indices;

    EvolveContrib(np_array_complex_t matrix, size_t other_dim, np_array_size_t indices) : matrix(matrix), other_dim(other_dim), indices(indices) {}
};

/* Init function for defining python bindings */
void celio::init(py::module_& m) {
    py::class_<celio::EvolveContrib>(m, "Celio_EvolveContrib").def(py::init<np_array_complex_t, size_t, np_array_size_t>());
}