#pragma once

/* Includes most likely to be used everywhere */
#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

/* Typedef to make using taking numpy arrays a bit clearer*/
using np_array_complex_t = py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast>;
using np_array_size_t = py::array_t<size_t, py::array::c_style | py::array::forcecast>;
using np_array_double_t = py::array_t<double, py::array::c_style | py::array::forcecast>;

/* Forward declarations of methods in this header allows for autocomplete
   where methods are needed in different source files to their own */

/* As a best practice init functions are defined here for each file and are
   called in main.cpp to define the bindings */

/* Functions for parallel operations */
namespace parallel {
void init(py::module_&);

/* Computes V^\dagger (M \otimes 1_{d}) V between a complex vector V and a
   matrix M where 1_{d} an identity matrix of size d */
double fast_measure(np_array_complex_t& V, np_array_complex_t& M, long int d);
double fast_measure_ptr(std::complex<double>* V_ptr, std::complex<double>* M_ptr, unsigned int M_dim, long int d);

/* Computes V^\dagger (M \otimes 1_{d}) V between a complex vector V and a
   Hermitian matrix M where 1_{d} an identity matrix of size d */
double fast_measure_h(np_array_complex_t& V, np_array_complex_t& M, long int d);
double fast_measure_h_ptr(std::complex<double>* V_ptr, std::complex<double>* M_ptr, unsigned int M_dim, long int d);

/* Computes (M \otimes 1_{d}) V, modifying V inplace, where V is a complex
   vector, M is a complex square matrix and 1_{d} an identity matrix of
   size d. The indices allow the ability to modify the order of the
   kronecker products (assuming 1_{d} is the result of multiple products
   of smaller identities). */
void fast_evolve(np_array_complex_t& V, np_array_complex_t& M, int d, np_array_size_t& indices);
void fast_evolve_ptr(std::complex<double>* V_ptr, std::complex<double>* M_ptr, unsigned int M_dim, long int d, size_t* indices);
};  // namespace parallel

/* Functions for Celio's method */
namespace celio {
void init(py::module_&);

/* Structure for storing information about a Hamiltonian contribution */
struct EvolveContrib;

/* Performs Celio's method and adds the result to an array in place */
void evolve(unsigned int num_times, np_array_complex_t& psi, np_array_complex_t& sigma_mu, size_t half_dim, unsigned int k, const py::list& evol_op_contribs, np_array_double_t& results);
}  // namespace celio