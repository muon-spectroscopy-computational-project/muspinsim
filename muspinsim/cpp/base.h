#pragma once

/* Includes most likely to be used everywhere */
#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

/* Typedef to make using taking numpy arrays a bit clearer*/
using np_array_complex_t = py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast>;

/* Forward declarations of methods in this header allows for autocomplete
   where methods are needed in different source files to their own */

/* As a best practice init functions are defined here for each file and are
   called in main.cpp to define the bindings */

/* Functions for testing pybind11 works*/
namespace test {
void init(py::module_&);

/* Adds two numbers */
int add(int i, int j);
};  // namespace test

/* Functions for parallel operations */
namespace parallel {
void init(py::module_&);

/* Computes V^\dagger (M \otimes 1_{d}) V between a complex vector V and a
   matrix M where 1_{d} an identity matrix of size d */
double fast_measure(np_array_complex_t V, np_array_complex_t M, size_t d);
double fast_measure_ptr(std::complex<double>* V_ptr, size_t V_dim, std::complex<double>* M_ptr, size_t M_dim, size_t d);

/* Computes V^\dagger (M \otimes 1_{d}) V between a complex vector V and a
   Hermitian matrix M where 1_{d} an identity matrix of size d */
double fast_measure_h(np_array_complex_t V, np_array_complex_t M, size_t d);
double fast_measure_h_ptr(std::complex<double>* V_ptr, size_t V_dim, std::complex<double>* M_ptr, size_t M_dim, size_t d);
};  // namespace parallel