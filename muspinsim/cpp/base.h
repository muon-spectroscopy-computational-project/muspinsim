#pragma once

/* Includes most likely to be used everywhere */
#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

/* Forward declarations of methods in this header allows for autocomplete
   where methods are needed in different source files to their own */

/* As a best practice init functions are defined here for each file and are
   called in main.cpp to define the bindings */

/* Functions for testing pybind11 works*/
namespace test {
void init(py::module_&);

/* Adds two numbers */
int add(int i, int j);
};