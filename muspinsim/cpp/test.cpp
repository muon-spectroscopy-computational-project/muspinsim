#include "base.h"

int test::add(int i, int j) {
    return i + j;
}

/* Init function for defining python bindings for functions */
void test::init(py::module_& m) {
    m.def("add", &test::add, "A function that adds two numbers");
}