#include "base.h"

PYBIND11_MODULE(cpp, m) {
    m.doc() = "MuSpinSim C++ Module";

    // Call all init functions here
    test::init(m);
    parallel::init(m);
    celio::init(m);
}