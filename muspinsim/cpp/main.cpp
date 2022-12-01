#include "base.h"

PYBIND11_MODULE(cpp, m) {
    m.doc() = "MuSpinSim C++ Module";

    // Call all init functions here
    parallel::init(m);
    celio::init(m);
}