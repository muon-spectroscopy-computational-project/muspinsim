#include "base.h"

/* Structure for storing information about a Hamiltonian contribution */
struct celio::EvolveContrib {
    np_array_complex_t matrix;
    size_t other_dim;
    np_array_size_t indices;

    EvolveContrib(np_array_complex_t matrix, size_t other_dim, np_array_size_t indices) : matrix(matrix), other_dim(other_dim), indices(indices) {}
};

/*
 * Performs Celio's method and adds the result to an array in place
 *
 * @param num_times Number of time steps to compute for
 * @param psi Initial approximated spin states
 * @param sigma_mu Spin matrix of the muon
 * @param half_dim Half the total dimension of the system
 * @param k Value of k used in the Trotter expansion
 * @param evol_contribs List of EvolveContrib structures containing information about the Hamiltonian contributions
 * @param results Array to add the results to (Should have a length equal to num_times)
 */
void celio::evolve(unsigned int num_times, np_array_complex_t& psi, np_array_complex_t& sigma_mu, size_t half_dim, unsigned int k, const py::list& evol_contribs, np_array_double_t& results) {
    py::buffer_info psi_info = psi.request();
    auto* psi_ptr = static_cast<std::complex<double>*>(psi_info.ptr);

    py::buffer_info sigma_mu_info = sigma_mu.request();
    auto* sigma_mu_ptr = static_cast<std::complex<double>*>(sigma_mu_info.ptr);

    py::buffer_info results_info = results.request();
    auto* results_ptr = static_cast<double*>(results_info.ptr);

    // Will be faster to do any casting before the method begins, so obtain the needed info
    // from the evol_contribs here
    std::vector<std::complex<double>*> evol_contrib_mat_ptrs(evol_contribs.size());
    std::vector<size_t> evol_contrib_mat_dims(evol_contribs.size());
    std::vector<size_t*> evol_contrib_index_ptrs(evol_contribs.size());
    std::vector<size_t> evol_contrib_other_dims(evol_contribs.size());

    for (unsigned int i = 0; i < evol_contribs.size(); ++i) {
        // Obtain a reference
        celio::EvolveContrib& evol_contrib = evol_contribs[i].cast<celio::EvolveContrib&>();

        py::buffer_info evol_contrib_mat_info = evol_contrib.matrix.request();
        py::buffer_info evol_contrib_indices_info = evol_contrib.indices.request();

        evol_contrib_mat_ptrs[i] = static_cast<std::complex<double>*>(evol_contrib_mat_info.ptr);
        evol_contrib_mat_dims[i] = evol_contrib_mat_info.shape[0];
        evol_contrib_index_ptrs[i] = static_cast<size_t*>(evol_contrib_indices_info.ptr);
        evol_contrib_other_dims[i] = evol_contrib.other_dim;
    }

    // Now compute for each time step
    for (unsigned int i = 0; i < num_times; ++i) {
        // Measure
        results_ptr[i] += parallel::fast_measure_h_ptr(psi_ptr, sigma_mu_ptr, sigma_mu_info.shape[0], half_dim);

        // Evolve
        for (unsigned int _k = 0; _k < k; ++_k) {
            for (unsigned int j = 0; j < evol_contribs.size(); ++j)
                parallel::fast_evolve_ptr(
                    psi_ptr,
                    evol_contrib_mat_ptrs[j],
                    evol_contrib_mat_dims[j],
                    evol_contrib_other_dims[j],
                    evol_contrib_index_ptrs[j]);
        }
    }
}

/* Init function for defining python bindings */
void celio::init(py::module_& m) {
    py::class_<celio::EvolveContrib>(m, "Celio_EvolveContrib")
        .def(py::init<np_array_complex_t, size_t, np_array_size_t>())
        // Allow access to python
        .def_readonly("matrix", &EvolveContrib::matrix)
        .def_readonly("other_dim", &EvolveContrib::other_dim)
        .def_readonly("indices", &EvolveContrib::indices);

    m.def("celio_evolve", &celio::evolve, "Performs Celio's method and adds the result to an array in place");
}