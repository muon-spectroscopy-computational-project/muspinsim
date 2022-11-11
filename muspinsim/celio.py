"""celio.py

A class for handling the computation of hamiltonian for Celio's method via a
trotter expansion

See Phys. Rev. Lett. 56 2720 (1986)
"""

from dataclasses import dataclass
import itertools
import logging
import time
from typing import List
import numpy as np
from scipy import sparse
from qutip import Qobj

from muspinsim.spinop import SpinOperator, DensityOperator


@dataclass
class CelioHContrib:
    """
    Stores a Hamiltonian contribution term for use with Celio's method

    Arguments:
            matrix {matrix} -- Sparse matrix representing a contribution to the
                               hamiltonian
            other_dimension {int} -- Defines the product of the matrix sizes of any
                                     remaining spins that are not included in this
                                     hamiltonian contribution
            permutation_order {[int]} -- Defines the order of permutations that will be
                                         needed when constructing the contributioin to
                                         the trotter hamiltonian after the matrix
                                         exponential
            permutation_dimensions {[int]} -- Defines the size of the matrices involved
                                              in the kronecker products that make up
                                              this contribution to the Hamiltonian
    """

    matrix: sparse.csr_matrix
    other_dimension: int
    permute_order: List[int]
    permute_dimensions: List[int]


class CelioHamiltonian:
    def __init__(self, terms, k, spinsys):
        """Create a CelioHamiltonian

        Create a CelioHamiltonian for applying Celio's method

        Arguments:
            terms {[InteractionTerm]} -- Interaction terms that will form part of the
                                         Trotter expansion
            k {int} -- Factor to be used in the Trotter expansion
            spinsys {SpinSystem} -- SpinSystem required for computing the time evolution
        """
        self._terms = terms
        self._k = k
        self._spinsys = spinsys

    def __add__(self, x):
        return CelioHamiltonian(self._terms + x._terms, self._k, self._spinsys)

    def _calc_H_contribs(self) -> List[CelioHContrib]:
        """Calculates and returns the Hamiltonian contributions required for Celio's
           method

        Returns the hamiltonian contributions defined by this system of spins and the
        given interactions. In general these are split up per group of indices defined
        in interactions to minimise the need of matrix exponentials.

        Returns:
            H_contribs {[CelioHContrib]} -- List of matrices representing contributions
                                            to the total system hamiltonians refered to
                                            in Celio's method as H_i
        """

        spin_indices = range(0, len(self._spinsys.spins))

        H_contribs = []

        for i in spin_indices:
            # Only want to include each interaction once, will make the choice here to
            # only add it to the H_i for the first particle listed in the interactions

            # Find the terms that have the current spin as its first or only index
            spin_ints = [term for term in (self._terms) if i == term.indices[0]]

            # List of spin indices not included here
            other_spins = list(range(0, len(self._spinsys.spins)))
            other_spins.remove(i)

            # Only include necessary terms
            if len(spin_ints) != 0:
                # Sum matrices with the same indices so we avoid lots of matrix
                # exponentials
                for indices, group in itertools.groupby(
                    spin_ints, lambda term: term.indices
                ):

                    grouped_spin_ints = list(group)
                    H_contrib = np.sum([term.matrix for term in grouped_spin_ints])

                    # Find indices of spins not involved in the current interactions
                    other_spins_copy = other_spins.copy()
                    for term in grouped_spin_ints:
                        for j in term.indices:
                            if j in other_spins_copy:
                                other_spins_copy.remove(j)

                    other_dimension = np.product(
                        [self._spinsys.dimension[j] for j in other_spins_copy]
                    )

                    # Detect Quadrupolar terms which will have the same index twice
                    indices = list(indices)
                    if len(indices) == 2 and indices[0] == indices[1]:
                        # Only include one of them for the ordering
                        indices.pop()

                    # Order in which kronecker products will be performed in Celio's
                    # method
                    spin_order = list(indices) + other_spins_copy

                    # Order we need to permute in order to obtain the same order as was
                    # given in the input
                    permute_order = np.zeros(len(spin_order), dtype=np.int32)
                    for i, value in enumerate(spin_order):
                        permute_order[value] = i

                    permute_dimensions = [
                        self._spinsys.dimension[i] for i in spin_order
                    ]

                    H_contribs.append(
                        CelioHContrib(
                            H_contrib,
                            other_dimension,
                            permute_order,
                            permute_dimensions,
                        )
                    )

        return H_contribs

    def _calc_trotter_evol_op_contribs(self, time_step):
        """Calculates and returns the contributions to the Trotter expansion of
        the time evolution operator computed from the Hamiltonian contributions

        Arguments:
            time_step {float} -- Timestep that will be used during the evolution

        Returns:
            evol_op_contribs {[matrix]} -- Contributions to the Trotter expansion
                                           of the evolution operator
        """

        H_contribs = self._calc_H_contribs()

        evol_op_contribs = []

        for H_contrib in H_contribs:
            # The matrix is currently stored in csr format, but expm wants it in csc so
            # convert here
            evol_op_contrib = sparse.linalg.expm(
                -2j * np.pi * H_contrib.matrix.tocsc() * time_step / self._k
            ).tocsr()

            if H_contrib.other_dimension > 1:
                evol_op_contrib = sparse.kron(
                    evol_op_contrib,
                    sparse.identity(H_contrib.other_dimension, format="csr"),
                )

            # For particle interactions that are not neighbours we must use a swap gate
            qtip_obj = Qobj(
                inpt=evol_op_contrib,
                dims=[H_contrib.permute_dimensions, H_contrib.permute_dimensions],
            )
            qtip_obj = qtip_obj.permute(H_contrib.permute_order)
            evol_op_contrib = qtip_obj.data

            evol_op_contribs.append(evol_op_contrib)

        return evol_op_contribs

    def evolve(self, rho0, times, operators=[]):
        """Time evolution of a state under this Hamiltonian

        Perform an evolution of a state described by a DensityOperator under
        this Hamiltonian and return either a sequence of DensityOperators or
        a sequence of expectation values for given SpinOperators.

        Arguments:
            rho0 {DensityOperator} -- Initial state
            times {ndarray} -- Times to compute the evolution for, in microseconds

        Keyword Arguments:
            operators {[SpinOperator]} -- List of SpinOperators to compute the
                                          expectation values of at each step.
                                          If omitted, the states' density
                                          matrices will be returned instead
                                           (default: {[]})

        Returns:
            [DensityOperator | ndarray] -- DensityOperators or expectation values

        Raises:
            TypeError -- Invalid operators
            ValueError -- Invalid values of times or operators
            RuntimeError -- Hamiltonian is not hermitian
        """

        if not isinstance(rho0, DensityOperator):
            raise TypeError("rho0 must be a valid DensityOperator")

        times = np.array(times)

        if len(times.shape) != 1:
            raise ValueError("times must be an array of values in microseconds")

        if isinstance(operators, SpinOperator):
            operators = [operators]
        if not all([isinstance(o, SpinOperator) for o in operators]):
            raise ValueError(
                "operators must be a SpinOperator or a list of SpinOperator objects"
            )

        time_step = times[1] - times[0]
        rho0 = rho0.matrix

        # Time evolution step that will modify the trotter_hamiltonian below
        evol_op = np.product(self._calc_trotter_evol_op_contribs(time_step)) ** self._k
        total_evol_op = sparse.identity(evol_op.shape[0], format="csr")

        mat_density = evol_op.getnnz() / np.prod(evol_op.shape)

        if mat_density >= 0.08:
            logging.warning(
                "Matrix density is %s >= 0.08 and so Celio's method is not suitable, "
                "consider disabling it.",
                mat_density,
            )
            # Matrix products with trotter_hamiltonian_dt is very likely to be slower
            # with sparse matrices than dense

            # We can still save some memory over Hamiltonian's evolve method at the
            # cost of performance by using dense matrices for trotter_hamiltonian
            # trotter_hamiltonian_dt but the improvement is minimal and as the problem
            # gets bigger the reduction in memory usage decreases and increase in time
            # increases so does not appear worth it

        # Avoid using append as assignment should be faster
        results = np.zeros((times.shape[0], len(operators)), dtype=np.complex128)

        # Obtain transpose of operators
        operatorsT = np.array([o.matrix.T for o in operators])

        if len(operators) > 0:
            # Compute expectation values one at a time
            for i in range(times.shape[0]):
                # Evolve rho0
                rho = total_evol_op.conj().T * (rho0 * total_evol_op)

                # When passing multiple operators we want to return results for each
                for j, operatorT in enumerate(operatorsT):
                    # This element wise multiplication then sum gives the equivalent
                    # as the trace of the matrix product (without the transpose) and
                    # and is faster
                    results[i][j] = np.sum(
                        np.sum(rho.multiply(operatorT), axis=1), axis=0
                    )

                # Evolution step
                total_evol_op = total_evol_op * evol_op

        return results

    def fast_evolve(self, sigma_mu, times, operators=[]):
        """Time evolution of a state under this Hamiltonian

        Perform an evolution of a state described by a DensityOperator under
        this Hamiltonian and return either a sequence of DensityOperators or
        a sequence of expectation values for given SpinOperators.

        Arguments:
            sigma_mu {matrix} -- Spin matrix of the muon
            times {ndarray} -- Times to compute the evolution for, in microseconds

        Keyword Arguments:
            operators {[SpinOperator]} -- List of SpinOperators to compute the
                                          expectation values of at each step.
                                          If omitted, the states' density
                                          matrices will be returned instead
                                           (default: {[]})

        Returns:
            [DensityOperator | ndarray] -- DensityOperators or expectation values

        Raises:
            TypeError -- Invalid operators
            ValueError -- Invalid values of times or operators
            RuntimeError -- Hamiltonian is not hermitian
        """

        times = np.array(times)

        if len(times.shape) != 1:
            raise ValueError("times must be an array of values in microseconds")

        if isinstance(operators, SpinOperator):
            operators = [operators]
        if not all([isinstance(o, SpinOperator) for o in operators]):
            raise ValueError(
                "operators must be a SpinOperator or a list of SpinOperator objects"
            )

        time_step = times[1] - times[0]

        e, v = np.linalg.eig(sigma_mu - np.eye(2))
        mu_psi = v[:, 1] if e[1] > 0.1 else v[:, 0]

        # Time evolution step that will modify the trotter_hamiltonian below
        evol_op_contribs = self._calc_trotter_evol_op_contribs(time_step)

        half_dim = int(evol_op_contribs[0].shape[0] / 2)

        # mat_density = evol_op.getnnz() / np.prod(evol_op.shape)

        # if mat_density >= 0.08:
        #     logging.warning(
        #         "Matrix density is %s >= 0.08 and so Celio's method is not suitable, "
        #         "consider disabling it.",
        #         mat_density,
        #     )
        #     # Matrix products with trotter_hamiltonian_dt is very likely to be slower
        #     # with sparse matrices than dense

        #     # We can still save some memory over Hamiltonian's evolve method at the
        #     # cost of performance by using dense matrices for trotter_hamiltonian
        #     # trotter_hamiltonian_dt but the improvement is minimal and as the problem
        #     # gets bigger the reduction in memory usage decreases and increase in time
        #     # increases so does not appear worth it

        # Avoid using append as assignment should be faster
        results = np.zeros((times.shape[0], len(operators)), dtype=np.complex128)

        # Obtain transpose of operators
        operatorsT = np.array([o.matrix.T for o in operators])

        averages = 12
        avg_factor = 1 / averages

        def compute_psi(mu_psi, half_dim):
            psi0 = np.exp(2j * np.pi * np.random.rand(half_dim))
            psi = np.kron(mu_psi.T, psi0)

            # Normalise
            psi = psi * (1 / np.sqrt(half_dim))
            return sparse.csr_matrix(psi).T

        start_t = time.time()
        if len(operators) > 0:
            for _ in range(averages):
                psi = compute_psi(mu_psi, half_dim)

                # Compute expectation values one at a time
                for i in range(times.shape[0]):
                    # When passing multiple operators we want to return results for each
                    for j, operatorT in enumerate(operatorsT):
                        results[i][j] += (psi.conj().T * (operatorT * psi))[0, 0]

                    # Evolution step
                    for _ in range(self._k):
                        for evol_op_contrib in evol_op_contribs:
                            psi = evol_op_contrib * psi
        print("Time", time.time() - start_t)

        return results * avg_factor

    def integrate_decaying(self, rho0, tau, operators=[]):
        """Called to integrate one or more expectation values in time with decay

        Raises:
            NotImplementedError
        """
        raise NotImplementedError(
            "integrate_decaying is not implemented for Celio's method"
        )
