"""hamiltonian.py

A class describing a spin Hamiltonian with various terms
"""

from typing import List
import numpy as np
from dataclasses import dataclass
import itertools
import logging
from typing import List
from qutip import Qobj

from scipy import sparse
from muspinsim.celio import CelioHContrib

from muspinsim.cython import parallel_fast_time_evolve
from muspinsim.spinop import SpinOperator, DensityOperator, Operator, Hermitian
from muspinsim.validation import (
    validate_evolve_params,
    validate_integrate_decaying_params,
    validate_times,
)


class Hamiltonian2(Operator, Hermitian):
    def __init__(self, terms, spinsys):
        """Create a CelioHamiltonian

        Create a CelioHamiltonian for applying Celio's method

        Arguments:
            terms {[InteractionTerm]} -- Interaction terms that will form part of the
                                         Trotter expansion
            k {int} -- Factor to be used in the Trotter expansion
            spinsys {SpinSystem} -- SpinSystem required for computing the time evolution
        """
        self._terms = terms
        self._spinsys = spinsys

    def __add__(self, x):
        return Hamiltonian2(self._terms + x._terms, self._spinsys)

    def _calc_H_contribs(self) -> List[CelioHContrib]:
        """Calculates and returns the Hamiltonian contributions required for Celio's
           method

        Returns the hamiltonian contributions defined by this system of spins and the
        given interactions. In general these are split up per group of indices defined
        in interactions to minimise the need of matrix exponentials.

        Returns:
            H_contribs {[CelioHContrib]} -- List of matrices representing contributions
                                            to the total system hamiltonians referred to
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
                    uninvolved_spins = other_spins.copy()
                    for term in grouped_spin_ints:
                        for j in term.indices:
                            if j in uninvolved_spins:
                                uninvolved_spins.remove(j)

                    other_dimension = np.product(
                        [self._spinsys.dimension[j] for j in uninvolved_spins]
                    )

                    # Detect Quadrupolar terms which will have the same index twice
                    indices = list(indices)
                    if len(indices) == 2 and indices[0] == indices[1]:
                        # Only include one of them for the ordering
                        indices.pop()

                    # Order in which kronecker products will be performed in Celio's
                    # method and the corresponding dimensions
                    spin_order = indices + uninvolved_spins
                    spin_dimensions = [self._spinsys.dimension[i] for i in spin_order]

                    H_contribs.append(
                        CelioHContrib(
                            H_contrib,
                            other_dimension,
                            spin_order,
                            spin_dimensions,
                        )
                    )

        return H_contribs

    def _calc_full_hamiltonian(self):
        """Calculates and returns the contributions to the Trotter expansion of
        the time evolution operator computed from the Hamiltonian contributions

        Arguments:
            time_step {float} -- Timestep that will be used during the evolution
            cpp {boolean} -- When true will calculate ready for use with the C++
                             version

        Returns:
            evol_op_contribs {[matrix]} -- Contributions to the Trotter expansion
                                           of the evolution operator
        """

        H_contribs = self._calc_H_contribs()

        total_H = None

        for H_contrib in H_contribs:
            # The matrix is currently stored in csr format, but expm wants it
            # in csc so convert here
            evol_op_contrib = H_contrib.matrix

            # Python version - requires kronecker products
            if H_contrib.other_dimension > 1:
                evol_op_contrib = sparse.kron(
                    evol_op_contrib,
                    sparse.identity(H_contrib.other_dimension, format="csr"),
                )

            # For particle interactions that are not neighbors we must use
            # a swap gate
            qtip_obj = Qobj(
                inpt=evol_op_contrib,
                dims=[H_contrib.spin_dimensions, H_contrib.spin_dimensions],
            )
            # Order we need to permute in order to obtain the same order
            # as was given in the input
            permute_order = np.argsort(H_contrib.spin_order)

            qtip_obj = qtip_obj.permute(permute_order)
            evol_op_contrib = qtip_obj.data

            if total_H is None:
                total_H = evol_op_contrib
            else:
                total_H += evol_op_contrib

        return total_H

    @classmethod
    def from_spin_operator(self, spinop):
        return self(spinop.matrix, spinop.dimension)

    def diag(self):
        total_H = self._calc_full_hamiltonian()

        return np.linalg.eigh(total_H.toarray())

    def evolve(self, rho0, times, operators=None):
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
        if operators is None:
            operators = []

        times = np.array(times)

        if isinstance(operators, SpinOperator):
            operators = [operators]

        validate_evolve_params(rho0, times, operators)

        # Diagonalize self
        evals, evecs = self.diag()

        # Turn the density matrix in the right basis
        dim = rho0.dimension
        rho0 = rho0.basis_change(evecs).matrix

        # Same for operators
        operatorsT = np.array([o.basis_change(evecs).matrix.T for o in operators])

        # Matrix of evolution operators
        ll = -2.0j * np.pi * (evals[:, None] - evals[None, :])

        def calc_single_rho(i):
            return np.exp(ll[None, :, :] * times[i, None, None]) * rho0[None, :, :]

        result = None
        if len(operators) > 0:
            # Actually compute expectation values one at a time
            for i in range(times.shape[0]):
                rho = calc_single_rho(i)

                # This element wise multiplication then sum gives the equivalent
                # as the trace of the matrix product (without the transpose) and
                # and is faster
                single_res = np.sum(
                    rho[0, None, :, :] * operatorsT[None, :, :, :], axis=(2, 3)
                )
                if result is None:
                    result = single_res
                else:
                    result = np.concatenate(([result, single_res]), axis=0)
        else:
            sceve = evecs.T.conj()
            for i in range(times.shape[0]):
                # Just return density matrices
                result = [
                    DensityOperator(calc_single_rho(i), dim).basis_change(sceve)
                    for i in range(times.shape[0])
                ]
        return result

    def integrate_decaying(self, rho0, tau, operators=None):
        """Integrate one or more expectation values in time with decay

        Perform an integral in time from 0 to +inf of an expectation value
        computed on an evolving state, times a decay factor exp(-t/tau).
        This can be done numerically using evolve, but here is executed with
        a single evaluation, making it a lot faster.

        Arguments:
            rho0 {DensityOperator} -- Initial state
            tau {float} -- Decay time, in microseconds

        Keyword Arguments:
            operators {list} -- Operators to compute the expectation values
                                of (default: {[]})

        Returns:
            ndarray -- List of integral values

        Raises:
            TypeError -- Invalid operators
            ValueError -- Invalid values of tau or operators
            RuntimeError -- Hamiltonian is not hermitian
        """
        if operators is None:
            operators = []

        if isinstance(operators, SpinOperator):
            operators = [operators]

        validate_integrate_decaying_params(rho0, tau, operators)

        # Diagonalize self
        evals, evecs = self.diag()

        # Turn the density matrix in the right basis
        rho0 = rho0.basis_change(evecs).matrix

        ll = 2.0j * np.pi * (evals[:, None] - evals[None, :])

        # Integral operators
        intops = np.array(
            [(-o.basis_change(evecs).matrix / (ll - 1.0 / tau)).T for o in operators]
        )

        result = np.sum(rho0[None, :, :] * intops[:, :, :], axis=(1, 2))

        return result

    def fast_evolve(self, sigma_mu, times, other_dimension):
        """Compute time evolution of a muon polarisation state using this
           Hamiltonian

        Computes the evolution of a muon polarisation state under this
        Hamiltonian and returns a sequence of expectation values.

        The muon polarisation is assumed to be first in the system in this
        computation.

        Arguments:
            sigma_mu {ndarray} -- Linear combination of Pauli spin matrices in
                                  the direction of the muon
            times {ndarray} -- Times to compute the evolution for, in microseconds
            other_dimension {int} -- Combined dimension of all non-muons in the
                                     system

        Returns:
            [ndarray] -- Expectation values

        Raises:
            ValueError -- Invalid values of times
            RuntimeError -- Hamiltonian is not hermitian
        """

        times = np.array(times)

        validate_times(times)

        # Diagonalize self
        evals, evecs = self.diag()

        # Expand to correct size (Assumes the muon is the first element in
        # the system)
        # Note: May be able to adapt method from C++ implementation of Celio's
        # to avoid the use of kron here to reduce memory and speed up -
        # although limiting factor at the moment is still the eigenvalue
        # computation
        sigma_mu = sparse.kron(sigma_mu, sparse.identity(other_dimension, format="csr"))

        # Compute the value of R^dagger * sigma * R
        A = np.dot(evecs.T.conjugate(), np.dot(sigma_mu.toarray(), evecs))

        # Mod square
        A = np.power(np.abs(A), 2)
        W = 2 * np.pi * np.subtract.outer(evals, evals)

        results = parallel_fast_time_evolve(times, other_dimension, A, W)

        # Divide by 2 as by convention rest of muspinsim gives results between
        # 0.5 and -0.5
        return results * 0.5
