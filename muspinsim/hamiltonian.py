"""hamiltonian.py

A class describing a spin Hamiltonian with various terms
"""

import numpy as np

from scipy import sparse

from muspinsim.cython import parallel_fast_time_evolve
from muspinsim.spinop import SpinOperator, DensityOperator, Operator, Hermitian
from muspinsim.validation import (
    validate_evolve_params,
    validate_integrate_decaying_params,
    validate_times,
)


class Hamiltonian(Operator, Hermitian):
    def __init__(self, matrix, dim=None):
        """Create an Hamiltonian

        Create an Hamiltonian from a hermitian complex matrix

        Arguments:
            matrix {ndarray} -- Matrix representation of the Hamiltonian

        Raises:
            ValueError -- Matrix isn't square or hermitian
        """

        super(Hamiltonian, self).__init__(matrix, dim)

    @classmethod
    def from_spin_operator(self, spinop):
        return self(spinop.matrix, spinop.dimension)

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

        times = np.array(times)

        if isinstance(operators, SpinOperator):
            operators = [operators]

        validate_evolve_params(rho0, times, operators)

        # Diagonalize self
        evals, evecs = self.diag()

        # Turn the density matrix in the right basis
        dim = rho0.dimension
        rho0 = rho0.basis_change(evecs).matrix.toarray()

        # Same for operators
        operatorsT = np.array(
            [o.basis_change(evecs).matrix.T.toarray() for o in operators]
        )

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

    def integrate_decaying(self, rho0, tau, operators=[]):
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

        if isinstance(operators, SpinOperator):
            operators = [operators]

        validate_integrate_decaying_params(rho0, tau, operators)

        # Diagonalize self
        evals, evecs = self.diag()

        # Turn the density matrix in the right basis
        rho0 = rho0.basis_change(evecs).matrix.toarray()

        ll = 2.0j * np.pi * (evals[:, None] - evals[None, :])

        # Integral operators
        intops = np.array(
            [
                (-o.basis_change(evecs).matrix.toarray() / (ll - 1.0 / tau)).T
                for o in operators
            ]
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
