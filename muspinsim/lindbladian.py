"""lindbladian.py

SuperOperator class for Lindbladian, used in open quantum dynamics
"""

import numpy as np
from muspinsim.celio import CelioHamiltonian

from muspinsim.hamiltonian import Hamiltonian
from muspinsim.spinop import SuperOperator, SpinOperator, DensityOperator
from muspinsim.validation import (
    validate_evolve_params,
    validate_integrate_decaying_params,
)


class Lindbladian(SuperOperator):
    @classmethod
    def from_hamiltonian(self, H, dissipators=[]):

        if isinstance(H, CelioHamiltonian):
            raise NotImplementedError(
                "Linbladian is not implemented for Celio's method"
            )
        if not isinstance(H, Hamiltonian):
            raise ValueError("Must use Hamiltonian to create Lindbladian")

        L = -1.0j * SuperOperator.commutator(H)
        L = self(L.matrix, L.dimension)

        for (A, gamma) in dissipators:
            L.add_dissipative_term(A, gamma)

        return L

    def add_dissipative_term(self, A, gamma=1.0):

        AA = A.dagger() * A
        Ld = gamma * (SuperOperator.bracket(A) - 0.5 * SuperOperator.anticommutator(AA))
        if Ld.dimension != self.dimension:
            raise ValueError("Invalid dissipation operator for this Lindbladian")

        self._matrix += Ld.matrix

    def evolve(self, rho0, times, operators=[]):
        """Time evolution of a state under this Lindbladian

        Perform an evolution of a state described by a DensityOperator under
        this Lindbladian and return either a sequence of DensityOperators or
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

        dim = rho0.dimension
        if self.dimension != dim * 2:
            raise ValueError("Incompatible rho0 dimension")

        if any([self.dimension != o.dimension * 2 for o in operators]):
            raise ValueError("Incompatible measure operator dimension")

        # Start by building the matrix
        L = self.matrix.toarray()

        # Diagonalize it
        evals, revecs = np.linalg.eig(L)

        # Vec-ing the density matrix
        rho0 = rho0.matrix.toarray().reshape(
            -1,
        )
        rho0 = np.linalg.solve(revecs, rho0)
        # And the operators
        operatorsT = np.array(
            [np.dot(o.matrix.T.toarray().reshape((-1,)), revecs) for o in operators]
        )

        rho = np.exp(2.0 * np.pi * evals[None, :] * times[:, None]) * rho0[None, :]

        if len(operators) > 0:
            # Expectation values
            result = np.sum(operatorsT[None, :, :] * rho[:, None, :], axis=-1)
        else:
            # Density matrices
            n = np.prod(dim)
            result = [
                DensityOperator(np.dot(revecs, r).reshape(n, n), dim) for r in rho
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

        # Start by building the matrix
        L = self.matrix

        # Diagonalize it

        # sparse matricies

        # need to convert this to use sparse matrices instead
        evals, revecs = np.linalg.eig(L.toarray())

        # evals, revecs = linalg.eigsh(self._matrix, k=self._matrix.shape[0]-2)
        # idx = evals.argsort()
        # evals = eigval[idx]
        # revecs = eigvec[:,idx]

        # Vec-ing the density matrix
        rho0 = rho0.matrix.toarray().reshape((-1,))
        rho0 = np.linalg.solve(revecs, rho0)

        # And the operators
        intops = np.array(
            [
                np.dot(o.matrix.T.toarray().reshape((-1,)), revecs)
                / (1.0 / tau - 2.0 * np.pi * evals)
                for o in operators
            ]
        )

        result = np.sum(rho0[None, :] * intops[:, :], axis=1)

        return result
