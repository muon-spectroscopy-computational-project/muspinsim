"""lindbladian.py

SuperOperator class for Lindbladian, used in open quantum dynamics
"""

import numpy as np

from muspinsim.hamiltonian import Hamiltonian
from muspinsim.spinop import SuperOperator, SpinOperator, DensityOperator


class Lindbladian(SuperOperator):

    @classmethod
    def from_hamiltonian(self, H, dissipators=[]):

        if not isinstance(H, Hamiltonian):
            raise ValueError('Must use Hamiltonian to create Lindbladian')

        L = -1.0j*SuperOperator.commutator(H)
        L = self(L.matrix, L.dimension)

        for (A, gamma) in dissipators:
            L.add_dissipative_term(A, gamma)

        return L

    def add_dissipative_term(self, A, gamma=1.0):

        AA = A.dagger()*A
        Ld = gamma*(SuperOperator.bracket(A) -
                    0.5*SuperOperator.anticommutator(AA))
        if Ld.dimension != self.dimension:
            raise ValueError('Invalid dissipation operator for this '
                             'Lindbladian')

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

        if not isinstance(rho0, DensityOperator):
            raise TypeError('rho0 must be a valid DensityOperator')

        times = np.array(times)

        if len(times.shape) != 1:
            raise ValueError(
                'times must be an array of values in microseconds')

        if isinstance(operators, SpinOperator):
            operators = [operators]
        if not all([isinstance(o, SpinOperator) for o in operators]):
            raise ValueError('operators must be a SpinOperator or a list'
                             ' of SpinOperator objects')

        dim = rho0.dimension
        if self.dimension != dim*2:
            raise ValueError('Incompatible rho0 dimension')

        if any([self.dimension != o.dimension*2 for o in operators]):
            raise ValueError('Incompatible measure operator dimension')

        # Start by building the matrix
        L = self.matrix

        # Diagonalize it
        evals, revecs = np.linalg.eig(L)
        # Left eigenvectors (transposed)
        levecs = np.linalg.inv(revecs)

        # Vec-ing the density matrix
        rho0 = rho0.matrix.reshape((-1,))
        rho0 = np.dot(levecs, rho0)
        # And the operators
        operatorsT = np.array([np.dot(o.matrix.T.reshape((-1,)), revecs)
                               for o in operators])

        rho = np.exp(2.0*np.pi*evals[None, :]*times[:, None])*rho0[None, :]

        if len(operators) > 0:
            # Expectation values
            result = np.sum(operatorsT[None, :, :]*rho[:, None, :], axis=-1)
        else:
            # Density matrices
            result = [DensityOperator(np.dot(revecs, r), dim) for r in rho]

        return result
