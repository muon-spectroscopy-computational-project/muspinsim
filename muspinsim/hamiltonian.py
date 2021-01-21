"""hamiltonian.py

A class describing a spin Hamiltonian with various terms
"""

import numpy as np
from numbers import Number
import scipy.constants as cnst

from muspinsim.constants import EFG_2_MHZ
from muspinsim.spinop import SpinOperator, DensityOperator
from muspinsim.spinsys import SpinSystem


class HamiltonianTerm(object):

    def __init__(self, label=None):

        self._label = 'Term' if label is None else label

    @property
    def label(self):
        return self._label

    def __repr__(self):
        return self.label


class SingleTerm(HamiltonianTerm):

    def __init__(self, i, vector, label='Single'):

        self._i = i
        self._v = np.array(vector)

        super(SingleTerm, self).__init__(label)

    @property
    def i(self):
        return self._i

    @property
    def vector(self):
        return np.array(self._v)

    def compile(self, spinsys):

        matrices = []
        for i in range(3):
            op = spinsys.operator({self.i: 'xyz'[i]})*self._v[i]
            matrices.append(op.matrix)

        return np.sum(matrices, axis=0)

    def __repr__(self):
        return '{0} {{ S_{1} * {2} }}'.format(self._label, self.i, self._v)


class DoubleTerm(HamiltonianTerm):

    def __init__(self, i, j, matrix, label='Double'):

        self._i = i
        self._j = j
        self._m = np.array(matrix)

        super(DoubleTerm, self).__init__(label)

    @property
    def i(self):
        return self._i

    @property
    def j(self):
        return self._j

    @property
    def matrix(self):
        return np.array(self._m)

    def compile(self, spinsys):

        matrices = []
        for i in range(3):
            for j in range(3):
                op = spinsys.operator({self.i: 'xyz'[i],
                                       self.j: 'xyz'[j]})*self._m[i, j]
                matrices.append(op.matrix)

        return np.sum(matrices, axis=0)

    def __repr__(self):
        return '{0} {{ S_{1} * [{2} {3} {4}] * S_{5} }}'.format(self._label,
                                                                self.i,
                                                                *self._m,
                                                                self.j)


class Hamiltonian(object):

    def __init__(self, spins=[]):

        self._spinsys = SpinSystem(spins)
        self._terms = []

    def add_linear_term(self, i, vector, label='Single'):

        if i < 0 or i >= len(self._spinsys):
            raise ValueError('Invalid index i')

        vector = np.array(vector)

        if vector.shape != (3,):
            raise ValueError('Invalid vector')

        term = SingleTerm(i, vector, label=label)
        self._terms.append(term)

    def add_bilinear_term(self, i, j, matrix, label='Double'):

        if i < 0 or i >= len(self._spinsys):
            raise ValueError('Invalid index i')

        if j < 0 or j >= len(self._spinsys):
            raise ValueError('Invalid index j')

        matrix = np.array(matrix)

        if matrix.shape != (3, 3):
            raise ValueError('Invalid matrix')

        term = DoubleTerm(i, j, matrix, label=label)
        self._terms.append(term)

    def remove_term(self, term):

        self._terms.remove(term)

    @property
    def spin_system(self):
        return self._spinsys

    @property
    def spins(self):
        return self._spinsys.spins

    @property
    def terms(self):
        return list(self._terms)

    @property
    def matrix(self):

        # Compile the full Hamiltonian matrix
        term_matrices = []

        for t in self._terms:
            term_matrices.append(t.compile(self._spinsys))

        return np.sum(term_matrices, axis=0)

    def rotate(self, rotmat):
        """Get a rotated version of the Hamiltonian

        Get a copy of this Hamiltonian that is rotated in space,
        aka, has the same terms but with matrices and vectors appropriately
        transformed. Takes in a rotation matrix defining the three vectors
        of the new axis system.

        Arguments:
            rotmat {ndarray} -- Rotation matrix

        Returns:
            Hamiltonian -- Rotated Hamiltonian
        """

        rH = Hamiltonian(self._spinsys.spins)
        R = rotmat

        for t in self._terms:
            if isinstance(t, SingleTerm):
                v = t.vector
                v = np.dot(v, R.T)
                rH.add_linear_term(t.i, v)
            elif isinstance(t, DoubleTerm):
                M = t.matrix
                M = np.linalg.multi_dot([R, M, R.T])
                rH.add_bilinear_term(t.i, t.j, M)

        return rH

    def evolve(self, rho0, times, operators=[]):

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

        # Start by building the matrix
        H = self.matrix

        # Sanity check - should never happen
        if not np.all(H == H.T.conj()):
            raise RuntimeError('Hamiltonian is not hermitian')

        # Diagonalize it
        evals, evecs = np.linalg.eigh(H)

        # Turn the density matrix in the right basis
        dim = rho0.dimension
        rho0 = rho0.basis_change(evecs).matrix

        # Same for operators
        operatorsT = np.array([o.basis_change(evecs).matrix.T
                               for o in operators])

        # Matrix of evolution operators
        ll = -2.0j*np.pi*(evals[:, None]-evals[None, :])
        rho = np.exp(ll[None, :, :]*times[:, None, None])*rho0[None, :, :]

        # Now, return values
        if len(operators) > 0:
            # Actually compute expectation values
            result = np.sum(rho[:, None, :, :]*operatorsT[None, :, :, :],
                            axis=(2, 3))
        else:
            # Just return density matrices
            sceve = evecs.T.conj()
            result = [DensityOperator(r, dim).basis_change(sceve) for r in rho]

        return result

    def integrate_decaying(self, rho0, tau, operators=[]):

        if not isinstance(rho0, DensityOperator):
            raise TypeError('rho0 must be a valid DensityOperator')

        if not (isinstance(tau, Number) and np.isreal(tau) and tau > 0):
            raise ValueError('tau must be a real number > 0')

        if isinstance(operators, SpinOperator):
            operators = [operators]
        if not all([isinstance(o, SpinOperator) for o in operators]):
            raise ValueError('operators must be a SpinOperator or a list'
                             ' of SpinOperator objects')

        H = self.matrix

        # Sanity check - should never happen
        if not np.all(H == H.T.conj()):
            raise RuntimeError('Hamiltonian is not hermitian')

        # Diagonalize it
        evals, evecs = np.linalg.eigh(H)

        # Turn the density matrix in the right basis
        dim = rho0.dimension
        rho0 = rho0.basis_change(evecs).matrix

        ll = 2.0j*np.pi*(evals[:, None]-evals[None, :])

        # Integral operators
        intops = np.array([(-o.basis_change(evecs).matrix/(ll*tau-1.0)).T
                           for o in operators])

        result = np.sum(rho0[None, :, :]*intops[:, :, :],
                        axis=(1, 2))

        return result


class MuonHamiltonian(Hamiltonian):

    def __init__(self, other_spins=[]):

        spins = ['e', 'mu'] + other_spins
        super(MuonHamiltonian, self).__init__(spins)

        # Zeeman terms
        self._Bfield = np.zeros(3)

        for i in range(len(spins)):
            self.add_linear_term(i, self._Bfield, 'Zeeman')

        self._zeeman_terms = self.terms

    def set_B_field(self, B=[0, 0, 0]):

        if isinstance(B, Number):
            B = [0, 0, B]

        B = np.array(B)

        for i, t in enumerate(self._zeeman_terms):
            t._v = B*self.spin_system.gamma(i)

    def add_hyperfine_term(self, i, A):

        if self.spins[i] == 'e':
            raise ValueError('Can not set a hyperfine coupling for an '
                             'electron')

        self.add_bilinear_term(0, i, A, 'Hyperfine')

    def add_dipolar_term(self, i, j, r):

        g_i = self._spinsys.gamma(i)
        g_j = self._spinsys.gamma(j)

        rnorm = np.linalg.norm(r)
        D = -(np.eye(3) - 3.0/rnorm**2.0*r[:, None]*r[None, :])
        dij = (- (cnst.mu_0*cnst.hbar*(g_i*g_j*1e12)) /
               (2*(rnorm*1e-10)**3))*1e-6  # MHz
        D *= dij

        self.add_bilinear_term(i, j, D, 'Dipolar')

    def add_quadrupolar_term(self, i, EFG):

        EFG = np.array(EFG)
        Q = self._spinsys.Q(i)
        I = self._spinsys.I(i)

        Qtens = EFG_2_MHZ*Q/(2*I*(2*I-1))*EFG

        self.add_bilinear_term(self, i, i, Qtens)

    def remove_term(self, term):

        if term in self._zeeman_terms:
            raise RuntimeError('Zeeman terms in a MuonHamiltonian can not be '
                               'removed manually; please set B = 0')
        else:
            super(MuonHamiltonian, self).remove_term(term)
