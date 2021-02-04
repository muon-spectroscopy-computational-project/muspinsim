"""hamiltonian.py

A class describing a spin Hamiltonian with various terms
"""

import numpy as np
from numbers import Number
import scipy.constants as cnst

from muspinsim.constants import EFG_2_MHZ, MU_TAU
from muspinsim.spinop import SpinOperator, DensityOperator, Operator
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


class Hamiltonian(Operator):

    def __init__(self, matrix, dim=None):
        """Create an Hamiltonian

        Create an Hamiltonian from a hermitian complex matrix

        Arguments:
            matrix {ndarray} -- Matrix representation of the Hamiltonian

        Raises:
            ValueError -- Matrix isn't square or hermitian
        """

        matrix = np.array(matrix)
        n = matrix.shape[0]

        if matrix.shape != (n, n) or np.any(matrix.T.conj() != matrix):
            raise ValueError('Matrix must be square and hermitian')

        super(Hamiltonian, self).__init__(matrix, dim)

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
        intops = np.array([(-o.basis_change(evecs).matrix/(ll-1.0/tau)).T
                           for o in operators])

        result = np.sum(rho0[None, :, :]*intops[:, :, :],
                        axis=(1, 2))

        return result


class SpinHamiltonian(Hamiltonian):

    def __init__(self, spins=[]):
        """Create a SpinHamiltonian from a list of spins

        Create a SpinHamiltonian from a list of spins

        Keyword Arguments:
            spins {list} -- List of symbols of the spins to use, same rules
                            as for a SpinSystem (default: {[]})
        """

        self._spinsys = SpinSystem(spins)
        self._terms = []

        # Start with zero as a matrix
        N = np.prod(self._spinsys.dimension)
        H = np.zeros((N, N)) + .0j

        super(SpinHamiltonian, self).__init__(H, self._spinsys.dimension)

    def add_linear_term(self, i, vector, label='Single'):
        """Add to the Hamiltonian a term linear in one spin

        Add a term of the form v*S_i, where S_i is the vector of the three
        spin operators:

        [S_x, S_y, S_z]

        for spin of index i.

        Arguments:
            i {int} -- Index of the spin
            vector {ndarray} -- Vector v

        Keyword Arguments:
            label {str} -- A label to name the term (default: {'Single'})

        Raises:
            ValueError -- Invalid index or vector
        """

        if i < 0 or i >= len(self._spinsys):
            raise ValueError('Invalid index i')

        vector = np.array(vector)

        if vector.shape != (3,):
            raise ValueError('Invalid vector')

        term = SingleTerm(i, vector, label=label)
        self._terms.append(term)
        self._matrix += term.compile(self._spinsys)

    def add_bilinear_term(self, i, j, matrix, label='Double'):
        """Add to the Hamiltonian a term bilinear in two spins

        Add a term of the form S_i*M*S_j, where S_i is the vector of the three
        spin operators:

        [S_x, S_y, S_z]

        for spin of index i, and same for S_j.

        Arguments:
            i {int} -- Index of first spin
            j {int} -- Index of second spin
            matrix {ndarray} -- Matrix M

        Keyword Arguments:
            label {str} -- A label to name the term (default: {'Double'})

        Raises:
            ValueError -- Invalid index or vector
        """

        if i < 0 or i >= len(self._spinsys):
            raise ValueError('Invalid index i')

        if j < 0 or j >= len(self._spinsys):
            raise ValueError('Invalid index j')

        matrix = np.array(matrix)

        if matrix.shape != (3, 3):
            raise ValueError('Invalid matrix')

        term = DoubleTerm(i, j, matrix, label=label)
        self._terms.append(term)
        self._matrix += term.compile(self._spinsys)

    def remove_term(self, term):
        """Remove the given term, if present

        Remove a coupling term from the SpinHamiltonian

        Arguments:
            term {HamiltonianTerm} -- Term to remove
        """

        self._terms.remove(term)
        self._matrix -= term.compile(self._spinsys)

    def recalc(self):
        """Recalculate matrix from terms"""

        self._matrix *= 0

        for term in self._terms:
            self._matrix += term.compile(self._spinsys)

    @property
    def spin_system(self):
        return self._spinsys

    @property
    def spins(self):
        return self._spinsys.spins

    @property
    def terms(self):
        return list(self._terms)

    def rotate(self, rotmat):
        """Get a rotated version of the Hamiltonian

        Get a copy of this Hamiltonian that is rotated in space,
        aka, has the same terms but with matrices and vectors appropriately
        transformed. Takes in a rotation matrix defining the three vectors
        of the new axis system.

        Arguments:
            rotmat {ndarray} -- Rotation matrix

        Returns:
            SpinHamiltonian -- Rotated Hamiltonian
        """

        rH = SpinHamiltonian.__new__(SpinHamiltonian)
        rH._spinsys = self._spinsys
        rH._terms = []

        super(SpinHamiltonian, rH).__init__(self._matrix*0.0,
                                            self._spinsys.dimension)

        R = np.array(rotmat)

        for t in self._terms:
            if isinstance(t, SingleTerm):
                v = t.vector
                v = np.dot(v, R.T)
                rH.add_linear_term(t.i, v, t.label)
            elif isinstance(t, DoubleTerm):
                M = t.matrix
                M = np.linalg.multi_dot([R, M, R.T])
                rH.add_bilinear_term(t.i, t.j, M, t.label)

        return rH


class MuonHamiltonian(SpinHamiltonian):

    STATIC_FIELD_LABEL = 'B field'

    def __init__(self, spins=['e', 'mu']):
        """Create a MuonHamiltonian

        Create a MuonHamiltonian, a SpinHamiltonian specialised for muonic
        systems. It has to contain exactly one muon (mu) and can contain up to
        one electron (e). The indices of these two special particles can for
        convenience be accessed with properties .e and .mu

        Keyword Arguments:
            spins {list} -- Symbols of the spins to be part of the system.
                            Must contain at least one 'mu' 
                            (default: {['e', 'mu']})

        Raises:
            ValueError -- Invalid spins for the system
        """

        # Find the electron and muon
        self._elec_i = [i for i, s in enumerate(spins) if s == 'e']
        self._mu_i = [i for i, s in enumerate(spins) if s == 'mu']

        if len(self._mu_i) != 1:
            raise ValueError('MuonHamiltonian must contain one and only one'
                             ' muon')
        else:
            self._mu_i = self._mu_i[0]

        self._elec_i = set(self._elec_i)

        super(MuonHamiltonian, self).__init__(spins)

        # Zeeman terms
        self._Bfield = np.zeros(3)

        for i in range(len(spins)):
            self.add_linear_term(i, self._Bfield, self.STATIC_FIELD_LABEL)

        self._Bfield_terms = self.terms

    def set_B_field(self, B=[0, 0, 0]):
        """Set a magnetic field for the Hamiltonian

        Set a magnetic field (in Tesla) for the Hamiltonian, adding it to all
        the Zeeman terms.        

        Keyword Arguments:
            B {list} -- Magnetic field to set, in Tesla (default: {[0, 0, 0]})
        """

        if isinstance(B, Number):
            B = [0, 0, B]

        B = np.array(B)

        for i, t in enumerate(self._Bfield_terms):
            t._v = B*self.spin_system.gamma(i)

        # Recalculate
        self.recalc()

    def add_zeeman_term(self, i, B):
        """Add a zeeman term

        Add a single term coupling a given spin to a magnetic field, 
        in addition to the global static one.

        Arguments:
            i {int} -- Index of the spin
            B {ndarray} -- Magnetic field vector, in Tesla
        """

        B = np.array(B)
        self.add_linear_term(i, B*self.spin_system.gamma(i), 'Zeeman')

    def add_hyperfine_term(self, i, A, j=None):
        """Add a hyperfine term

        Add a hyperfine term for a given spin, provided that an electron is
        present.

        Arguments:
            i {int} -- Index of the spin (must be different from electron)
            A {[type]} -- Hyperfine tensor (in MHz)
            j {int} -- Index of the electron spin. If not specified uses the
                       one that is present, if there is one (default: None)

        Raises:
            ValueError -- Invalid index
        """

        elec_i = self.e

        if j is None:
            if len(elec_i) > 1:
                raise ValueError('Must specify an electron index in system '
                                 'with multiple electrons')
            else:
                j = list(elec_i)[0]
        else:
            if j not in elec_i:
                raise ValueError('Second index in hyperfine coupling must'
                                 ' refer to an electron')
        if i in elec_i:
            raise ValueError('First index in hyperfine coupling must'
                             ' not refer to an electron')

        self.add_bilinear_term(i, j, A, 'Hyperfine')

    def add_dipolar_term(self, i, j, r):
        """Add a dipolar term

        Add a spin-spin dipolar coupling between two distinct spins. The 
        coupling is calculated geometrically from the vector connecting them,
        in Angstrom.

        Arguments:
            i {int} -- Index of the first spin
            j {int} -- Index of the second spin
            r {ndarray} -- Vector connecting the two spins (in Angstrom)

        Raises:
            ValueError -- Raised if i == j
        """

        if i == j:
            raise ValueError('Can not set up dipolar coupling with itself')

        r = np.array(r)

        g_i = self._spinsys.gamma(i)
        g_j = self._spinsys.gamma(j)

        rnorm = np.linalg.norm(r)
        D = -(np.eye(3) - 3.0/rnorm**2.0*r[:, None]*r[None, :])
        dij = (- (cnst.mu_0*cnst.hbar*(g_i*g_j*1e12)) /
               (2*(rnorm*1e-10)**3))*1e-6  # MHz
        D *= dij

        self.add_bilinear_term(i, j, D, 'Dipolar')

    def add_quadrupolar_term(self, i, EFG):
        """Add a quadrupolar term

        Add a quadrupolar term to a nucleus with I >= 1 from its Electric
        Field Gradient tensor.

        Arguments:
            i {int} -- Index of the spin
            EFG {ndarray} --  Electric Field Gradient tensor
        """

        EFG = np.array(EFG)
        Q = self._spinsys.Q(i)
        I = self._spinsys.I(i)

        Qtens = EFG_2_MHZ*Q/(2*I*(2*I-1))*EFG

        self.add_bilinear_term(self, i, i, Qtens, 'Quadrupolar')

    @property
    def e(self):
        return self._elec_i

    @property
    def mu(self):
        return self._mu_i

    def remove_term(self, term):
        """Remove the given term, if present

        Remove a coupling term from the SpinHamiltonian

        Arguments:
            term {HamiltonianTerm} -- Term to remove. Must not be a Zeeman term
        """

        if term in self._Bfield_terms:
            raise RuntimeError('Static field terms in a MuonHamiltonian can'
                               'not be removed manually; please set B = 0')
        else:
            super(MuonHamiltonian, self).remove_term(term)

    # def reduced_hamiltonian(self, branch='up'):
    #     """Return a reduced Hamiltonian

    #     Return a reduced Hamiltonian from one containing an electron. This
    #     makes it possible to create a smaller Hamiltonian (speeding up
    #     calculations) by removing the electron, which is usually a good
    #     enough approximation for high values of the magnetic field.

    #     Keyword Arguments:
    #         branch {str} -- Electron branch to use. Can be 'up' or 'down'.
    #                         In the ideal limit (high field) should make no
    #                         difference (default: {'up'})

    #     Returns:
    #         Hamiltonian -- The reduced Hamiltonian

    #     Raises:
    #         RuntimeError -- No electron present
    #         ValueError -- Invalid branch name
    #     """

    #     if self.e is None:
    #         raise RuntimeError('Can only reduce the Hamiltonian if it '
    #                            'contains one electron')

    #     if not (branch in ('up', 'down')):
    #         raise ValueError('Branch must be either up or down')

    #     # Reshape
    #     e_i = self.e
    #     b_i = ['up', 'down'].index(branch)
    #     dim = self._spinsys.dim
    #     H = self.matrix.reshape(dim+dim)

    #     # Energy
    #     E = np.linalg.norm(self._zeeman_terms[e_i]._v)*(0.5-b_i)

    #     dred = tuple([int(np.prod(dim)/2)]*2)

    #     Haa = np.take(np.take(H, b_i, e_i+len(dim)), b_i, e_i).reshape(dred)
    #     Hab = np.take(np.take(H, 1-b_i, e_i+len(dim)), b_i, e_i).reshape(dred)
    #     Hba = np.take(np.take(H, b_i, e_i+len(dim)), 1-b_i, e_i).reshape(dred)
    #     Hbb = np.take(np.take(H, 1-b_i, e_i+len(dim)),
    #                   1-b_i, e_i).reshape(dred)

    #     invH = np.linalg.inv(Hbb-np.eye(dred[0]))

    #     Hred = Haa - np.linalg.multi_dot([Hab, invH, Hba])

    #     # Fix any residual non-hermitianity due to numerical errors
    #     Hred = (Hred+Hred.conj().T)/2.0

    #     return Hamiltonian(Hred)

    def rotate(self, rotmat):
        """Get a rotated version of the Hamiltonian

        Get a copy of this Hamiltonian that is rotated in space,
        aka, has the same terms but with matrices and vectors appropriately
        transformed. Takes in a rotation matrix defining the three vectors
        of the new axis system.

        Arguments:
            rotmat {ndarray} -- Rotation matrix

        Returns:
            MuonHamiltonian -- Rotated Hamiltonian
        """

        rH = MuonHamiltonian.__new__(MuonHamiltonian)
        rH._spinsys = self._spinsys
        rH._elec_i = self._elec_i
        rH._mu_i = self._mu_i
        rH._Bfield = self._Bfield
        rH._terms = []
        rH._Bfield_terms = []

        super(SpinHamiltonian, rH).__init__(self._matrix*0.0,
                                            self._spinsys.dimension)

        R = np.array(rotmat)

        for t in self._terms:
            if t in self._Bfield_terms:
                rH.add_linear_term(t.i, t.vector, t.label)
                rH._Bfield_terms.append(rH._terms[-1])
            elif isinstance(t, SingleTerm):
                v = t.vector
                v = np.dot(v, R.T)
                rH.add_linear_term(t.i, v, t.label)
            elif isinstance(t, DoubleTerm):
                M = t.matrix
                M = np.linalg.multi_dot([R, M, R.T])
                rH.add_bilinear_term(t.i, t.j, M, t.label)

        return rH
