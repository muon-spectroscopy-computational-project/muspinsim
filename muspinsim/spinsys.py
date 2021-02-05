"""spinsys.py

A class to hold a given spin system, defined by specific nuclei
"""

import numpy as np
from copy import deepcopy

from muspinsim.utils import Clonable
from muspinsim.spinop import SpinOperator
from muspinsim.hamiltonian import Hamiltonian
from muspinsim.constants import gyromagnetic_ratio, spin, quadrupole_moment


class InteractionTerm(Clonable):

    def __init__(self, spinsys, indices=[], tensor=0, label=None):

        self._spinsys = spinsys
        self._indices = np.array(indices)
        self._tensor = np.array(tensor)
        self._label = 'Term' if label is None else label

        if np.any(np.array(self._tensor.shape) != 3):
            raise ValueError('Tensor is not fully three-dimensional')

        total_op = None
        d = len(self._tensor.shape)

        if d > 0:
            index_tuples = np.indices(self._tensor.shape).reshape((d, -1)).T
        else:
            index_tuples = [[]]

        for ii in index_tuples:
            op = self._spinsys.operator({ind: 'xyz'[ii[i]]
                                         for i, ind in
                                         enumerate(self._indices)}
                                        )*self._tensor[tuple(ii)]
            if total_op is None:
                total_op = op
            else:
                total_op += op

        self._operator = total_op

    @property
    def label(self):
        return self._label

    @property
    def indices(self):
        return tuple(self._indices)

    @property
    def tensor(self):
        return np.array(self._tensor)

    @property
    def operator(self):
        return self._operator.clone()

    def __repr__(self):
        return self.label


class SingleTerm(InteractionTerm):

    def __init__(self, spinsys, i, vector, label='Single'):

        super(SingleTerm, self).__init__(spinsys, [i], vector, label)

    @property
    def i(self):
        return self._indices[0]

    def rotate(self, rotmat):

        R = np.array(rotmat)
        v = self._tensor
        v = np.dot(v, R.T)

        rt = self.clone()
        rt._tensor = v

        return rt

    def __repr__(self):
        return '{0} {{ S_{1} * {2} }}'.format(self._label, self.i,
                                              self._tensor)


class DoubleTerm(InteractionTerm):

    def __init__(self, spinsys, i, j, matrix, label='Double'):

        super(DoubleTerm, self).__init__(spinsys, [i, j], matrix, label)

    @property
    def i(self):
        return self._indices[0]

    @property
    def j(self):
        return self._indices[1]

    def rotate(self, rotmat):

        R = np.array(rotmat)
        M = self._tensor
        M = np.linalg.multi_dot([R, M, R.T])

        rt = self.clone()
        rt._tensor = M

        return rt

    def __repr__(self):
        return '{0} {{ S_{1} * [{2} {3} {4}] * S_{5} }}'.format(self._label,
                                                                self.i,
                                                                *self._tensor,
                                                                self.j)


class SpinSystem(Clonable):

    def __init__(self, spins=[]):
        """Create a SpinSystem object

        Create an object representing a system of particles with spins (muons,
        electrons and atomic nuclei) and holding their operators.

        Keyword Arguments:
            spins {list} -- List of symbols representing the various particles.
                            Each element can be 'e' (electron), 'mu' (muon) a
                            chemical symbol, or a (str, int) tuple with a
                            chemical symbol and an isotope (default: {[]})
        """

        gammas = []
        Qs = []
        Is = []
        operators = []

        for s in spins:
            if isinstance(s, tuple):
                el, iso = s
            else:
                el, iso = s, None

            gammas.append(gyromagnetic_ratio(el, iso))
            Qs.append(quadrupole_moment(el, iso))
            Is.append(spin(el, iso))

            opdict = {a: SpinOperator.from_axes(Is[-1], a) for a in 'xyz+-0'}

            operators.append(opdict)

        self._spins = list(spins)
        self._gammas = np.array(gammas)
        self._Qs = np.array(Qs)
        self._Is = np.array(Is)
        self._dim = tuple((2*self._Is+1).astype(int))

        self._operators = operators

        self._terms = []

    @property
    def spins(self):
        return list(self._spins)

    @property
    def dimension(self):
        return self._dim

    def add_term(self, indices, tensor, label='Term'):
        """Add to the spin system a generic interaction term

        Add a term of the form T*S_i*S_j*S_k*..., where S_i is the vector of 
        the three spin operators:

        [S_x, S_y, S_z]

        for spin of index i.

        Arguments:
            indices {[int]} -- Indices of spins appearing in the term
            tensor {ndarray} -- Tensor with n dimensions (n = len(indices)), 
                                each of length 3, describing the interaction.

        Keyword Arguments:
            label {str} -- A label to name the term (default: {'Term'})

        Returns:
            term {InteractionTerm} -- The term just created

        Raises:
            ValueError -- Invalid index or vector
        """

        for i in indices:
            if i < 0 or i >= len(self._spins):
                raise ValueError('Invalid index i')

        tensor = np.array(tensor)

        term = InteractionTerm(self, indices, tensor, label=label)
        self._terms.append(term)

        return term

    def add_linear_term(self, i, vector, label='Single'):
        """Add to the spin system a term linear in one spin

        Add a term of the form v*S_i, where S_i is the vector of the three
        spin operators:

        [S_x, S_y, S_z]

        for spin of index i.

        Arguments:
            i {int} -- Index of the spin
            vector {ndarray} -- Vector v

        Keyword Arguments:
            label {str} -- A label to name the term (default: {'Single'})

        Returns:
            term {SingleTerm} -- The term just created

        Raises:
            ValueError -- Invalid index or vector
        """

        if i < 0 or i >= len(self._spins):
            raise ValueError('Invalid index i')

        vector = np.array(vector)

        term = SingleTerm(self, i, vector, label=label)
        self._terms.append(term)

        return term

    def add_bilinear_term(self, i, j, matrix, label='Double'):
        """Add to the spin system a term bilinear in two spins

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

        Returns:
            term {DoubleTerm} -- The term just created

        Raises:
            ValueError -- Invalid index or vector
        """

        if i < 0 or i >= len(self._spins):
            raise ValueError('Invalid index i')

        if j < 0 or j >= len(self._spins):
            raise ValueError('Invalid index j')

        matrix = np.array(matrix)

        term = DoubleTerm(self, i, j, matrix, label=label)
        self._terms.append(term)

        return term

    def remove_term(self, term):
        """Remove a term from the spin system

        Remove an interaction term from this spin system.

        Arguments:
            term {InteractionTerm} -- Term to remove

        Raises:
            ValueError -- The term is not contained in this system
        """

        self._terms.remove(term)

    def gamma(self, i):
        """Returns the gyromagnetic ratio of a given particle

        Arguments:
            i {int} -- Index of the particle

        Returns:
            float -- Gyromagnetic ratio in MHz/T
        """
        return self._gammas[i]

    def Q(self, i):
        """Returns the quadrupole moment of a given particle

        Arguments:
            i {int} -- Index of the particle

        Returns:
            float -- Quadrupole moment in Barn
        """
        return self._Qs[i]

    def I(self, i):
        """Returns the spin of a given particle

        Arguments:
            i {int} -- Index of the particle

        Returns:
            float -- Spin in units of hbar
        """

        return self._Is[i]

    def operator(self, terms={}):
        """Return an operator for this spin system

        Return a SpinOperator for this system containing the specified terms.        

        Keyword Arguments:
            terms {dict} -- A dictionary of terms to include. The keys should
                            indices of particles and the values should be 
                            symbols indicating one spin operator (either x, y,
                            z, +, - or 0). Wherever not specified, the identity
                            operaror is applied (default: {{}})

        Returns:
            SpinOperator -- The requested operator
        """

        ops = [self._operators[i][terms.get(i, '0')]
               for i in range(len(self))]

        M = ops[0]

        for i in range(1, len(ops)):
            M = M.kron(ops[i])

        return M

    def rotate(self, rotmat=np.eye(3)):

        # Make a clone
        rssys = self.clone()

        # Edit the terms
        try:
            rssys._terms = [t.rotate(rotmat) for t in rssys._terms]
        except AttributeError:
            raise RuntimeError('Can only rotate SpinSystems containing Single'
                               ' or Double terms')

        return rssys

    @property
    def hamiltonian(self):

        H = np.sum([t.operator.matrix for t in self._terms], axis=0)
        H = Hamiltonian(H, dim=self.dimension)

        return H

    def __len__(self):
        return len(self._gammas)
