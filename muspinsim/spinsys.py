"""spinsys.py

A class to hold a given spin system, defined by specific nuclei
"""

import numpy as np

from muspinsim.constants import gyromagnetic_ratio, spin, quadrupole_moment
from muspinsim.spinop import SpinOperator


class InteractionTerm(object):

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

    def __repr__(self):
        return '{0} {{ S_{1} * [{2} {3} {4}] * S_{5} }}'.format(self._label,
                                                                self.i,
                                                                *self._tensor,
                                                                self.j)


class SpinSystem(object):

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

    @property
    def spins(self):
        return list(self._spins)

    @property
    def dimension(self):
        return self._dim

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

    def __len__(self):
        return len(self._gammas)
