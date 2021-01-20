"""spinsys.py

A class to hold a given spin system, defined by specific nuclei
"""

import numpy as np

from muspinsim.constants import gyromagnetic_ratio, spin, quadrupole_moment
from muspinsim.spinop import SpinOperator


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

        self._operators = operators

    @property
    def spins(self):
        return list(self._spins)

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
