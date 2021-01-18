"""spinsys.py

A class to hold a given spin system, defined by specific nuclei
"""

import numpy as np

from muspinsim.constants import gyromagnetic_ratio, spin, quadrupole_moment
from muspinsim.spinop import SpinOperator


class SpinSystem(object):

    def __init__(self, spins=[]):

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

            opdict = {a: SpinOperator(Is[-1], a) for a in 'xyz+-0'}

            operators.append(opdict)

        self._gammas = np.array(gammas)
        self._Qs = np.array(Qs)

        self._operators = operators

    def gamma(self, i):
        return self._gammas[i]

    def Q(self, i):
        return self._Qs[i]

    def operator(self, terms={}):

        ops = [self._operators[i][terms.get(i, '0')]
               for i in range(len(self))]

        M = ops[0]

        for i in range(1, len(ops)):
            M = M.kron(ops[i])

        return M

    def __len__(self):
        return len(self._gammas)
