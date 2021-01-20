"""hamiltonian.py

A class describing a spin Hamiltonian with various terms
"""

import numpy as np

from muspinsim.spinop import SpinOperator
from muspinsim.spinsys import SpinSystem


class SingleTerm(object):

    def __init__(self, i, vector):

        self._i = i
        self._v = np.array(vector)

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


class DoubleTerm(object):

    def __init__(self, i, j, matrix):

        self._i = i
        self._j = j
        self._m = np.array(matrix)

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


class Hamiltonian(object):

    def __init__(self, spins=[]):

        self._spinsys = SpinSystem(spins)
        self._terms = []

    def add_linear_term(self, i, vector):

        if i < 0 or i >= len(self._spinsys):
            raise ValueError('Invalid index i')

        vector = np.array(vector)

        if vector.shape != (3,):
            raise ValueError('Invalid vector')

        term = SingleTerm(i, vector)
        self._terms.append(term)

    def add_bilinear_term(self, i, j, matrix):

        if i < 0 or i >= len(self._spinsys):
            raise ValueError('Invalid index i')

        if j < 0 or j >= len(self._spinsys):
            raise ValueError('Invalid index j')

        matrix = np.array(matrix)

        if matrix.shape != (3, 3):
            raise ValueError('Invalid matrix')

        term = DoubleTerm(i, j, matrix)
        self._terms.append(term)

    def remove_term(self, term):

        self._terms.remove(term)

    @property
    def spin_system(self):
        return self._spinsys

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
