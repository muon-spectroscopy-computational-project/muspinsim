"""spinop.py

Utility functions to create and manipulate spin operators
"""

import numpy as np
from numbers import Number


def _Sp(mvals):
    return np.diag(0.5*(np.cumsum(2*mvals)[:-1]**0.5), k=1)+.0j


def _Sm(mvals):
    return _Sp(mvals).T


def _Sx(mvals):
    o = _Sp(mvals)
    return o + o.T


def _Sy(mvals):
    o = _Sp(mvals)
    return 1.0j*(o.T-o)


def _Sz(mvals):
    return np.diag(mvals)+.0j


def _S0(mvals):
    return np.eye(len(mvals))+.0j


class SpinOperator(object):

    def __init__(self, Is=0.5, axes='x', prefactor=1.0):

        if not hasattr(Is, '__getitem__'):
            Is = [Is]

        if not hasattr(axes, '__getitem__'):
            axes = [axes]

        if len(Is) != len(axes):
            raise ValueError(
                'Arrays of moments and axes must have same length')

        self._Is = Is
        self._prefac = prefactor
        self._dim = tuple([int(2*I+1) for I in Is])
        self._matrices = []

        for I, axis in zip(Is, axes):

            if I % 0.5 or I < 0.5:
                raise ValueError('{0} is not a valid spin value'.format(I))

            if not (axis in 'xyz+-0'):
                raise ValueError('{0} is not a valid spin axis'.format(axis))

            mvals = np.linspace(I, -I, int(2*I+1))

            o = {
                'x': _Sx,
                'y': _Sy,
                'z': _Sz,
                '+': _Sp,
                '-': _Sm,
                '0': _S0
            }[axis](mvals)

            self._matrices.append(o)

    @property
    def Is(self):
        return np.array(self._Is)

    @property
    def dimension(self):
        return self._dim

    @property
    def matrices(self):
        return [np.array(m) for m in self._matrices]

    @property
    def full_matrix(self):

        M = self._matrices[0]

        for m in self._matrices[1:]:
            M = np.kron(M, m)

        return self._prefac*M

    def clone(self):

        ans = SpinOperator.__new__(SpinOperator)
        ans._Is = list(self.Is)
        ans._prefac = self._prefac
        ans._dim = tuple(self._dim)
        ans._matrices = self.matrices

        return ans

    def __add__(self, x):

        if isinstance(x, SpinOperator):

            if self.dimension != x.dimension:
                raise ArithmeticError('Can not multiply to SpinOperators'
                                      ' with different dimensions')

            ans = self.clone()
            ans._prefac = 1.0
            ans._matrices = [(self._prefac*m1+x._prefac*m2)
                             for m1, m2 in zip(self._matrices, x._matrices)]

            return ans

        raise TypeError('Unsupported operation for SpinOperator')

    def __sub__(self, x):

        if isinstance(x, SpinOperator):

            if self.dimension != x.dimension:
                raise ArithmeticError('Can not multiply to SpinOperators'
                                      ' with different dimensions')

            x = x.clone()
            x._prefac *= -1

            return self + x

        raise TypeError('Unsupported operation for SpinOperator')

    def __mul__(self, x):

        if isinstance(x, SpinOperator):

            if self.dimension != x.dimension:
                raise ArithmeticError('Can not multiply to SpinOperators'
                                      ' with different dimensions')

            ans = self.clone()
            ans._prefac *= x._prefac
            ans._matrices = [np.dot(m1, m2)
                             for m1, m2 in zip(self._matrices, x._matrices)]

            return ans

        elif isinstance(x, Number):

            ans = self.clone()
            ans._prefac = ans._prefac*x

            return ans

        raise TypeError('Unsupported operation for SpinOperator')

    def __rmul__(self, x):

        if isinstance(x, Number):
            return self.__mul__(x)
        elif isinstance(x, SpinOperator):
            return x*self

        raise TypeError('Unsupported operation for SpinOperator')

    def __truediv__(self, x):

        if isinstance(x, Number):
            x = 1.0/x
            return self*x

        raise TypeError('Unsupported operation for SpinOperator')

    def kron(self, x):

        if not isinstance(x, SpinOperator):
            raise ValueError('Can only perform Kronecker product with'
                             ' another SpinOperator')

        # Doing it this way saves some time
        ans = SpinOperator.__new__(SpinOperator)

        ans._Is = list(self._Is) + list(x._Is)
        ans._prefac = self._prefac*x._prefac
        ans._dim = self._dim + x._dim
        ans._matrices = self.matrices + x.matrices

        return ans
