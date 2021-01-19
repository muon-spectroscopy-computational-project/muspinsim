"""spinop.py

Utility class to create and manipulate spin operators
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

    def __init__(self, matrix, dim=None):
        """Create a SpinOperator object

        Create an object representing a spin operator. These can 
        be manipulated by e.g. multiplying them by a scalar or among themselves
        (equivalent to a dot product), or adding and subtracting them.

        Arguments:
            matrix {ndarray} -- Matrix describing the operator (must be a 
                                square 2D array)

        Keyword Arguments:
            dim {(int,...)} -- Tuple of the dimensions of the operator. For example, 
                               (2,2) corresponds to two 1/2 spins and a 4x4 matrix. 
                               If not specified, it's taken from the size of 
                               the matrix (default: {None})

        Raises:
            ValueError -- Any of the passed values are invalid
        """

        matrix = np.array(matrix)+.0j

        if not (matrix.shape[0] == matrix.shape[1]):
            raise ValueError('Matrix passed to SpinOperator must be square')

        if dim is None:
            dim = (matrix.shape[0],)
        elif np.prod(dim) != matrix.shape[0]:
            raise ValueError('Dimensions are not compatible with matrix')

        self._dim = dim
        self._matrix = matrix

    @classmethod
    def from_axes(self, Is=0.5, axes='x'):
        """Construct a SpinOperator from spins and axes

        Construct a SpinOperator from a list of spin values and directions. For
        example, Is=[0.5, 0.5] axes=['x', 'z'] will create a SxIz operator between
        two spin 1/2 particles.

        Keyword Arguments:
            Is {[number]} -- List of spins (must be half-integers). Can pass a 
                             number if it's only one value (default: {0.5})
            axes {[str]} -- List of axes, can pass a single character if it's 
                            only one value. Each value can be x, y, z, +, -, 
                            or 0 (for the identity operator) (default: {'x'})

        Returns:
            SpinOperator -- Operator built according to specifications

        Raises:
            ValueError -- Any of the values passed is invalid
        """

        if not hasattr(Is, '__getitem__'):
            Is = [Is]

        if not hasattr(axes, '__getitem__'):
            axes = [axes]

        if len(Is) != len(axes) or len(Is) == 0:
            raise ValueError(
                'Arrays of moments and axes must have same length > 0')

        dim = tuple(int(2*I+1) for I in Is)
        matrices = []

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

            matrices.append(o)

        M = matrices[0]
        for m in matrices[1:]:
            M = np.kron(M, m)

        return self(M, dim=dim)

    @property
    def Is(self):
        return tuple((d-1)/2.0 for d in self._dim)

    @property
    def dimension(self):
        return self._dim

    @property
    def matrix(self):
        return np.array(self._matrix)

    @property
    def is_hermitian(self):
        return np.all(self._matrix == self._matrix.conj().T)

    def clone(self):
        """Return a copy of this SpinOperator

        Return a copy of this SpinOperator

        Returns:
            SpinOperator -- Clone of this operator
        """

        ans = SpinOperator.__new__(SpinOperator)
        ans._dim = tuple(self._dim)
        ans._matrix = self.matrix

        return ans

    def __add__(self, x):

        if isinstance(x, SpinOperator):

            if self.dimension != x.dimension:
                raise ArithmeticError('Can not multiply to SpinOperators'
                                      ' with different dimensions')

            ans = self.clone()
            ans._matrix += x._matrix

            return ans

        raise TypeError('Unsupported operation for SpinOperator')

    def __sub__(self, x):

        if isinstance(x, SpinOperator):

            if self.dimension != x.dimension:
                raise ArithmeticError('Can not multiply to SpinOperators'
                                      ' with different dimensions')

            ans = self.clone()
            ans._matrix -= x._matrix

            return ans

        raise TypeError('Unsupported operation for SpinOperator')

    def __mul__(self, x):

        if isinstance(x, SpinOperator):

            if self.dimension != x.dimension:
                raise ArithmeticError('Can not multiply to SpinOperators'
                                      ' with different dimensions')

            ans = self.clone()
            ans._matrix = np.dot(ans._matrix, x._matrix)

            return ans

        elif isinstance(x, Number):

            ans = self.clone()
            ans._matrix *= x

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

    def __eq__(self, x):

        if not isinstance(x, SpinOperator):
            return False

        if self.dimension != x.dimension:
            return False

        return np.all(self._matrix == x._matrix)

    def kron(self, x):
        """Tensor product between this and another SpinOperator

        Performs a tensor product between this and another SpinOperator,
        raising the overall rank of the tensor they represent.

        Arguments:
            x {SpinOperator} -- Other operator

        Returns:
            SpinOperator -- Result

        Raises:
            ValueError -- Thrown if x is not the right type of object
        """

        if not isinstance(x, SpinOperator):
            raise ValueError('Can only perform Kronecker product with'
                             ' another SpinOperator')

        # Doing it this way saves some time
        ans = SpinOperator.__new__(SpinOperator)
        ans._dim = self._dim + x._dim
        ans._matrix = np.kron(self._matrix, x._matrix)

        return ans
