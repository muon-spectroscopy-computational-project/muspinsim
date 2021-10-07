"""spinop.py

Utility class to create and manipulate spin operators
"""

import numpy as np
from numbers import Number
from muspinsim.utils import Clonable


def _mvals(I):
    return np.linspace(I, -I, int(2 * I + 1))


def _Sp(mvals):
    return 2 * np.diag(0.5 * (np.cumsum(2 * mvals)[:-1] ** 0.5), k=1) + 0.0j


def _Sm(mvals):
    return _Sp(mvals).T


def _Sx(mvals):
    o = _Sp(mvals)
    return 0.5 * (o + o.T)


def _Sy(mvals):
    o = _Sp(mvals)
    return 0.5j * (o.T - o)


def _Sz(mvals):
    return np.diag(mvals) + 0.0j


def _S0(mvals):
    return np.eye(len(mvals)) + 0.0j


class Hermitian(object):
    """A helper mixin for operators that are also Hermitian"""

    def __init__(self):

        if not self.is_hermitian:
            raise ValueError("Operator must be hermitian")

    def diag(self):
        """Diagonalise the operator

        Return eigenvalues and corresponding eigenvectors for this operator.

        Returns:
            (ndarray, ndarray) -- Eigenvalues and eigenvector matrix, as
                                  returned by numpy.linalg.eigh
        """

        try:
            dd = self._diagdata
            if np.all(dd["matrix"] == self._matrix):
                return dd["eigh"]
        except AttributeError:
            pass

        eigh = np.linalg.eigh(self._matrix)

        self._diagdata = {"matrix": self._matrix.copy(), "eigh": eigh}

        return eigh


class Operator(Clonable):
    def __init__(self, matrix, dim=None, hermtol=1e-6):
        """Create a Operator object

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
            hermtol {float} -- Tolerance used to check for hermitianity of the
                               matrix (default: {1e-6})

        Raises:
            ValueError -- Any of the passed values are invalid
        """

        matrix = np.array(matrix) + 0.0j

        if not (matrix.shape[0] == matrix.shape[1]):
            raise ValueError("Matrix passed to Operator must be square")

        if dim is None:
            dim = (matrix.shape[0],)
        elif np.prod(dim) != matrix.shape[0]:
            raise ValueError("Dimensions are not compatible with matrix")

        self._dim = tuple(dim)
        self._matrix = matrix
        self._htol = hermtol

        super(Operator, self).__init__()

    @property
    def Is(self):
        return tuple((d - 1) / 2.0 for d in self._dim)

    @property
    def dimension(self):
        return self._dim

    @property
    def N(self):
        return np.prod(self._dim)

    @property
    def matrix(self):
        return np.array(self._matrix)

    @property
    def is_hermitian(self):
        return np.all(np.abs(self._matrix - self._matrix.conj().T) < self._htol)

    def dagger(self):
        """Return the transpose conjugate of this Operator

        Return the transpose conjugate of this Operator

        Returns:
            Operator -- Transpose conjugate of this operator
        """

        MyClass = self.__class__
        ans = MyClass.__new__(MyClass)
        ans._dim = tuple(self._dim)
        ans._matrix = self.matrix.conj().T

        return ans

    def __add__(self, x):

        if isinstance(x, Operator):

            if self.dimension != x.dimension:
                raise ArithmeticError(
                    "Can not add to Operators" " with different dimensions"
                )

            ans = self.clone()
            ans._matrix += x._matrix

            return ans

        elif isinstance(x, Number):

            ans = self.clone()
            ans._matrix += np.eye(ans._matrix.shape[0]) * x

            return ans

        raise TypeError("Unsupported operation for Operator")

    def __sub__(self, x):

        if isinstance(x, Operator):

            if self.dimension != x.dimension:
                raise ArithmeticError(
                    "Can not subtract to Operators" " with different dimensions"
                )

            ans = self.clone()
            ans._matrix -= x._matrix

            return ans

        elif isinstance(x, Number):

            return self + (-x)

        raise TypeError("Unsupported operation for Operator")

    def __mul__(self, x):

        if isinstance(x, Operator):

            if self.dimension != x.dimension:
                raise ArithmeticError(
                    "Can not multiply to Operators" " with different dimensions"
                )

            ans = self.clone()
            ans._matrix = np.dot(ans._matrix, x._matrix)

            return ans

        elif isinstance(x, Number):

            ans = self.clone()
            ans._matrix *= x

            return ans

        raise TypeError("Unsupported operation for Operator")

    def __rmul__(self, x):

        if isinstance(x, Number):
            return self.__mul__(x)
        elif isinstance(x, Operator):
            return x * self

        raise TypeError("Unsupported operation for Operator")

    def __truediv__(self, x):

        if isinstance(x, Number):
            x = 1.0 / x
            return self * x

        raise TypeError("Unsupported operation for Operator")

    def __eq__(self, x):

        if not isinstance(x, Operator):
            return False

        if self.dimension != x.dimension:
            return False

        return np.all(self._matrix == x._matrix)

    def kron(self, x):
        """Tensor product between this and another Operator

        Performs a tensor product between this and another Operator,
        raising the overall rank of the tensor they represent.

        Arguments:
            x {Operator} -- Other operator

        Returns:
            Operator -- Result

        Raises:
            ValueError -- Thrown if x is not the right type of object
        """

        if not isinstance(x, Operator):
            raise ValueError(
                "Can only perform Kronecker product with" " another Operator"
            )

        # Doing it this way saves some time
        ans = self.__class__.__new__(self.__class__)
        ans._dim = self._dim + x._dim
        ans._matrix = np.kron(self._matrix, x._matrix)

        return ans

    def hilbert_schmidt(self, x):
        """Hilbert-Schmidt product between this and another Operator


        Performs a Hilbert-Schmidt product between this and another Operator,
        that acts as an inner product.

        Arguments:
            x {Operator} -- Other operator

        Returns:
            number -- Result

        Raises:
            ValueError -- Thrown if x is not the right type of object
        """

        if not isinstance(x, Operator):
            raise ValueError(
                "Can only perform Hilbert-Schmidt product with" " another Operator"
            )

        if not x.dimension == self.dimension:
            raise ValueError(
                "Operators must have the same dimension to "
                "perform Hilbert-Schmidt product"
            )

        A = self.matrix
        B = x.matrix

        return np.trace(np.dot(A.conj().T, B))

    def basis_change(self, basis):
        """Return a version of this Operator with different basis

        Transform this Operator to use a different basis. The basis
        must be a matrix of orthogonal vectors. Passing as basis
        the eigenvectors of the operator will diagonalise it.

        Arguments:
            basis {ndarray} -- Basis to transform the operator to.

        Returns:
            Operator -- Basis transformed version of this operator
        """

        ans = self.clone()
        ans._matrix = np.linalg.multi_dot([basis.T.conj(), ans._matrix, basis])

        return ans


class SpinOperator(Operator):
    @classmethod
    def from_axes(self, Is=0.5, axes="x"):
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

        if isinstance(Is, Number):
            Is = [Is]

        if len(Is) != len(axes) or len(Is) == 0:
            raise ValueError("Arrays of moments and axes must have same length > 0")

        dim = tuple(int(2 * I + 1) for I in Is)
        matrices = []

        for I, axis in zip(Is, axes):

            if I % 0.5 or I < 0.5:
                raise ValueError("{0} is not a valid spin value".format(I))

            if not (axis in "xyz+-0"):
                raise ValueError("{0} is not a valid spin axis".format(axis))

            mvals = _mvals(I)

            o = {"x": _Sx, "y": _Sy, "z": _Sz, "+": _Sp, "-": _Sm, "0": _S0}[axis](
                mvals
            )

            matrices.append(o)

        M = matrices[0]
        for m in matrices[1:]:
            M = np.kron(M, m)

        return self(M, dim=dim)


class DensityOperator(Operator):
    def __init__(self, matrix, dim=None):
        """Create a DensityOperator object

        Create an object representing a density operator. These can
        be manipulated by e.g. multiplying them by a scalar or among themselves
        (equivalent to a dot product), or adding and subtracting them.

        Arguments:
            matrix {ndarray} -- Matrix describing the operator (must be a
                                square hermitian 2D array and have non-zero
                                trace; will be normalised to have trace 1)

        Keyword Arguments:
            dim {(int,...)} -- Tuple of the dimensions of the operator. For example,
                               (2,2) corresponds to two 1/2 spins and a 4x4 matrix.
                               If not specified, it's taken from the size of
                               the matrix (default: {None})

        Raises:
            ValueError -- Any of the passed values are invalid
        """
        super(DensityOperator, self).__init__(matrix, dim)
        # Enforce unitarity
        tr = np.trace(self._matrix)

        if tr == 0:
            raise ValueError("Can not define a DensityOperator with zero trace")
        else:
            self.normalize()

        if not self.is_hermitian:
            raise ValueError("DensityOperator must be hermitian!")

    @classmethod
    def from_vectors(self, Is=0.5, vectors=[0, 0, 1], gammas=0):
        """Construct a density matrix state from real space vectors

        Construct a density matrix state by specifying a number of spins and
        real space directions. The state is initialised as the tensor product
        of independent spin states each pointing in the specified direction.
        A parameter gamma can be used to include decoherence effects and thus
        dampen or zero out all off-diagonal elements.

        Keyword Arguments:
            Is {[number]} -- List of spins (must be half-integers). Can pass a
                             number if it's only one value (default: {0.5})
            vectors {[ndarray]} -- List of vectors. Can pass a single 3D vector
                                   if it's only one value (default: {[0, 0, 1]})
            gammas {[number]} -- List of gamma factors. Can pass a single number
                                 if it's only one value. All off-diagonal
                                 elements for each corresponding density matrix
                                 will be multiplied by 1-gamma. (default: {0})

        Returns:
            DensityOperator -- The composite density operator

        Raises:
            ValueError -- Any of the passed values are invalid
        """

        if isinstance(Is, Number):
            Is = [Is]

        if len(np.array(vectors).shape) == 1:
            vectors = [vectors]

        if isinstance(gammas, Number):
            gammas = [gammas]

        if len(Is) != len(vectors) or len(Is) != len(gammas) or len(Is) == 0:
            raise ValueError(
                "Arrays of moments, axes and gammas must have same length > 0"
            )

        dim = tuple(int(2 * I + 1) for I in Is)
        matrices = []

        for I, vec, gamma in zip(Is, vectors, gammas):

            if I % 0.5 or I < 0.5:
                raise ValueError("{0} is not a valid spin value".format(I))

            if not len(vec) == 3:
                raise ValueError("{0} is not a valid 3D vector".format(vec))

            if gamma < 0 or gamma > 1:
                raise ValueError("{0} is not a valid gamma value".format(gamma))

            mvals = _mvals(I)

            S = [_Sx(mvals), _Sy(mvals), _Sz(mvals)]

            o = sum([S[i] * vec[i] for i in range(3)])

            evals, evecs = np.linalg.eigh(o)

            psi = evecs[:, np.argmax(evals)]

            m = psi[:, None] * psi[None, :].conj()
            m *= (1 - gamma) * np.ones(m.shape) + gamma * np.eye(m.shape[0])

            matrices.append(m)

        M = matrices[0]
        for m in matrices[1:]:
            M = np.kron(M, m)

        return self(M, dim=dim)

    @property
    def trace(self):
        return np.trace(self._matrix)

    def normalize(self):
        """Normalize this DensityOperator to have trace equal to one."""
        self._matrix /= self.trace

    def partial_trace(self, tracedim=[]):
        """Perform a partial trace operation

        Perform a partial trace over the specified dimensions and return the
        resulting DensityOperator.

        Keyword Arguments:
            tracedim {[int]} -- Indices of dimensions to perform the partial
                                trace over (default: {[]})

        Returns:
            DensityOperator -- Operator with partial trace
        """

        dim = list(self._dim)
        tdim = list(sorted(tracedim))

        m = self._matrix.reshape(dim + dim)

        while len(tdim) > 0:
            td = tdim.pop(-1)
            # Trace along tdim
            m = np.trace(m, axis1=td, axis2=td + len(dim))
            dim.pop(td)  # Reduce dimension accordingly

        return DensityOperator(m, dim)

    def expectation(self, operator):
        """Compute expectation value of one operator

        Compute expectation value of an operator over the state defined by
        this DensityOperator.

        Arguments:
            operator {SpinOperator} -- Operator to compute the expectation
                                       value of

        Returns:
            number -- Expectation value

        Raises:
            TypeError -- The argument isn't a SpinOperator
            ValueError -- The operator isn't compatible with this one
        """

        if not isinstance(operator, SpinOperator):
            raise TypeError("Argument must be a SpinOperator")

        if not operator.dimension == self.dimension:
            raise ValueError(
                "SpinOperator and DensityOperator do not have" " compatible dimensions"
            )

        return np.sum(operator.matrix * self.matrix.T)


class SuperOperator(Operator):
    def __init__(self, matrix, dim=None):

        matrix = np.array(matrix)
        n = matrix.shape[0] ** 0.5

        if int(n) != n:
            raise ValueError("Matrix rank should be a perfect square")

        if dim is None:
            dim = (n, n)

        super(SuperOperator, self).__init__(matrix, dim)

    @classmethod
    def left_multiplier(self, operator):
        """Create a SuperOperator that performs a left multiplication

        Create a superoperator L from an operator O such that

        L*rho = O*rho

        Arguments:
            operator {Operator} -- Operator O

        Returns:
            SuperOperator -- SuperOperator L
        """

        m = operator.matrix
        d = operator.dimension

        M = np.kron(m, np.eye(m.shape[0]))

        return self(M, d + d)

    @classmethod
    def right_multiplier(self, operator):
        """Create a SuperOperator that performs a right multiplication

        Create a superoperator L from an operator O such that

        L*rho = rho*O

        Arguments:
            operator {Operator} -- Operator O

        Returns:
            SuperOperator -- SuperOperator L
        """

        m = operator.matrix
        d = operator.dimension

        M = np.kron(np.eye(m.shape[0]), m.T)

        return self(M, d + d)

    @classmethod
    def commutator(self, operator):
        """Create a SuperOperator that performs a commutation

        Create a superoperator L from an operator O such that

        L*rho = O*rho-rho*O

        Arguments:
            operator {Operator} -- Operator O

        Returns:
            SuperOperator -- SuperOperator L
        """

        return self.left_multiplier(operator) - self.right_multiplier(operator)

    @classmethod
    def anticommutator(self, operator):
        """Create a SuperOperator that performs an anticommutation

        Create a superoperator L from an operator O such that

        L*rho = O*rho+rho*O

        Arguments:
            operator {Operator} -- Operator O

        Returns:
            SuperOperator -- SuperOperator L
        """

        return self.left_multiplier(operator) + self.right_multiplier(operator)

    @classmethod
    def bracket(self, operator):
        """Create a SuperOperator that performs a basis change

        Create a superoperator L from an operator O such that

        L*rho = O*rho*O^

        where O^ is the conjugate transpose of O.

        Arguments:
            operator {Operator} -- Operator O

        Returns:
            SuperOperator -- SuperOperator L
        """

        m = operator.matrix
        d = operator.dimension

        M = np.kron(m, m.conj())

        return self(M, d + d)

    def __add__(self, x):

        if isinstance(x, SuperOperator) or isinstance(x, Number):
            return super(SuperOperator, self).__add__(x)

        raise TypeError("Unsupported operation for SuperOperator")

    def __sub__(self, x):

        if isinstance(x, SuperOperator) or isinstance(x, Number):
            return super(SuperOperator, self).__sub__(x)

        raise TypeError("Unsupported operation for SuperOperator")

    def __mul__(self, x):

        if isinstance(x, SuperOperator) or isinstance(x, Number):
            return super(SuperOperator, self).__mul__(x)
        elif isinstance(x, Operator):
            # Vectorize x
            m = x.matrix
            s = m.shape
            m = m.reshape((-1,))
            m = np.dot(self.matrix, m).reshape(s)
            return Operator(m, x.dimension)

        raise TypeError("Unsupported operation for Operator")

    def __rmul__(self, x):

        if isinstance(x, Number):
            return self.__mul__(x)
        elif isinstance(x, SuperOperator):
            return x * self

        raise TypeError("Unsupported operation for Operator")

    def __truediv__(self, x):

        if isinstance(x, Number):
            x = 1.0 / x
            return self * x

        raise TypeError("Unsupported operation for Operator")

    def __eq__(self, x):

        if not isinstance(x, Operator):
            return False

        if self.dimension != x.dimension:
            return False

        return np.all(self._matrix == x._matrix)
