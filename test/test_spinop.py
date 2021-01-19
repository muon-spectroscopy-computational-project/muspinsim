import unittest
import numpy as np

from muspinsim.spinop import SpinOperator


class TestSpinOperator(unittest.TestCase):

    def test_construct(self):

        data = np.eye(4)

        s = SpinOperator(data)
        self.assertEqual(s.dimension, (4,))

        s = SpinOperator(data, dim=(2, 2))
        self.assertEqual(s.dimension, (2, 2))

        with self.assertRaises(ValueError):
            SpinOperator([[1, 2]])  # Not square

        with self.assertRaises(ValueError):
            SpinOperator(data, dim=(2, 3))  # Incompatible dimensions

    def test_herm(self):

        data = np.array([[1, 1.0j], [0, 1]])

        s = SpinOperator(data)
        self.assertFalse(s.is_hermitian)
        s = SpinOperator(data+data.T.conj())
        self.assertTrue(s.is_hermitian)

    def test_pauli(self):

        sx = SpinOperator.from_axes(0.5, 'x')
        sy = SpinOperator.from_axes(0.5, 'y')
        sz = SpinOperator.from_axes(0.5, 'z')

        self.assertTrue(np.all(sx.matrix == [[0, 0.5], [0.5, 0]]))
        self.assertTrue(np.all(sy.matrix == [[0, -0.5j], [0.5j, 0]]))
        self.assertTrue(np.all(sz.matrix == [[0.5, 0], [0, -0.5]]))

    def test_spin1(self):

        Sx = SpinOperator.from_axes(1, 'x')
        Sz = SpinOperator.from_axes(1, 'z')

        self.assertTrue(np.all(Sz.matrix == np.diag([1, 0, -1])))
        self.assertTrue(np.all(np.isclose(Sx.matrix,
                                          np.array([[0, 1, 0.],
                                                    [1, 0, 1.],
                                                    [0, 1, 0.]])/2**0.5)))

    def test_operations(self):

        sx = SpinOperator.from_axes(0.5, 'x')
        sy = SpinOperator.from_axes(0.5, 'y')
        sz = SpinOperator.from_axes(0.5, 'z')

        # Scalar operations
        self.assertTrue(np.all((2*sx).matrix == [[0, 1], [1, 0]]))
        self.assertTrue(np.all((sx/2).matrix == [[0, 0.25], [0.25, 0]]))

        # Operators (test commutation relations)
        self.assertTrue(np.all((sx*sy-sy*sx).matrix ==
                               (1.0j*sz).matrix))
        self.assertTrue(np.all((sy*sz-sz*sy).matrix ==
                               (1.0j*sx).matrix))
        self.assertTrue(np.all((sz*sx-sx*sz).matrix ==
                               (1.0j*sy).matrix))

        # Test equality
        self.assertTrue(2*sx == SpinOperator.from_axes(0.5, 'x', 2))

    def test_multi(self):

        Sx = SpinOperator.from_axes()
        Ix = SpinOperator.from_axes(1.0, 'x')
        SxIx = SpinOperator.from_axes([0.5, 1.0], 'xx')
        SxSx = SpinOperator.from_axes([0.5, 0.5], 'xx')
        SySy = SpinOperator.from_axes([0.5, 0.5], 'yy')

        self.assertTrue(np.all(Sx.kron(Ix).matrix == SxIx.matrix))

        with self.assertRaises(ArithmeticError):
            _ = Sx*SxIx

        self.assertTrue(np.all((16*SxSx*SySy).matrix ==
                               np.diag([-1, 1, 1, -1])))
