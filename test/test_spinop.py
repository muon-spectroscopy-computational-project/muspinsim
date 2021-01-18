import unittest
import numpy as np

from muspinsim.spinop import SpinOperator


class TestSpinOperator(unittest.TestCase):

    def test_pauli(self):

        sx = SpinOperator(0.5, 'x')
        sy = SpinOperator(0.5, 'y')
        sz = SpinOperator(0.5, 'z')

        self.assertTrue(np.all(sx.full_matrix == [[0, 0.5], [0.5, 0]]))
        self.assertTrue(np.all(sy.full_matrix == [[0, -0.5j], [0.5j, 0]]))
        self.assertTrue(np.all(sz.full_matrix == [[0.5, 0], [0, -0.5]]))

    def test_spin1(self):

        Sx = SpinOperator(1, 'x')
        Sz = SpinOperator(1, 'z')

        self.assertTrue(np.all(Sz.full_matrix == np.diag([1, 0, -1])))
        self.assertTrue(np.all(np.isclose(Sx.full_matrix,
                                          np.array([[0, 1, 0.],
                                                    [1, 0, 1.],
                                                    [0, 1, 0.]])/2**0.5)))

    def test_operations(self):

        sx = SpinOperator(0.5, 'x')
        sy = SpinOperator(0.5, 'y')
        sz = SpinOperator(0.5, 'z')

        # Scalar operations
        self.assertTrue(np.all((2*sx).full_matrix == [[0, 1], [1, 0]]))
        self.assertTrue(np.all((sx/2).full_matrix == [[0, 0.25], [0.25, 0]]))

        # Operators (test commutation relations)
        self.assertTrue(np.all((sx*sy-sy*sx).full_matrix ==
                               (1.0j*sz).full_matrix))
        self.assertTrue(np.all((sy*sz-sz*sy).full_matrix ==
                               (1.0j*sx).full_matrix))
        self.assertTrue(np.all((sz*sx-sx*sz).full_matrix ==
                               (1.0j*sy).full_matrix))

        # Test equality
        self.assertTrue(2*sx == SpinOperator(0.5, 'x', 2))

    def test_multi(self):

        Sx = SpinOperator()
        Ix = SpinOperator(1.0, 'x')
        SxIx = SpinOperator([0.5, 1.0], 'xx')
        SxSx = SpinOperator([0.5, 0.5], 'xx')
        SySy = SpinOperator([0.5, 0.5], 'yy')

        self.assertTrue(np.all(Sx.kron(Ix).full_matrix == SxIx.full_matrix))

        with self.assertRaises(ArithmeticError):
            _ = Sx*SxIx

        self.assertTrue(np.all((16*SxSx*SySy).full_matrix ==
                               np.diag([-1, 1, 1, -1])))
