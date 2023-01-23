import unittest
import numpy as np

from muspinsim.spinop import SpinOperator, DensityOperator, Operator, SuperOperator


class TestSpinOperator(unittest.TestCase):
    def test_construct(self):

        data = np.eye(4)

        s = SpinOperator(data)
        self.assertEqual(s.dimension, (4,))

        s = SpinOperator(data, dim=(2, 2))
        self.assertEqual(s.dimension, (2, 2))
        self.assertEqual(s.N, 4)

        with self.assertRaises(ValueError):
            SpinOperator(np.array([[1, 2]]))  # Not square

        with self.assertRaises(ValueError):
            SpinOperator(data, dim=(2, 3))  # Incompatible dimensions

    def test_clone(self):

        s = SpinOperator.from_axes()

        s2 = s.clone()

        self.assertIsInstance(s2, SpinOperator)
        self.assertIsInstance(s2, Operator)
        self.assertNotIsInstance(s2, DensityOperator)

    def test_herm(self):

        data = np.array([[1, 1.0j], [0, 1]])

        s = SpinOperator(data)
        self.assertFalse(s.is_hermitian)
        s = SpinOperator(data + data.T.conj())
        self.assertTrue(s.is_hermitian)

    def test_pauli(self):

        sx = SpinOperator.from_axes(0.5, "x")
        sy = SpinOperator.from_axes(0.5, "y")
        sz = SpinOperator.from_axes(0.5, "z")

        self.assertTrue(np.all(sx.matrix.toarray() == [[0, 0.5], [0.5, 0]]))
        self.assertTrue(np.all(sy.matrix.toarray() == [[0, -0.5j], [0.5j, 0]]))
        self.assertTrue(np.all(sz.matrix.toarray() == [[0.5, 0], [0, -0.5]]))

    def test_spin1(self):

        Sx = SpinOperator.from_axes(1, "x")
        Sz = SpinOperator.from_axes(1, "z")

        self.assertTrue(np.all(Sz.matrix.toarray() == np.diag([1, 0, -1])))
        self.assertTrue(
            np.all(
                np.isclose(
                    Sx.matrix.toarray(),
                    np.array([[0, 1, 0.0], [1, 0, 1.0], [0, 1, 0.0]]) / 2**0.5,
                )
            )
        )

    def test_operations(self):

        sx = SpinOperator.from_axes(0.5, "x")
        sy = SpinOperator.from_axes(0.5, "y")
        sz = SpinOperator.from_axes(0.5, "z")

        # Scalar operations
        self.assertTrue(np.all((2 * sx).matrix.toarray() == [[0, 1], [1, 0]]))
        self.assertTrue(np.all((sx / 2).matrix.toarray() == [[0, 0.25], [0.25, 0]]))

        # Operators (test commutation relations)
        self.assertTrue(
            np.all((sx * sy - sy * sx).matrix.toarray() == (1.0j * sz).matrix)
        )
        self.assertTrue(
            np.all((sy * sz - sz * sy).matrix.toarray() == (1.0j * sx).matrix)
        )
        self.assertTrue(
            np.all((sz * sx - sx * sz).matrix.toarray() == (1.0j * sy).matrix)
        )

        self.assertTrue(np.all((sx + 0.5).matrix.toarray() == 0.5 * np.ones((2, 2))))
        self.assertTrue(np.all((sz - 0.5).matrix.toarray() == np.diag([0, -1])))

        # Test equality
        self.assertTrue(
            np.allclose(sx.matrix.data, SpinOperator.from_axes(0.5, "x").matrix.data)
        )
        self.assertFalse(SpinOperator(np.eye(4)) == SpinOperator(np.eye(4), (2, 2)))

        # Test Kronecker product
        sxsz = sx.kron(sz)
        self.assertEqual(sxsz.dimension, (2, 2))
        self.assertTrue(
            np.all(
                4 * sxsz.matrix.toarray()
                == [[0, 0, 1, 0], [0, 0, 0, -1], [1, 0, 0, 0], [0, -1, 0, 0]]
            )
        )

        # Test Hilbert-Schmidt product
        rho = DensityOperator.from_vectors(0.5, np.array([1, 1, 0]) / 2**0.5)
        sx = SpinOperator.from_axes(0.5, "x")
        self.assertEqual(2 * np.real(rho.hilbert_schmidt(sx)), 0.5**0.5)

    def test_multi(self):

        Sx = SpinOperator.from_axes()
        Ix = SpinOperator.from_axes(1.0, "x")
        SxIx = SpinOperator.from_axes([0.5, 1.0], "xx")
        SxSx = SpinOperator.from_axes([0.5, 0.5], "xx")
        SySy = SpinOperator.from_axes([0.5, 0.5], "yy")

        self.assertTrue(np.allclose(Sx.kron(Ix).matrix.data, SxIx.matrix.data))

        with self.assertRaises(ArithmeticError):
            _ = Sx * SxIx

        self.assertTrue(
            np.allclose((16 * SxSx * SySy).matrix.toarray(), np.diag([-1, 1, 1, -1]))
        )

    def test_density(self):

        rho = DensityOperator(np.eye(6) / 6.0, (2, 3))
        rhosmall = rho.partial_trace([1])

        self.assertEqual(rhosmall.dimension, (2,))
        self.assertTrue(np.all(np.isclose(rhosmall.matrix.toarray(), np.eye(2) / 2)))

        with self.assertRaises(ValueError):
            DensityOperator(np.array([[0, 1], [1, 0]]))

        with self.assertRaises(ValueError):
            DensityOperator(np.array([[1, 1], [0, 1]]))

        rho = DensityOperator.from_vectors(0.5, [1, 0, 0])

        self.assertTrue(np.all(rho.matrix.toarray() == np.ones((2, 2)) * 0.5))

        rho = DensityOperator.from_vectors(0.5, [0, 1, 0], 0.5)

        self.assertTrue(
            np.all(
                np.isclose(
                    rho.matrix.toarray(), np.array([[0.5, -0.25j], [0.25j, 0.5]])
                )
            )
        )

    def test_superoperator(self):

        sx = SpinOperator.from_axes()
        rho0 = DensityOperator.from_vectors()
        lsx = SuperOperator.left_multiplier(sx)
        rsx = SuperOperator.right_multiplier(sx)
        csx = SuperOperator.commutator(sx)
        acsx = SuperOperator.anticommutator(sx)
        bksx = SuperOperator.bracket(sx)

        self.assertTrue(np.allclose((sx * rho0).matrix.data, (lsx * rho0).matrix.data))
        self.assertTrue(
            np.allclose((sx * rho0 - rho0 * sx).matrix.data, (csx * rho0).matrix.data)
        )
        self.assertTrue(
            np.allclose((sx * rho0 + rho0 * sx).matrix.data, (acsx * rho0).matrix.data)
        )
        self.assertTrue(
            np.allclose(
                (sx * rho0 * sx.dagger()).matrix.data, (bksx * rho0).matrix.data
            )
        )
