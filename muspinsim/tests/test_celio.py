import unittest
import numpy as np

from muspinsim.cpp import Celio_EvolveContrib
from muspinsim.celio import CelioHamiltonian
from muspinsim.spinop import DensityOperator
from muspinsim.spinsys import MuonSpinSystem, SingleTerm, SpinSystem


class TestCelioHamilto(unittest.TestCase):
    def test_sum(self):
        ssys = SpinSystem(["mu", "e"], celio_k=10)
        ssys.add_linear_term(0, [1, 0, 0])  # Precession around x
        extra_terms = [SingleTerm(ssys, 1, [0, 1, 0])]

        H2 = CelioHamiltonian(extra_terms, 10, ssys)
        H_sum = ssys.hamiltonian + H2

        self.assertEqual(H_sum._terms[1], extra_terms[0])

    def test_calc_H_contribs(self):
        ssys = SpinSystem(["mu", "F", ("e", 2)], celio_k=10)
        ssys.add_linear_term(0, [1, 0, 0])
        ssys.add_bilinear_term(0, 1, [[0, 0, 1], [0, 0, 1], [0, 0, 1]])
        ssys.add_bilinear_term(0, 2, [[0, 0, -1], [0, 0, -1], [0, 0, -1]])
        H = ssys.hamiltonian

        H_contribs = H._calc_H_contribs()

        self.assertEqual(len(H_contribs), 3)

        def check_H_contrib(
            H_contrib, matrix, other_dimension, spin_order, spin_dimensions
        ):
            self.assertTrue(np.allclose(H_contrib.matrix.toarray(), matrix))
            self.assertEqual(H_contrib.other_dimension, other_dimension)
            self.assertTrue(np.allclose(H_contrib.spin_order, spin_order))
            self.assertTrue(np.allclose(H_contrib.spin_dimensions, spin_dimensions))

        check_H_contrib(
            H_contrib=H_contribs[0],
            matrix=[[0, 0.5], [0.5, 0]],
            other_dimension=6,
            spin_order=[0, 1, 2],
            spin_dimensions=[2, 2, 3],
        )

        check_H_contrib(
            H_contrib=H_contribs[1],
            matrix=[
                [0.25, 0, 0.25 - 0.25j, 0],
                [0, -0.25, 0, -0.25 + 0.25j],
                [0.25 + 0.25j, 0, -0.25, 0],
                [0, -0.25 - 0.25j, 0, 0.25],
            ],
            other_dimension=3,
            spin_order=[0, 1, 2],
            spin_dimensions=[2, 2, 3],
        )

        check_H_contrib(
            H_contrib=H_contribs[2],
            matrix=[
                [-0.5, 0, 0, -0.5 + 0.5j, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0.5, 0, 0, 0.5 - 0.5j],
                [-0.5 - 0.5j, 0, 0, 0.5, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0.5 + 0.5j, 0, 0, -0.5],
            ],
            other_dimension=2,
            spin_order=[0, 2, 1],
            spin_dimensions=[2, 3, 2],
        )

    def test_calc_trotter_evol_op(self):
        ssys = SpinSystem(["mu", "e"], celio_k=10)
        ssys.add_linear_term(0, [1, 0, 0])
        H = ssys.hamiltonian

        evol_op_contribs = H._calc_trotter_evol_op_contribs(1, False)

        self.assertEqual(len(evol_op_contribs), 1)

        self.assertTrue(
            np.allclose(
                evol_op_contribs[0].toarray(),
                [
                    [0.95105652, 0, -0.30901699j, 0],
                    [0, 0.95105652, 0, -0.30901699j],
                    [-0.30901699j, 0, 0.95105652, 0],
                    [0, -0.30901699j, 0, 0.95105652],
                ],
            )
        )

        # Should be 2x2 for the fast variant
        ssys = SpinSystem(["mu", "e"], celio_k=10)
        ssys.add_linear_term(0, [1, 0, 0])
        H = ssys.hamiltonian

        evol_op_contribs = H._calc_trotter_evol_op_contribs(1, True)

        self.assertEqual(len(evol_op_contribs), 1)
        self.assertTrue(isinstance(evol_op_contribs[0], Celio_EvolveContrib))

        self.assertTrue(
            np.allclose(
                evol_op_contribs[0].matrix,
                [
                    [0.95105652, -0.30901699j],
                    [-0.30901699j, 0.95105652],
                ],
            )
        )

        self.assertEqual(evol_op_contribs[0].other_dim, 2)
        self.assertTrue(np.allclose(evol_op_contribs[0].indices, [0, 1, 2, 3]))

    def test_evolve(self):
        ssys = SpinSystem(["e"], celio_k=10)
        ssys.add_linear_term(0, [1, 0, 0])  # Precession around x
        H = ssys.hamiltonian
        rho0 = DensityOperator.from_vectors()  # Start along z
        t = np.linspace(0, 1, 100)

        self.assertTrue(isinstance(H, CelioHamiltonian))

        evol = H.evolve(rho0, t, ssys.operator({0: "z"}))

        self.assertTrue(np.all(np.isclose(evol[:, 0], 0.5 * np.cos(2 * np.pi * t))))

    def test_fast_evolve(self):
        ssys = MuonSpinSystem(["mu", "e"], celio_k=10)
        ssys.add_linear_term(0, [1, 0, 0])  # Precession around x
        H = ssys.hamiltonian
        t = np.linspace(0, 1, 100)

        self.assertTrue(isinstance(H, CelioHamiltonian))

        # Start along z
        evol = H.fast_evolve(ssys.sigma_mu([0, 0, 1]), t, 10)

        # This test is subject to randomness, but np.isclose appears to avoid
        # any issues
        self.assertTrue(np.all(np.isclose(evol[:], 0.5 * np.cos(2 * np.pi * t))))

    def test_integrate(self):
        ssys = SpinSystem(["e"], celio_k=10)
        ssys.add_linear_term(0, [1, 0, 0])  # Precession around x
        H = ssys.hamiltonian
        rho0 = DensityOperator.from_vectors()  # Start along z

        with self.assertRaises(NotImplementedError):
            H.integrate_decaying(rho0, 1.0, ssys.operator({0: "z"}))
