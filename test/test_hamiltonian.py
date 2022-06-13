import unittest
import numpy as np

from muspinsim.spinop import SpinOperator, DensityOperator
from muspinsim.spinsys import SpinSystem
from muspinsim.hamiltonian import Hamiltonian


class TestHamiltonian(unittest.TestCase):
    def test_creation(self):

        H = np.array([[1, 0], [0, -1]])
        H = Hamiltonian(H)

        self.assertEqual(H.dimension, (2,))

        # Error if not Hermitian
        with self.assertRaises(ValueError):
            Hamiltonian([[1, 1], [0, 1]])

    def test_diag(self):

        Sx = SpinOperator.from_axes()
        H = Hamiltonian(Sx.matrix)

        evals, evecs = H.diag()
        evecsT = np.array([[1.0, 1.0], [-1.0, 1.0]]) / 2**0.5

        self.assertTrue(np.all(evals == [-0.5, 0.5]))
        self.assertTrue(np.all(np.isclose(abs(np.dot(evecs, evecsT)), np.eye(2))))

        Hrot = H.basis_change(evecs)

        self.assertTrue(np.all(np.isclose(Hrot.matrix, np.diag(evals))))

    def test_evolve(self):

        ssys = SpinSystem(["e"])
        ssys.add_linear_term(0, [1, 0, 0])  # Precession around x
        H = ssys.hamiltonian
        rho0 = DensityOperator.from_vectors()  # Start along z
        t = np.linspace(0, 1, 100)

        evol = H.evolve(rho0, t, ssys.operator({0: "z"}))

        self.assertTrue(np.all(np.isclose(evol[:, 0], 0.5 * np.cos(2 * np.pi * t))))

    def test_integrate(self):

        ssys = SpinSystem(["e"])
        ssys.add_linear_term(0, [1, 0, 0])  # Precession around x
        H = ssys.hamiltonian
        rho0 = DensityOperator.from_vectors()  # Start along z
        avg = H.integrate_decaying(rho0, 1.0, ssys.operator({0: "z"}))

        self.assertTrue(np.isclose(avg[0], 0.5 / (1.0 + 4 * np.pi**2)))
