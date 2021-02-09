import unittest
import numpy as np

from muspinsim import constants
from muspinsim.spinop import SpinOperator, DensityOperator
from muspinsim.spinsys import SpinSystem
from muspinsim.hamiltonian import Hamiltonian
from muspinsim.lindbladian import Lindbladian


class TestLindbladian(unittest.TestCase):

    def test_creation(self):

        # Create a basic Hamiltonian
        sx = SpinOperator.from_axes()
        sz = SpinOperator.from_axes(0.5, 'z')

        H = Hamiltonian(sz.matrix)

        # No dissipation
        L = Lindbladian.from_hamiltonian(H)

        self.assertEqual(L.dimension, (2, 2))
        self.assertTrue(np.all(L.matrix == 1.0j*np.diag([0, -1, 1, 0])))

        # Dissipation
        L = Lindbladian.from_hamiltonian(H, [(sx, 0.1)])

        self.assertTrue(np.all(np.isclose(L.matrix, [
            [-0.025, 0, 0, 0.025],
            [0, -0.025-1.0j, 0.025, 0],
            [0, 0.025, -0.025+1.0j, 0],
            [0.025, 0, 0, -0.025]
        ])))

    def test_evolve(self):

        # Basic test
        sx = SpinOperator.from_axes()
        sz = SpinOperator.from_axes(0.5, 'z')

        H = Hamiltonian(sz.matrix)
        L = Lindbladian.from_hamiltonian(H)
        rho0 = DensityOperator.from_vectors(0.5, [1, 0, 0], 0)
        t = np.linspace(0, 1, 100)

        evol = L.evolve(rho0, t, sx)

        self.assertTrue(np.all(np.isclose(evol[:, 0], 0.5*np.cos(2*np.pi*t))))

        # Same but with decay
        g = 2.0
        L.add_dissipative_term(sx, g)
        evol = L.evolve(rho0, t, sx)

        ap = -0.5*np.pi*g+((0.5*np.pi*g)**2-4*np.pi**2)**0.5
        am = -0.5*np.pi*g-((0.5*np.pi*g)**2-4*np.pi**2)**0.5
        A = ap*am/(am-ap)

        # Analytical solution for this case
        sol = np.real(0.5*A*(np.exp(ap*t)/ap-np.exp(am*t)/am))

        self.assertTrue(np.all(np.isclose(evol[:, 0], sol)))
