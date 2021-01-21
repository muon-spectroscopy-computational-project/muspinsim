import unittest
import numpy as np

from muspinsim import constants
from muspinsim.spinop import SpinOperator, DensityOperator
from muspinsim.hamiltonian import (Hamiltonian, MuonHamiltonian,
                                   SingleTerm, DoubleTerm)


class TestHamiltonian(unittest.TestCase):

    def test_creation(self):

        H = Hamiltonian(['e'])

        self.assertEqual(len(H.spin_system), 1)

    def test_terms(self):

        H = Hamiltonian(['mu', 'e'])

        H.add_linear_term(0, [0, 0, 1])
        H.add_bilinear_term(0, 1, np.eye(3).astype(int))

        terms = H.terms

        self.assertEqual(len(terms), 2)
        self.assertIsInstance(terms[0], SingleTerm)
        self.assertEqual(terms[0].i, 0)
        self.assertTrue(np.all(terms[0].compile(H.spin_system) ==
                               np.diag([0.5, 0.5, -0.5, -0.5])))
        self.assertEqual(terms[0].label, 'Single')
        self.assertEqual(str(terms[0]), 'Single { S_0 * [0 0 1] }')
        self.assertEqual(str(terms[1]), 'Double { S_0 * [[1 0 0] [0 1 0]'
                                        ' [0 0 1]] * S_1 }')

        H.remove_term(terms[0])

        with self.assertRaises(ValueError):
            H.add_linear_term(2, [0, 0, 1])

        with self.assertRaises(ValueError):
            H.add_linear_term(-1, [0, 0, 1])

        with self.assertRaises(ValueError):
            H.add_linear_term(0, [0, 0])

    def test_compile(self):

        H = Hamiltonian(['mu', 'e'])

        H.add_linear_term(0, [0, 0, 1])
        H.add_linear_term(1, [0, 0, 2])
        H.add_bilinear_term(0, 1, np.diag([4, 0, 0]))

        self.assertTrue(np.all(H.matrix ==
                               np.array([[1.5, 0,   0,   1],
                                         [0,  -0.5, 1,   0],
                                         [0,   1,   0.5, 0],
                                         [1,   0,   0,  -1.5]])))

    def test_rotate(self):

        H = Hamiltonian(['mu'])

        H.add_linear_term(0, [0, 1, 0])
        R = np.array([[0, 1, 0],
                      [-1, 0, 0],
                      [0, 0, 1]])
        rH = H.rotate(R)

        self.assertTrue(np.all(rH.matrix == np.array([[0, 0.5],
                                                      [0.5, 0]])))

    def test_muonham(self):

        mH = MuonHamiltonian()

        self.assertEqual(len(mH.spin_system), 2)

        gmu = constants.MU_GAMMA
        ge = constants.ELEC_GAMMA

        mH.set_B_field(1.0)
        mH.add_hyperfine_term(1, np.eye(3)*100)

        self.assertTrue(np.all(np.isclose(np.diag(mH.matrix),
                                          [0.5*(ge+gmu)+25,
                                           0.5*(ge-gmu)-25,
                                           0.5*(-ge+gmu)-25,
                                           0.5*(-ge-gmu)+25])))

        mH.remove_term(mH.terms[-1])
        with self.assertRaises(RuntimeError):
            # Can not remove Zeeman terms
            mH.remove_term(mH.terms[-1])

        self.assertEqual(mH.terms[0].label, 'Zeeman')

        # A more complex Hamiltonian for dipolar and quadrupolar couplings
        mH = MuonHamiltonian([('H', 2)])

    def test_evolve(self):

        H = Hamiltonian(['e'])

        H.add_linear_term(0, [1, 0, 0])       # Precession around x
        rho0 = DensityOperator.from_vectors()  # Start along z
        t = np.linspace(0, 2*np.pi, 100)
        evol = H.evolve(rho0, t, H.spin_system.operator({0: 'z'}))

        self.assertTrue(np.all(np.isclose(evol[:, 0], 0.5*np.cos(2*np.pi*t))))

    def test_integrate(self):

        H = Hamiltonian(['e'])

        H.add_linear_term(0, [1, 0, 0])       # Precession around x
        rho0 = DensityOperator.from_vectors()  # Start along z
        avg = H.integrate_decaying(rho0, 1.0,
                                   H.spin_system.operator({0: 'z'}))

        self.assertTrue(np.isclose(avg[0], 0.5/(1.0+4*np.pi**2)))
