import unittest
import numpy as np

from muspinsim import constants
from muspinsim.spinop import SpinOperator, DensityOperator
from muspinsim.hamiltonian import (Hamiltonian, SpinHamiltonian,
                                   MuonHamiltonian, SingleTerm, DoubleTerm)


class TestSpinHamiltonian(unittest.TestCase):

    def test_creation(self):

        H = SpinHamiltonian(['e'])

        self.assertEqual(len(H.spin_system), 1)

    def test_terms(self):

        H = SpinHamiltonian(['mu', 'e'])

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

        H = SpinHamiltonian(['mu', 'e'])

        H.add_linear_term(0, [0, 0, 1])
        H.add_linear_term(1, [0, 0, 2])
        H.add_bilinear_term(0, 1, np.diag([4, 0, 0]))

        self.assertTrue(np.all(H.matrix ==
                               np.array([[1.5, 0,   0,   1],
                                         [0,  -0.5, 1,   0],
                                         [0,   1,   0.5, 0],
                                         [1,   0,   0,  -1.5]])))

    def test_rotate(self):

        H = SpinHamiltonian(['mu'])

        H.add_linear_term(0, [0, 1, 0])
        R = np.array([[0, 1, 0],
                      [-1, 0, 0],
                      [0, 0, 1]])
        rH = H.rotate(R)

        self.assertTrue(np.all(rH.matrix == np.array([[0, 0.5],
                                                      [0.5, 0]])))

        # MuonHamiltonian is special
        H = MuonHamiltonian(['mu'])

        H.set_B_field(1.0/constants.MU_GAMMA)
        H.add_linear_term(0, [0, 0, 1])

        rH = H.rotate([[0, 0, 1],
                       [0, 1, 0],
                       [-1, 0, 0]])

        self.assertTrue(np.all(rH.matrix == np.array([[0.5, 0.5],
                                                      [0.5, -0.5]])))

    def test_muonham(self):

        mH = MuonHamiltonian(['e', 'mu'])

        self.assertEqual(len(mH.spin_system), 2)
        self.assertEqual(mH.e, {0})
        self.assertEqual(mH.mu, 1)

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

        self.assertEqual(mH.terms[0].label, mH.STATIC_FIELD_LABEL)

        mH = MuonHamiltonian(['e', 'e', 'mu'])

        # Double electrons, check that it works
        self.assertEqual(mH.e, {0, 1})
        with self.assertRaises(ValueError):
            mH.add_hyperfine_term(2, np.eye(3))  # Must specify electron index

        mH.add_zeeman_term(2, [0, 0, 1])

        self.assertTrue(np.all(mH.matrix ==
                               mH.spin_system.operator({2: 'z'}).matrix*gmu))

        # A more complex Hamiltonian for dipolar and quadrupolar couplings
        mH = MuonHamiltonian(['mu', ('H', 2)])

    def test_evolve(self):

        H = SpinHamiltonian(['e'])

        H.add_linear_term(0, [1, 0, 0])       # Precession around x
        rho0 = DensityOperator.from_vectors()  # Start along z
        t = np.linspace(0, 2*np.pi, 100)
        evol = H.evolve(rho0, t, H.spin_system.operator({0: 'z'}))

        self.assertTrue(np.all(np.isclose(evol[:, 0], 0.5*np.cos(2*np.pi*t))))

    def test_integrate(self):

        H = SpinHamiltonian(['e'])

        H.add_linear_term(0, [1, 0, 0])       # Precession around x
        rho0 = DensityOperator.from_vectors()  # Start along z
        avg = H.integrate_decaying(rho0, 1.0,
                                   H.spin_system.operator({0: 'z'}))

        self.assertTrue(np.isclose(avg[0], 0.5/(1.0+4*np.pi**2)))

    # Commented out for the time being
    # def test_reduced(self):

    #     Amu = 1
    #     AH = 0.1
    #     # Field at which mu and H resonate
    #     B = (Amu-AH)/(2*(constants.MU_GAMMA-42.577))

    #     t = np.linspace(0, 2*np.pi, 100)

    #     mHnoipso = MuonHamiltonian(['e', 'mu'])
    #     mHnoipso.set_B_field(B)

    #     mHnoipso.add_hyperfine_term(1, np.eye(3)*Amu)

    #     mH = MuonHamiltonian(['e', 'mu', 'H'])
    #     mH.set_B_field(5.0)

    #     mH.add_hyperfine_term(1, np.eye(3)*Amu)
    #     mH.add_hyperfine_term(2, np.eye(3)*AH)

    #     mHred = mH.reduced_hamiltonian()

    #     rho0 = DensityOperator.from_vectors([0.5, 0.5, 0.5],
    #                                         [[0, 0, 1],
    #                                          [1, 0, 0],
    #                                          [1, 0, 0]],
    #                                         [0, 0, 0])
    #     evol = mH.evolve(rho0, t, mH.spin_system.operator({1: 'x'}))

    #     rho0red = DensityOperator.from_vectors([0.5, 0.5],
    #                                            [[1, 0, 0],
    #                                             [1, 0, 0]],
    #                                            [0, 0])
    #     opred = SpinOperator.from_axes([0.5, 0.5], ['x', '0'])
    #     evolred = mHred.evolve(rho0red, t, opred)

    #     rho0noipso = DensityOperator.from_vectors([0.5, 0.5],
    #                                               [[0, 0, 1],
    #                                                [1, 0, 0]],
    #                                               [0, 0])
    #     opnoipso = SpinOperator.from_axes([0.5, 0.5], ['0', 'x'])
    #     evolnoipso = mHnoipso.evolve(rho0noipso, t, opnoipso)

    #     errred = np.sum(abs(evolred-evol))
    #     errnoipso = np.sum(abs(evolnoipso-evol))

    #     self.assertTrue(errred < errnoipso)
    #     self.assertTrue(errred/len(t) < 1e-3)
