import unittest
import numpy as np

from muspinsim.hamiltonian import Hamiltonian, SingleTerm, DoubleTerm


class TestHamiltonian(unittest.TestCase):

    def test_creation(self):

        H = Hamiltonian(['e'])

        self.assertEqual(len(H.spin_system), 1)

    def test_terms(self):

        H = Hamiltonian(['mu', 'e'])

        H.add_linear_term(0, [0, 0, 1])

        terms = H.terms

        self.assertEqual(len(terms), 1)
        self.assertIsInstance(terms[0], SingleTerm)
        self.assertEqual(terms[0].i, 0)
        self.assertTrue(np.all(terms[0].compile(H.spin_system) ==
                               np.diag([0.5, 0.5, -0.5, -0.5])))

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
