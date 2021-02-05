import unittest
import numpy as np

from muspinsim import constants
from muspinsim.spinop import SpinOperator
from muspinsim.spinsys import (SpinSystem, InteractionTerm, SingleTerm,
                               DoubleTerm)


class TestSpinSystem(unittest.TestCase):

    def test_create(self):

        ssys = SpinSystem(['mu', 'e'])

        ssys = SpinSystem(['mu', ('H', 2)])

        # Test cloning
        ssys2 = ssys.clone()

        self.assertEqual(ssys.dimension, ssys2.dimension)

        # Check that the copy is deep
        ssys2._spins[0] = 'C'
        self.assertEqual(ssys.spins, ['mu', ('H', 2)])

    def test_terms(self):

        ssys = SpinSystem(['mu', 'e'])

        # A scalar term
        term = InteractionTerm(ssys, [], 2.0)
        self.assertEqual(term.operator, ssys.operator({})*2)

        # Linear term
        term = SingleTerm(ssys, 0, [0, 0, 1])
        self.assertEqual(term.operator, ssys.operator({0: 'z'}))

        # Test copy
        copy = term.clone()
        copy._tensor[0] = 1

        self.assertTrue(isinstance(copy, SingleTerm))
        self.assertFalse(np.all(copy.tensor == term.tensor))

    def test_check(self):

        ssys = SpinSystem(['mu', 'e'])

        self.assertEqual(ssys.gamma(0), constants.MU_GAMMA)
        self.assertEqual(ssys.gamma(1), constants.ELEC_GAMMA)

        self.assertEqual(len(ssys), 2)
        self.assertEqual(ssys.dimension, (2, 2))

    def test_operator(self):

        ssys = SpinSystem(['mu', 'e'])

        self.assertEqual(ssys.operator({0: 'x'}),
                         SpinOperator.from_axes([0.5, 0.5], 'x0'))
        self.assertEqual(ssys.operator({0: 'z', 1: 'y'}),
                         SpinOperator.from_axes([0.5, 0.5], 'zy'))
        self.assertEqual(ssys.dimension, (2, 2))

    def test_addterms(self):

        ssys = SpinSystem(['mu', 'e'])

        t1 = ssys.add_linear_term(0, [1, 0, 1])

        self.assertEqual(t1.label, 'Single')
        self.assertEqual(t1.operator, ssys.operator({0: 'x'}) +
                         ssys.operator({0: 'z'}))

        t2 = ssys.add_bilinear_term(0, 1, np.eye(3))

        self.assertEqual(t2.label, 'Double')
        self.assertEqual(t2.operator, ssys.operator({0: 'x', 1: 'x'}) +
                         ssys.operator({0: 'y', 1: 'y'}) +
                         ssys.operator({0: 'z', 1: 'z'}))

        H = ssys.hamiltonian

        self.assertTrue(np.all(np.isclose(H.matrix,
                                          np.array([[0.75, 0, 0.5, 0.0],
                                                    [0, 0.25, 0.5, 0.5],
                                                    [0.5, 0.5, -0.75, 0],
                                                    [0.0, 0.5, 0, -0.25]]))))
