import unittest
import numpy as np

from muspinsim import constants
from muspinsim.spinop import SpinOperator
from muspinsim.spinsys import SpinSystem


class TestSpinSystem(unittest.TestCase):

    def test_create(self):

        ssys = SpinSystem(['mu', 'e'])

        ssys = SpinSystem(['mu', ('H', 2)])

    def test_check(self):

        ssys = SpinSystem(['mu', 'e'])

        self.assertEqual(ssys.gamma(0), constants.MU_GAMMA)
        self.assertEqual(ssys.gamma(1), constants.ELEC_GAMMA)

        self.assertEqual(len(ssys), 2)

    def test_operator(self):

        ssys = SpinSystem(['mu', 'e'])

        self.assertEqual(ssys.operator({0: 'x'}),
                         SpinOperator.from_axes([0.5, 0.5], 'x0'))
        self.assertEqual(ssys.operator({0: 'z', 1: 'y'}),
                         SpinOperator.from_axes([0.5, 0.5], 'zy'))
