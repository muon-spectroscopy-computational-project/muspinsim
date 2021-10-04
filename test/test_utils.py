import unittest

import numpy as np
from muspinsim.utils import Clonable, deepmap, quat_from_polar, zcw_gen


class TestUtils(unittest.TestCase):

    def test_clonable(self):

        class TestClass(Clonable):
            def __init__(self):
                self.data = {'x': [1, 2, 3]}

        tc = TestClass()
        # Clone it
        tc2 = tc.clone()
        # Change values
        tc2.data['x'][0] = 2
        # Check it didn't change the original
        self.assertEqual(tc.data['x'][0], 1)

    def test_deepmap(self):

        data = [[1, 2, 3], [4, 5], [6, [7, 8]]]

        def square(x):
            return x**2

        data2 = deepmap(square, data)

        self.assertEqual(data2[0], [1, 4, 9])
        self.assertEqual(data2[1], [16, 25])
        self.assertEqual(data2[2][0], 36)
        self.assertEqual(data2[2][1], [49, 64])

    def test_quat(self):

        q1 = quat_from_polar(np.pi/4.0, 0)
        z1 = q1.rotate([0, 0, 1])

        self.assertTrue(np.isclose(z1, [2**(-0.5), 0, 2**(-0.5)]).all())

        theta = 0.6*np.pi
        phi = 0.4*np.pi
        ct = np.cos(theta)
        st = np.sin(theta)
        cp = np.cos(phi)
        sp = np.sin(phi)

        q2 = quat_from_polar(theta, phi)
        z2 = q2.rotate([0, 0, 1])

        self.assertTrue(np.isclose(z2, [st*cp, st*sp, ct]).all())

    def test_zcw(self):

        N = 1000
        orients = zcw_gen(N)

        # Are these correct? Basic test
        f = 3.0*np.cos(orients[:, 0])**2-1.0
        self.assertAlmostEqual(np.average(f), 0.0, 5)
