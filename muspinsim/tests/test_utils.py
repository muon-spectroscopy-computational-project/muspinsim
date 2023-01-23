import unittest

import numpy as np
from ase.quaternions import Quaternion

from muspinsim.utils import (
    Clonable,
    deepmap,
    quat_from_polar,
    zcw_gen,
    eulrange_gen,
    get_xy,
)


class TestUtils(unittest.TestCase):
    def test_clonable(self):
        class TestClass(Clonable):
            def __init__(self):
                self.data = {"x": [1, 2, 3]}

        tc = TestClass()
        # Clone it
        tc2 = tc.clone()
        # Change values
        tc2.data["x"][0] = 2
        # Check it didn't change the original
        self.assertEqual(tc.data["x"][0], 1)

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

        q1 = quat_from_polar(np.pi / 4.0, 0)
        z1 = q1.rotate([0, 0, 1])

        self.assertTrue(np.isclose(z1, [2 ** (-0.5), 0, 2 ** (-0.5)]).all())

        theta = 0.6 * np.pi
        phi = 0.4 * np.pi
        ct = np.cos(theta)
        st = np.sin(theta)
        cp = np.cos(phi)
        sp = np.sin(phi)

        q2 = quat_from_polar(theta, phi)
        z2 = q2.rotate([0, 0, 1])

        self.assertTrue(np.isclose(z2, [st * cp, st * sp, ct]).all())

        # Let's check that the inverse works as planned too
        q3 = q2.conjugate()
        z3 = q3.rotate([0, 0, 1])

        self.assertTrue(np.isclose(z3, [-st * cp, st * sp, ct]).all())

    def test_zcw(self):

        N = 1000
        orients = zcw_gen(N)

        # Are these correct? Basic test
        f = 3.0 * np.cos(orients[:, 0]) ** 2 - 1.0
        self.assertAlmostEqual(np.average(f), 0.0, 5)

    def test_eulrange(self):

        N = 10
        ow = eulrange_gen(N)

        A = np.array([[1, 2, 3], [2, 4, 4], [3, 4, -3]])

        Asum = np.zeros((3, 3))
        quats = [Quaternion.from_euler_angles(a, b, c).q for (a, b, c, w) in ow]

        for (a, b, c, w) in ow:
            q = Quaternion.from_euler_angles(a, b, c)
            R = q.rotation_matrix()
            Asum += (R @ A @ R.T) * w

        Asum /= np.sum(ow[:, -1])

        err = np.sum(np.abs(Asum - np.eye(3) * np.trace(A) / 3))

        # Yeah, it's really crude for now, sadly
        self.assertLess(err, 1.0)

    def test_xy(self):

        z = np.array([0, 0, 1])
        x, y = get_xy(z)

        self.assertTrue(np.isclose(x, [1, 0, 0]).all())

        z = np.array([1, 1, 2.0])
        x, y = get_xy(z)
        z /= np.linalg.norm(z)

        self.assertTrue(np.isclose(np.cross(x, y), z).all())

        with self.assertRaises(ValueError):
            get_xy([0, 0, 0])
