import unittest
import numpy as np
from io import StringIO
from ase.quaternions import Quaternion

from muspinsim.input import MuSpinInput
from muspinsim.simconfig import MuSpinConfig, MuSpinConfigError


class TestConfig(unittest.TestCase):
    def test_config(self):

        stest = StringIO(
            """
spins
    mu 2H e
field
    range(1, 11, 2)
temperature
    range(0, 10, 2)
time
    range(0, 10, 21)
orientation
    0 0
    0 180
polarization
    0 1 1
zeeman 1
    1 0 0
dipolar 1 2
    0 1 0
hyperfine 1 3
    10 2 2
    2 10 2
    2 2 10
quadrupolar 2
    -2 0 0
    0  1 0
    0  0 1
dissipation 2
    0.1
"""
        )

        itest = MuSpinInput(stest)

        cfg = MuSpinConfig(itest.evaluate())

        self.assertEqual(cfg.name, "muspinsim")
        self.assertEqual(len(cfg), 8)
        self.assertEqual(cfg.results.shape, (2, 2, 21))

        # Try getting one configuration snapshot
        cfg0 = cfg[0]

        self.assertTrue((cfg0.B == [0, 0, 1]).all())
        self.assertEqual(cfg0.T, 0)
        self.assertTrue((cfg0.t == np.linspace(0, 10, 21)).all())
        self.assertTrue(np.isclose(np.linalg.norm(cfg0.mupol), 1.0))

        # Now try recording results
        for c in cfg:
            res = np.ones(len(c.t))
            cfg.store_time_slice(c.id, res)
        self.assertTrue((cfg.results == 1).all())

        # System checks
        self.assertEqual(cfg.system.spins[0], "mu")
        self.assertEqual(cfg.system.spins[1], ("H", 2))
        self.assertEqual(cfg.system.spins[2], "e")
        self.assertEqual(len(cfg.system._terms), 4)
        self.assertEqual(cfg._dissip_terms[1], 0.1)

        self.assertIn("B", cfg._file_ranges)
        self.assertIn("T", cfg._file_ranges)
        self.assertIn("t", cfg._x_range)
        self.assertIn("orient", cfg._avg_ranges)

        # Now try a few errors

        stest = StringIO(
            """
spins
    mu e
zeeman 3
    0 0 1
"""
        )

        itest = MuSpinInput(stest)

        with self.assertRaises(MuSpinConfigError):
            cfg = MuSpinConfig(itest.evaluate())

        stest = StringIO(
            """
spins
    mu e
zeeman 1
    2 0
"""
        )

        itest = MuSpinInput(stest)

        with self.assertRaises(MuSpinConfigError):
            cfg = MuSpinConfig(itest.evaluate())

        stest = StringIO(
            """
y_axis
    integral
"""
        )

        itest = MuSpinInput(stest)

        with self.assertRaises(MuSpinConfigError):
            cfg = MuSpinConfig(itest.evaluate())

    def test_orient(self):
        # Some special tests to check how orientations are dealt with

        stest = StringIO(
            """
orientation
    0  0  0  1.0
    0.5*pi 0 0 2.0
"""
        )

        itest = MuSpinInput(stest)

        cfg = MuSpinConfig(itest.evaluate())

        orange = cfg._avg_ranges["orient"]

        self.assertEqual(sum([w for (q, w) in orange]), 2.0)
        self.assertTrue(
            np.isclose(orange[1][0].q, [2 ** (-0.5), 0, 0, -(2 ** (-0.5))]).all()
        )

        # Some more complex euler angles combinations
        rng = np.linspace(0, np.pi, 4)
        angles = np.array(np.meshgrid(*[rng, rng, rng])).reshape((3, -1)).T
        ablock = "\n".join(map(lambda x: "\t{0} {1} {2}".format(*x), angles))

        stest = StringIO("orientation\n" + ablock)

        itest = MuSpinInput(stest)

        cfg = MuSpinConfig(itest.evaluate())

        for ((a, b, c), (q1, w)) in zip(angles, cfg._avg_ranges["orient"]):
            q2 = Quaternion.from_axis_angle([0, 0, 1], c)
            q2 *= Quaternion.from_axis_angle([0, 1, 0], b)
            q2 *= Quaternion.from_axis_angle([0, 0, 1], a)
            q2 = q2.conjugate()

            self.assertTrue(np.isclose(q1.q, q2.q).all())

        # Same, but for mode zxz
        stest = StringIO("orientation zxz\n" + ablock)

        itest = MuSpinInput(stest)

        cfg = MuSpinConfig(itest.evaluate())

        for ((a, b, c), (q1, w)) in zip(angles, cfg._avg_ranges["orient"]):
            q2 = Quaternion.from_axis_angle([0, 0, 1], c)
            q2 *= Quaternion.from_axis_angle([1, 0, 0], b)
            q2 *= Quaternion.from_axis_angle([0, 0, 1], a)
            q2 = q2.conjugate()

            self.assertTrue(np.isclose(q1.q, q2.q).all())

        # Same, but for theta and phi angles alone
        angles = np.array(np.meshgrid(*[rng, rng])).reshape((2, -1)).T
        ablock = "\n".join(map(lambda x: "\t{0} {1}".format(*x), angles))

        stest = StringIO("orientation\n" + ablock)

        itest = MuSpinInput(stest)

        cfg = MuSpinConfig(itest.evaluate())

        for ((theta, phi), (q1, w)) in zip(angles, cfg._avg_ranges["orient"]):
            q2 = Quaternion.from_axis_angle([0, 0, 1], phi)
            q2 *= Quaternion.from_axis_angle([0, 1, 0], theta)
            q2 *= Quaternion.from_axis_angle([0, 0, 1], phi)
            q2 = q2.conjugate()

            self.assertTrue(np.isclose(q1.q, q2.q).all())

    def test_fitting(self):
        # Special tests for fitting tasks

        stest = StringIO(
            """
fitting_variables
    x
fitting_data
    0   0.5
    1   0.2
"""
        )

        itest = MuSpinInput(stest)
        cfg = MuSpinConfig(itest.evaluate(x=0.0))
        # Check that the time axis has been overridden
        self.assertTrue((np.array(cfg._x_range["t"]) == [0, 1]).all())

        # Should fail due to file_ranges
        stest = StringIO(
            """
fitting_variables
    x
fitting_data
    0   0.5
    1   0.2
average_axes
    none
orientation
    0  0
    0  1
"""
        )

        itest = MuSpinInput(stest)
        with self.assertRaises(MuSpinConfigError):
            cfg = MuSpinConfig(itest.evaluate(x=0.0))
