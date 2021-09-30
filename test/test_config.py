import unittest
import numpy as np
from io import StringIO

from muspinsim.input import MuSpinInput
from muspinsim.simconfig import (MuSpinConfig, MuSpinConfigError)


class TestConfig(unittest.TestCase):

    def test_config(self):

        stest = StringIO("""
spins
    mu N e
field
    range(1, 11, 2)
temperature
    range(0, 10, 2)
time 
    range(0, 10, 21)
orientation
    0 0
    0 180
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
""")

        itest = MuSpinInput(stest)

        cfg = MuSpinConfig(itest.evaluate())

        self.assertEqual(cfg.name, 'muspinsim')
        self.assertEqual(len(cfg), 8)
        self.assertEqual(cfg.results.shape, (2, 2, 21))

        # Try getting one configuration snapshot
        cfg0 = cfg[0]

        self.assertTrue((cfg0.B == [0,0,1]).all())
        self.assertEqual(cfg0.T, 0)
        self.assertTrue((cfg0.t == np.linspace(0, 10, 21)).all())

        # Now try recording results
        for c in cfg:
            res = np.ones(len(c.t))
            cfg.store_time_slice(c.id, res)
        self.assertTrue((cfg.results == 1).all())

        # System checks
        self.assertTrue((np.array(cfg.system.spins) == ['mu', 'N', 'e']).all())
        self.assertEqual(len(cfg.system._terms), 4)
        self.assertEqual(cfg._dissip_terms[1], 0.1)

        self.assertIn('B', cfg._file_ranges)
        self.assertIn('T', cfg._file_ranges)
        self.assertIn('t', cfg._x_range)
        self.assertIn('orient', cfg._avg_ranges)

        # Now try a few errors

        stest = StringIO("""
spins
    mu e
zeeman 3
    0 0 1
""")

        itest = MuSpinInput(stest)

        with self.assertRaises(MuSpinConfigError):
            cfg = MuSpinConfig(itest.evaluate())

        stest = StringIO("""
spins
    mu e
zeeman 1
    2 0
""")

        itest = MuSpinInput(stest)

        with self.assertRaises(MuSpinConfigError):
            cfg = MuSpinConfig(itest.evaluate())

        stest = StringIO("""
y_axis
    integral
""")

        itest = MuSpinInput(stest)

        with self.assertRaises(MuSpinConfigError):
            cfg = MuSpinConfig(itest.evaluate())
