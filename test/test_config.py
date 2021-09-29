import unittest
import numpy as np
from io import StringIO

from muspinsim.input import MuSpinInput
from muspinsim.simconfig import MuSpinConfig, MuSpinConfigRange, MuSpinConfigError


class TestConfig(unittest.TestCase):

    def test_config(self):

        stest = StringIO("""
spins
    mu N e
field
    1.0
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

        self.assertTrue((cfg.get('B') == [0, 0, 1]).all())
        self.assertTrue((np.array(cfg.system.spins) == ['mu', 'N', 'e']).all())
        self.assertEqual(len(cfg.system._terms), 4)
        self.assertEqual(cfg._dissip_terms[1], 0.1)

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
