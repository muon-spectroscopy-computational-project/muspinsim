import unittest
import numpy as np
from io import StringIO

from muspinsim.input_old import MuSpinInput, _read_tensor, _read_list


class TestMuSpinInput(unittest.TestCase):

    def test_parseblock(self):

        l = _read_list(' mu e H\n')
        self.assertEqual(l, ['mu', 'e', 'H'])

        T = _read_tensor(['  0 1\n', '1 0'])
        self.assertEqual(T, [[0, 1], [1, 0]])

    def test_read_io(self):

        stream = StringIO("""
spins
    mu e 2H
polarization 
    longitudinal
field 
    3   5   100
time 
    0   5   100
save
    evolution
    integral
powder zcw
    100
hyperfine 1
    1   0   0
    0   1   0
    0   0   1
dipolar 1 3
    0   0   1
quadrupolar 3
    1   0   0
    0   1   0
    0   0  -2
""")

        f = MuSpinInput(stream)

        self.assertEqual(f.spins, ['mu', 'e', ('H', 2)])
        self.assertEqual(f.polarization, 'longitudinal')
        self.assertEqual(f.field, [3.0, 5.0, 100.0])
        self.assertEqual(f.time, [0.0, 5.0, 100.0])
        self.assertEqual(f.save, {'evolution', 'integral'})
        self.assertEqual(f.powder, ('zcw', 100))

        # Couplings
        self.assertEqual(f.hyperfine[(0, None)], [[1.0, 0.0, 0.0],
                                                  [0.0, 1.0, 0.0],
                                                  [0.0, 0.0, 1.0]])
        self.assertEqual(f.dipolar[(0, 2)], [0.0, 0.0, 1.0])
        self.assertEqual(f.quadrupolar[2], [[1.0, 0.0, 0.0],
                                            [0.0, 1.0, 0.0],
                                            [0.0, 0.0, -2.0]])

    def test_experiment(self):

        stream = StringIO("""
experiment
    ALC
""")
        f = MuSpinInput(stream)

        self.assertEqual(f.polarization, 'longitudinal')
        self.assertEqual(f.save, {'integral'})
