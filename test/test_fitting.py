import unittest
import numpy as np
from io import StringIO

from muspinsim.input import MuSpinInput
from muspinsim.fitting import FittingRunner


class TestFitting(unittest.TestCase):
    def test_fitset(self):

        s1 = StringIO(
            """
fitting_variables
    x  0.0
    y  1.0 0 2.0
fitting_data
    0   0.5
    1   0.5
    2   0.5
"""
        )

        i1 = MuSpinInput(s1)
        f1 = FittingRunner(i1)

        self.assertEqual(f1._xnames, ("x", "y"))
        self.assertTrue((f1._x == [0, 1]).all())
        self.assertEqual(f1._xbounds[0], (-np.inf, np.inf))
        self.assertEqual(f1._xbounds[1], (0.0, 2.0))

    def test_fitrun(self):

        # Try fitting a very basic exponential decay
        data = np.zeros((100, 3))
        data[:, 0] = np.linspace(0, 10.0, len(data))
        g = 0.2
        data[:, 1] = 0.5 * np.exp(-g * data[:, 0])
        dblock = "\n".join(["\t{0} {1}".format(*d) for d in data])

        s1 = StringIO(
            """
spins
    mu
fitting_variables
    g   0.5
fitting_data
{data}
dissipation 1
    g
""".format(
                data=dblock
            )
        )

        i1 = MuSpinInput(s1)
        f1 = FittingRunner(i1)

        sol = f1.run()

        self.assertAlmostEqual(sol.x[0], g, 3)
