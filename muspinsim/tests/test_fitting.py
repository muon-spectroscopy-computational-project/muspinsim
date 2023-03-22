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
    A  0.0
    B  1.0 0 2.0
fitting_data
    0   0.5
    1   0.5
    2   0.5
"""
        )

        i1 = MuSpinInput(s1)
        f1 = FittingRunner(i1)

        self.assertEqual(f1._xnames, ("A", "B"))
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
            f"""
spins
    mu
fitting_variables
    g   0.5
fitting_data
{dblock}
dissipation 1
    g
"""
        )

        i1 = MuSpinInput(s1)
        f1 = FittingRunner(i1)

        sol = f1.run()

        self.assertAlmostEqual(sol.x[0], g, 3)

    def test_fit_results_function(self):

        # Try fitting a basic cosine function
        A = 2
        B = 1
        x_values = np.arange(0, 10)
        y_values = A * np.cos(x_values) + B
        dblock = "\n".join(["\t{0} {1}".format(*d) for d in zip(x_values, y_values)])

        s1 = StringIO(
            f"""
spins
    mu
results_function
    A*cos(x)+B
fitting_variables
    A 0.5
    B 0.5
fitting_data
{dblock}
"""
        )

        i1 = MuSpinInput(s1)
        f1 = FittingRunner(i1)

        sol = f1.run()

        self.assertAlmostEqual(sol.x[0], A, 3)
        self.assertAlmostEqual(sol.x[1], B, 3)
