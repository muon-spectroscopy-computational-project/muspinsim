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
        self.assertTrue((f1._xbounds[0] == [-np.inf, np.inf]).all())
        self.assertTrue((f1._xbounds[1] == [0.0, 2.0]).all())

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

        # Try fitting a very basic exponential decay
        # with lbfgs
        s1 = StringIO(
            f"""
spins
    mu
fitting_method
    lbfgs
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

        # Try fitting a very basic exponential decay with least_squares
        # with an unconstrained problem and no initial guess
        s1 = StringIO(
            f"""
spins
    mu
fitting_method
    least-squares
fitting_variables
    g
fitting_data
{dblock}
dissipation 1
    g
"""
        )

        i1 = MuSpinInput(s1)
        f1 = FittingRunner(i1)
        self.assertFalse(f1._constrained)

        # Try fitting a very basic exponential decay with least_squares
        # with an unconstrained problem and an initial guess
        s1 = StringIO(
            f"""
spins
    mu
fitting_method
    least-squares
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
        self.assertFalse(f1._constrained)

        sol = f1.run()

        self.assertAlmostEqual(sol.x[0], g, 2)

        # Try fitting a very basic exponential decay
        # with least_squares with a constrained problem
        s1 = StringIO(
            f"""
spins
    mu
fitting_method
    least-squares
fitting_variables
    g   0.5 0 5
fitting_data
{dblock}
dissipation 1
    g
"""
        )

        i1 = MuSpinInput(s1)
        f1 = FittingRunner(i1)
        self.assertTrue(f1._constrained)

        sol = f1.run()

        self.assertAlmostEqual(sol.x[0], g, 2)

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

        # Try fitting a basic cosine with lbfgs
        s1 = StringIO(
            f"""
spins
    mu
results_function
    A*cos(x)+B
fitting_method
    lbfgs
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

        self.assertAlmostEqual(sol.x[0], A, 1)
        self.assertAlmostEqual(sol.x[1], B, 1)

        # Try fitting a basic cosine with least_squares - with unconstrained
        # bounds and no initial guess
        s1 = StringIO(
            f"""
spins
    mu
results_function
    A*cos(x)+B
fitting_method
    least-squares
fitting_variables
    A
    B
fitting_data
{dblock}
"""
        )

        i1 = MuSpinInput(s1)
        f1 = FittingRunner(i1)
        self.assertFalse(f1._constrained)

        sol = f1.run()

        self.assertAlmostEqual(sol.x[0], A, 2)
        self.assertAlmostEqual(sol.x[1], B, 2)

        # Try fitting a basic cosine with least_squares - with unconstrained
        # bounds and an initial guess
        s1 = StringIO(
            f"""
spins
    mu
results_function
    A*cos(x)+B
fitting_method
    least-squares
fitting_variables
    A 0.5
    B 0.5
fitting_data
{dblock}
"""
        )

        i1 = MuSpinInput(s1)
        f1 = FittingRunner(i1)
        self.assertFalse(f1._constrained)

        sol = f1.run()

        self.assertAlmostEqual(sol.x[0], A, 2)
        self.assertAlmostEqual(sol.x[1], B, 2)

        # Try fitting a basic cosine with least_squares - with constrained
        # bounds
        s1 = StringIO(
            f"""
spins
    mu
results_function
    A*cos(x)+B
fitting_method
    least-squares
fitting_variables
    A 0.5 0 10
    B 0.5 0 10
fitting_data
{dblock}
"""
        )

        i1 = MuSpinInput(s1)
        f1 = FittingRunner(i1)
        self.assertTrue(f1._constrained)

        sol = f1.run()

        self.assertAlmostEqual(sol.x[0], A, 2)
        self.assertAlmostEqual(sol.x[1], B, 2)

        # Try fitting a basic ALC experiment
        s1 = StringIO(
            """
spins
    mu e
hyperfine 1
    580 5   10
    5   580 9
    10  9   580
orientation
    zcw(10)
field
    range(1.8, 2.6, 2)
experiment
    alc
results_function
    A*y
fitting_variables
    A 0.5
fitting_data
    1.8 1
    2.6 1
"""
        )

        i1 = MuSpinInput(s1)
        f1 = FittingRunner(i1)

        sol = f1.run()

        self.assertAlmostEqual(sol.x[0], 2, 1)
