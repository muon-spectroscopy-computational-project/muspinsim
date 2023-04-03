import unittest
import numpy as np
import scipy.constants as cnst
from io import StringIO

from muspinsim import constants
from muspinsim.hamiltonian import Hamiltonian
from muspinsim.experiment import ExperimentRunner
from muspinsim.input import MuSpinInput


class TestExperiment(unittest.TestCase):
    def test_create(self):

        gmu = constants.MU_GAMMA
        ge = constants.ELEC_GAMMA

        stest = StringIO(
            """
name
    test
spins
    mu
zeeman 1
    0 0 1
"""
        )
        itest = MuSpinInput(stest)
        ertest = ExperimentRunner(itest)
        ertest.B = np.array([0, 0, 1.0])

        self.assertTrue((ertest.Hsys.matrix == np.diag([0.5 * gmu, -0.5 * gmu])).all())
        self.assertTrue(np.allclose(ertest.Hz.matrix.data, ertest.Hsys.matrix.data))

        stest = StringIO(
            """
name
    test
spins
    e mu
"""
        )

        ertest.B = [0, 0, 1.0]

        itest = MuSpinInput(stest)
        ertest = ExperimentRunner(itest)

        ertest.B = np.array([0, 0, 1.0])

        self.assertEqual(ertest._system.elec_indices, {0})
        self.assertEqual(ertest._system.muon_index, 1)

        self.assertTrue(
            np.all(
                ertest.Hz.matrix.toarray()
                == np.diag(
                    [
                        0.5 * (ge + gmu),
                        0.5 * (ge - gmu),
                        0.5 * (-ge + gmu),
                        0.5 * (-ge - gmu),
                    ]
                )
            )
        )
        self.assertTrue(isinstance(ertest.Hz, Hamiltonian))

        Sx_mu = ertest._system.operator({1: "x"})
        Sz_e = ertest._system.operator({0: "z"})

        self.assertAlmostEqual(ertest.rho0.expectation(Sx_mu), 0.5)
        self.assertAlmostEqual(ertest.rho0.expectation(Sz_e), 0.0)

    def test_rho0(self):

        stest = StringIO(
            """
spins
    e mu
"""
        )
        itest = MuSpinInput(stest)
        ertest = ExperimentRunner(itest)

        rho0 = ertest.rho0

        self.assertTrue(
            np.all(
                np.isclose(
                    rho0.matrix.toarray(),
                    [
                        [0.25, 0.25, 0, 0],
                        [0.25, 0.25, 0, 0],
                        [0, 0, 0.25, 0.25],
                        [0, 0, 0.25, 0.25],
                    ],
                )
            )
        )

        gmu = constants.MU_GAMMA
        ge = constants.ELEC_GAMMA

        T = 100
        ertest.B = [0, 0, 2.0e-6 * cnst.k * T / (ge * cnst.h)]
        ertest.T = T
        ertest.p = [0, 0, 1.0]

        rho0 = ertest.rho0

        Z = np.exp([-1, 1])
        Z /= np.sum(Z)

        self.assertTrue(
            np.all(np.isclose(np.diag(rho0.matrix.toarray()), [Z[0], 0, Z[1], 0]))
        )

    def test_run(self):

        # Empty system
        stest = StringIO(
            """
spins
    e mu
time
    range(0, 10)
"""
        )
        itest = MuSpinInput(stest)
        ertest = ExperimentRunner(itest)

        results = ertest.run()

        self.assertTrue(np.all(results == 0.5))

        # Simple system
        stest = StringIO(
            """
spins
    e mu
time
    range(0, 10)
zeeman 2
    0 0 1.0/muon_gyr
"""
        )
        itest = MuSpinInput(stest)
        ertest = ExperimentRunner(itest)

        results = ertest.run()
        times = ertest.config.x_axis_values

        self.assertTrue(np.all(np.isclose(results, 0.5 * np.cos(2 * np.pi * times))))

        tau = constants.MU_TAU

        stest = StringIO(
            """
spins
    e mu
time
    range(0, 10)
zeeman 2
    0 0 1.0/muon_gyr
y_axis
    integral
x_axis
    field
field
    0
    1
"""
        )
        itest = MuSpinInput(stest)
        ertest = ExperimentRunner(itest)

        results = ertest.run()

        self.assertAlmostEqual(results[0], 0.5 / (1.0 + 4 * np.pi**2 * tau**2))

        # Same result as above should hold for an intrinsic field
        stest = StringIO(
            """
spins
    e mu
time
    range(0, 10)
zeeman 2
    0 0 1.0/muon_gyr
y_axis
    integral
x_axis
    intrinsic_field
intrinsic_field
    0
    1
"""
        )
        itest = MuSpinInput(stest)
        ertest = ExperimentRunner(itest)

        results = ertest.run()

        self.assertAlmostEqual(results[0], 0.5 / (1.0 + 4 * np.pi**2 * tau**2))

    def test_run_results_function(self):

        # Empty system, modifying range to be 0 to 1
        stest = StringIO(
            """
spins
    e mu
time
    range(0, 10)
results_function
    2*y
"""
        )
        itest = MuSpinInput(stest)
        ertest = ExperimentRunner(itest)

        results = ertest.run()

        self.assertTrue(np.all(results == 1))

        # Simple system
        stest = StringIO(
            """
spins
    e mu
time
    range(0, 10)
zeeman 2
    0 0 1.0/muon_gyr
results_function
    2*y/e
"""
        )
        itest = MuSpinInput(stest)
        ertest = ExperimentRunner(itest)

        results = ertest.run()
        times = ertest.config.x_axis_values

        self.assertTrue(np.all(np.isclose(results, np.cos(2 * np.pi * times) / np.e)))

        # Results function is expected to be ignored by experiment when
        # fitting variables are introduced (as it would be applied by
        # FittingRunner in normal, command-line usage)
        stest = StringIO(
            """
spins
    e mu
time
    range(0, 10)
results_function
    2*y
fitting_variables
    g
fitting_data
    0 0.5
    10 0.5
"""
        )
        itest = MuSpinInput(stest)
        ertest = ExperimentRunner(itest, variables={"g": 1.0})

        results = ertest.run()

        self.assertTrue(np.all(results == 0.5))

    def test_run_intrinsic_field(self):
        # Check results from rotating are different when using field or intrinsic_field
        stest = StringIO(
            """
spins
    e mu
time
    range(0, 10)
zeeman 2
    0 0 1.0/muon_gyr
orientation
    zcw(1)
field
    10 0 0
"""
        )
        itest = MuSpinInput(stest)
        ertest = ExperimentRunner(itest)

        results = ertest.run()

        stest = StringIO(
            """
spins
    e mu
time
    range(0, 10)
zeeman 2
    0 0 1.0/muon_gyr
orientation
    zcw(1)
intrinsic_field
    10 0 0
"""
        )
        itest = MuSpinInput(stest)
        ertest = ExperimentRunner(itest)

        results_intrinsic = ertest.run()

        np.testing.assert_raises(
            AssertionError,
            np.testing.assert_array_almost_equal,
            results,
            results_intrinsic,
        )

    def test_run_celio(self):

        # Empty system
        stest = StringIO(
            """
spins
    e mu
time
    range(0, 10)
celio
    10
"""
        )
        itest = MuSpinInput(stest)
        ertest = ExperimentRunner(itest)

        with self.assertRaises(ValueError):
            results = ertest.run()

        # Simple system that should fail as has dissipation
        stest = StringIO(
            """
spins
    e mu
time
    range(0, 10)
zeeman 2
    0 0 1.0/muon_gyr
celio
    10
dissipation 1
    0.5
"""
        )
        itest = MuSpinInput(stest)
        ertest = ExperimentRunner(itest)

        with self.assertRaises(NotImplementedError):
            results = ertest.run()

        # Simple system
        stest = StringIO(
            """
spins
    e mu
time
    range(0, 10)
zeeman 2
    0 0 1.0/muon_gyr
celio
    10
"""
        )
        itest = MuSpinInput(stest)
        ertest = ExperimentRunner(itest)

        results = ertest.run()
        times = ertest.config.x_axis_values

        self.assertTrue(np.all(np.isclose(results, 0.5 * np.cos(2 * np.pi * times))))

        # Simple system using faster method - should fail as muon is not first
        stest = StringIO(
            """
spins
    e mu
time
    range(0, 10)
zeeman 2
    0 0 1.0/muon_gyr
celio
    10 8
"""
        )
        itest = MuSpinInput(stest)
        ertest = ExperimentRunner(itest)

        self.assertEqual(ertest.config.celio_k, 10)
        self.assertEqual(ertest.config.celio_averages, 8)

        # Check raises error when muon is not first
        with self.assertRaises(ValueError):
            results = ertest.run()

        # Simple system using faster method
        stest = StringIO(
            """
spins
    mu e
time
    range(0, 10)
zeeman 1
    0 0 1.0/muon_gyr
celio
    10 8
"""
        )
        itest = MuSpinInput(stest)
        ertest = ExperimentRunner(itest)

        results = ertest.run()
        times = ertest.config.x_axis_values

        # This test is subject to randomness, but np.isclose appears to avoid
        # any issues
        self.assertTrue(np.all(np.isclose(results, 0.5 * np.cos(2 * np.pi * times))))

        # Simple system using faster method with a temperature - should fail
        # as need T -> inf
        stest = StringIO(
            """
spins
    mu e
time
    range(0, 10)
zeeman 2
    0 0 1.0/muon_gyr
celio
    10 8
temperature
    1.0
"""
        )
        itest = MuSpinInput(stest)
        ertest = ExperimentRunner(itest)

        with self.assertRaises(ValueError):
            results = ertest.run()

    # For testing the faster time evolution method is run
    # correctly in the right scenarios
    def test_run_fast(self):

        # System without muon first should revert to the
        # slower method
        stest = StringIO(
            """
spins
    e mu
time
    range(0, 10)
zeeman 2
    0 0 1.0/muon_gyr
"""
        )
        itest = MuSpinInput(stest)
        ertest = ExperimentRunner(itest)

        results = ertest.run()
        times = ertest.config.x_axis_values

        self.assertFalse(ertest._T_inf_speedup)
        self.assertTrue(np.all(np.isclose(results, 0.5 * np.cos(2 * np.pi * times))))

        # System with the muon first should use the faster method
        stest = StringIO(
            """
spins
    mu e
time
    range(0, 10)
zeeman 1
    0 0 1.0/muon_gyr
"""
        )
        itest = MuSpinInput(stest)
        ertest = ExperimentRunner(itest)

        results = ertest.run()
        times = ertest.config.x_axis_values

        self.assertTrue(ertest._T_inf_speedup)
        self.assertTrue(np.all(np.isclose(results, 0.5 * np.cos(2 * np.pi * times))))

        # System with the a low temperature but no field should use the fast
        # method
        stest = StringIO(
            """
spins
    mu e
time
    range(0, 10)
temperature
    1
zeeman 1
    0 0 1.0/muon_gyr
"""
        )
        itest = MuSpinInput(stest)
        ertest = ExperimentRunner(itest)

        results = ertest.run()
        times = ertest.config.x_axis_values

        self.assertTrue(ertest._T_inf_speedup)
        self.assertTrue(np.all(np.isclose(results, 0.5 * np.cos(2 * np.pi * times))))

        # System with T -> inf but a zero field should still use the fast
        # method
        stest = StringIO(
            """
spins
    mu e
time
    range(0, 10)
field
    0.01
zeeman 1
    0 0 1.0/muon_gyr
"""
        )
        itest = MuSpinInput(stest)
        ertest = ExperimentRunner(itest)

        results = ertest.run()
        times = ertest.config.x_axis_values

        self.assertTrue(ertest._T_inf_speedup)

        # System with a low temperature and a non-zero field should not use
        # the fast method
        stest = StringIO(
            """
spins
    mu e
time
    range(0, 10)
field
    0.01
temperature
    1
zeeman 1
    0 0 1.0/muon_gyr
"""
        )
        itest = MuSpinInput(stest)
        ertest = ExperimentRunner(itest)

        results = ertest.run()
        times = ertest.config.x_axis_values

        self.assertFalse(ertest._T_inf_speedup)

    def test_dissipation(self):

        # Simple system
        g = 1.0
        stest = StringIO(
            """
spins
    mu
dissipation 1
    {g}
""".format(
                g=g
            )
        )

        itest = MuSpinInput(stest)
        ertest = ExperimentRunner(itest)

        times = ertest.config.x_axis_values

        results = ertest.run()

        solx = np.real(0.5 * np.exp(-g * times))
        self.assertTrue(np.all(np.isclose(results, solx)))

        # Check for temperature equilibrium
        T = 0.1
        stest = StringIO(
            """
spins
    mu
dissipation 1
    {g}
temperature
    {T}
field
    -1.0
time
    0
    20
""".format(
                g=g, T=T
            )
        )

        itest = MuSpinInput(stest)
        ertest = ExperimentRunner(itest)

        times = ertest.config.x_axis_values

        beta = 1.0 / (cnst.k * T)
        Z = np.exp(-cnst.h * constants.MU_GAMMA * 1e6 * beta)

        # Just let it evolve, see the result at long times
        Sz = ertest.system.operator({0: "z"})

        # Ready the first config
        ertest.load_config(ertest.config[0])
        results = ertest.Htot.evolve(ertest.rho0, times, operators=[Sz])

        self.assertAlmostEqual(np.real(results[-1, 0]), 0.5 * (1 - Z) / (1 + Z))
