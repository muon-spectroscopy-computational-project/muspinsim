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
        self.assertTrue(ertest.Hsys == ertest.Hz)

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
                ertest.Hz.matrix
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
                    rho0.matrix,
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

        self.assertTrue(np.all(np.isclose(np.diag(rho0.matrix), [Z[0], 0, Z[1], 0])))

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
