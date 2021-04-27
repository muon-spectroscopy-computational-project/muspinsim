import unittest
import numpy as np
import scipy.constants as cnst

from muspinsim import constants
from muspinsim.spinop import SpinOperator, DensityOperator
from muspinsim.spinsys import SpinSystem
from muspinsim.hamiltonian import Hamiltonian
from muspinsim.experiment import MuonExperiment


class TestExperiment(unittest.TestCase):

    def test_create(self):

        muexp = MuonExperiment(['e', 'mu'])
        self.assertEqual(muexp.spin_system.elec_indices, {0})
        self.assertEqual(muexp.spin_system.muon_index, 1)

        gmu = constants.MU_GAMMA
        ge = constants.ELEC_GAMMA

        self.assertTrue(np.all(muexp._Hz.matrix == np.diag([0.5*(ge+gmu),
                                                            0.5*(ge-gmu),
                                                            0.5*(-ge+gmu),
                                                            0.5*(-ge-gmu)])))
        self.assertTrue(isinstance(muexp._Hz, Hamiltonian))

    def test_powder(self):

        muexp = MuonExperiment()

        N = 20
        muexp.set_powder_average(N)
        self.assertTrue(muexp.orientations.shape[0] == muexp.weights.shape[0])
        self.assertTrue(muexp.weights.shape[0] >= N)

        # Are these correct? Basic test
        muexp.set_powder_average(1000)
        o = muexp.orientations
        w = muexp.weights

        f = 3*np.cos(o[:, 0])**2-1
        self.assertAlmostEqual(np.sum(f*w), 0.0, 5)

    def test_rho0(self):

        muexp = MuonExperiment(['e', 'mu'])
        rho0 = muexp.get_starting_state()

        self.assertTrue(np.all(np.isclose(rho0.matrix,
                                          [[0.25, 0.25, 0,    0],
                                           [0.25, 0.25, 0,    0],
                                              [0, 0,    0.25, 0.25],
                                              [0, 0,    0.25, 0.25]])))

        gmu = constants.MU_GAMMA
        ge = constants.ELEC_GAMMA

        T = 100
        muexp.set_magnetic_field(2.0e-6*cnst.k*T/(ge*cnst.h))
        muexp.set_temperature(T)
        muexp.set_muon_polarization('z')

        rho0 = muexp.get_starting_state()

        Z = np.exp([-1, 1])
        Z /= np.sum(Z)

        self.assertTrue(np.all(np.isclose(np.diag(rho0.matrix),
                                          [Z[0], 0, Z[1], 0])))

    def test_run(self):

        # Empty system
        muexp = MuonExperiment(['e', 'mu'])
        times = np.linspace(0, 10)

        results = muexp.run_experiment(times)

        self.assertTrue(np.all(results['e'] == 0.5))

        # Simple system
        muexp.spin_system.add_linear_term(1, [0, 0, 1.0])
        results = muexp.run_experiment(times, acquire='ei')

        tau = constants.MU_TAU

        self.assertTrue(np.all(np.isclose(results['e'][:, 0],
                                          0.5*np.cos(2*np.pi*times))))
        self.assertAlmostEqual(results['i'][0, 0],
                               0.5/(1.0+4*np.pi**2*tau**2))

    def test_dissipation(self):

        # Simple system
        g = 1.0
        muexp = MuonExperiment(['mu'])
        muexp.spin_system.set_dissipation(0, g)

        times = np.linspace(0, 10)

        results = muexp.run_experiment(times)
        evol = results['e']

        solx = np.real(0.5*np.exp(-np.pi*g*times))
        self.assertTrue(np.all(np.isclose(evol[:, 0], solx)))

        # Check for temperature equilibrium
        T = 0.1
        muexp.set_magnetic_field(-1.0)
        muexp.set_temperature(T)

        beta = 1.0/(cnst.k*T)
        Z = np.exp(-cnst.h*constants.MU_GAMMA*1e6*beta)

        # Just let it evolve, see the result at long times
        Sz = muexp.spin_system.operator({0: 'z'})
        rhoinf = muexp.run_experiment([20], Sz)['e']

        self.assertAlmostEqual(rhoinf[0, 0], 0.5*(1-Z)/(1+Z))

    def test_slices(self):

        gmu = constants.MU_GAMMA

        muexp = MuonExperiment(['mu'])
        muexp.spin_system.add_zeeman_term(0, 1.0/gmu)
        muexp.set_powder_average(100)

        times = np.linspace(0, 1.0)

        cos = np.cos(2*np.pi*times)
        signal = muexp.run_experiment(times)['e'][:, 0]

        self.assertTrue(np.all(np.isclose(signal,
                                          0.5*(2.0/3.0 +
                                               1.0/3.0*cos),
                                          atol=1e-3)))

        # Now slice
        signal_0 = muexp.run_experiment(times,
                                        orient_slice=slice(0, 1))['e'][:, 0]

        self.assertTrue(np.all(np.isclose(signal_0,
                                          0.5*cos*muexp.weights[0],
                                          atol=1e-3)))
