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
        muexp.set_starting_state()

        self.assertTrue(np.all(np.isclose(muexp.rho0.matrix,
                                          [[0.25, 0.25, 0,    0],
                                           [0.25, 0.25, 0,    0],
                                              [0, 0,    0.25, 0.25],
                                              [0, 0,    0.25, 0.25]])))

        gmu = constants.MU_GAMMA
        ge = constants.ELEC_GAMMA

        T = 100
        muexp.set_magnetic_field(2.0e-6*cnst.k*T/(ge*cnst.h))
        muexp.set_starting_state('z', T=T)

        Z = np.exp([-1, 1])
        Z /= np.sum(Z)

        self.assertTrue(np.all(np.isclose(np.diag(muexp.rho0.matrix),
                                          [Z[0], 0, Z[1], 0])))

    def test_run(self):

        # Empty system
        muexp = MuonExperiment(['e', 'mu'])
        muexp.set_starting_state()

        results = muexp.run_experiment()

        self.assertTrue(np.all(results['e'] == 0.5))

        # Simple system
        muexp.spin_system.add_linear_term(1, [0, 0, 1.0])
        results = muexp.run_experiment(acquire='ei')

        tau = constants.MU_TAU

        self.assertTrue(np.all(np.isclose(results['e'][:, 0],
                                          0.5*np.cos(2*np.pi*results['t']))))
        self.assertAlmostEqual(results['i'][0],
                               0.5*tau/(1.0+4*np.pi**2*tau**2))
        
