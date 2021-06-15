import cProfile
import numpy as np
from muspinsim.experiment import MuonExperiment

exp = MuonExperiment(['e', 'mu', 'H'])
exp.spin_system.add_hyperfine_term(1, np.diag([1,1,4])*100)
exp.spin_system.add_hyperfine_term(2, np.diag([1,1,4])*100/3)
exp.set_powder_average(100)
exp.set_magnetic_field(5.0)
exp.set_dissipation_coupling(0, 1.0)
exp.set_muon_polarization('z')

times = np.linspace(0, 1, 1000)
op = exp.spin_system.operator({1: 'z'})

cProfile.run('exp.run_experiment(times, [op], "i")', sort='cumulative')