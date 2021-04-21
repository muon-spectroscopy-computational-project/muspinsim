import os
import numpy as np
import argparse as ap
from datetime import datetime

from muspinsim.input import MuSpinInput
from muspinsim.experiment import MuonExperiment


def _make_range(rvals, default_n=100):
    rvals = list(rvals)
    l = len(rvals)
    if l == 1:
        rvals = [rvals[0], rvals[0], 1]  # Single value
    elif l == 2:
        rvals = rvals + [100]
    elif l > 3:
        raise RuntimeError('Invalid range definition has more than three '
                           'values')
    else:
        rvals[2] = int(rvals[2])

    return rvals


class MuonExperimentalSetup(object):

    def __init__(self, params, logfile=None):

        self._log = logfile

        # Create
        self.experiment = MuonExperiment(params.spins)

        self.log('Hamiltonian created with spins:')
        self.log(', '.join(map(str, params.spins)) + '\n')

        self.log('Adding Hamiltonian terms:')
        for i, B in params.zeeman.items():
            self.experiment.spin_system.add_zeeman_term(i, B)
            self.log('\tAdded zeeman term to spin {0}'.format(i+1))

        for (i, j), A in params.hyperfine.items():
            self.experiment.spin_system.add_hyperfine_term(i, np.array(A), j)
            self.log('\tAdded hyperfine term to spin {0}'.format(i+1))

        for (i, j), r in params.dipolar.items():
            self.experiment.spin_system.add_dipolar_term(i, j, r)
            self.log('\tAdded dipolar term to spins {0}, {1}'.format(i+1, j+1))

        for i, EFG in params.quadrupolar.items():
            self.experiment.spin_system.add_quadrupolar_term(i, EFG)
            self.log('\tAdded quadrupolar term to spin {0}'.format(i+1))

        for i, d in params.dissipation.items():
            self.experiment.spin_system.set_dissipation(i, d)
            self.log('\tSet dissipation parameter for spin '
                     '{0} to {1} MHz'.format(i+1, d))
        self.log('')

        # Ranges
        trange = _make_range(params.time)
        self.log('Using time range: '
                 '{0} to {1} μs in {2} steps\n'.format(*trange))
        self.time_axis = np.linspace(*trange)

        brange = _make_range(params.field)
        self.log('Using field range: '
                 '{0} to {1} T in {2} steps\n'.format(*brange))
        self.field_axis = np.linspace(*brange)

        # Powder averaging
        if params.powder is None:
            self.experiment.set_single_crystal(0, 0)
        else:
            scheme = params.powder[0]
            N = params.powder[1]

            self.experiment.set_powder_average(N, scheme)

            self.log('Using powder averaging scheme '
                     '{0}\n'.format(scheme.upper()))
            self.log(
                '{0} orientations generated\n'.format(
                    len(experiment.weights)))

        if params.polarization == 'longitudinal':
            self.muon_axis = 'z'
        elif params.polarization == 'transverse':
            self.muon_axis = 'x'
        else:
            raise RuntimeError(
                'Invalid polarization {0}'.format(params.polarization))
        self.log('Muon beam polarized along axis {0}\n'.format(self.muon_axis))

        # Temperature
        self.temperature = params.temperature
        self.log('Using temperature of {0} K\n'.format(self.temperature))

        # What to save
        self.save = params.save

        self.log('*'*20 + '\n')

    def log(self, message):
        if self._log:
            self._log.write(message + '\n')

    def run(self):

        exp = self.experiment
        ssys = self.experiment.spin_system

        if ssys.is_dissipative:
            self.log('Spin system is dissipative; using Lindbladian')

        observable = ssys.operator({ssys.muon_index: self.muon_axis})

        results = []

        # Loop over fields
        for B in self.field_axis:

            self.log('Performing calculations for B = {0} T'.format(B))

            exp.set_magnetic_field(B)
            exp.set_muon_polarization(self.muon_axis)
            exp.set_temperature(self.temperature)

            acquire = [p[0] for p in self.save]

            field_results = exp.run_experiment(self.time_axis,
                                               operators=[observable],
                                               acquire=acquire)

            results.append(field_results)

            self.log('\n')

        self.log('*'*20 + '\n')

        return results


def main():
    # Entry point for script

    parser = ap.ArgumentParser()
    parser.add_argument('input_file', type=str, default=None, help="""YAML
                        formatted file with input parameters.""")
    args = parser.parse_args()

    fs = open(args.input_file)
    params = MuSpinInput(fs)

    for s in params.save:
        if not (s in ('evolution', 'integral')):
            raise RuntimeError('Invalid save mode {0}'.format(s))

    path = os.path.split(args.input_file)[0]

    if params.name is None:
        params.name = os.path.splitext(os.path.split(args.input_file)[1])[0]

    logfile = open(params.name + '.log', 'w')

    tstart = datetime.now()

    setup = MuonExperimentalSetup(params, logfile)
    data = setup.run()

    tend = datetime.now()

    logfile.write('Simulation completed in '
                  '{0:.3f} seconds\n'.format((tend-tstart).total_seconds()) +
                  '*'*20 + '\n')

    x = setup.field_axis
    t = setup.time_axis
    y = np.array(data)

    if 'evolution' in params.save:
        # Save evolution files
        for B, scandata in zip(x, y):
            fname = os.path.join(path, params.name +
                                 '_B{0}_evol.dat'.format(B))
            d = np.real(scandata['e'][:, 0])
            np.savetxt(fname, np.array([t, d]).T,
                       header="B field = {0} T".format(B))

    if 'integral' in params.save:

        fname = os.path.join(path, params.name +
                             '_intgr.dat')

        data = [sd['i'][0] for sd in y]

        np.savetxt(fname, np.array([x, data]).T)

    logfile.close()


if __name__ == '__main__':
    main()
