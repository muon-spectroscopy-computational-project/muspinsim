import os
import numpy as np
import argparse as ap
from datetime import datetime

from muspinsim.input import MuSpinInput
from muspinsim.experiment import MuonExperiment
from muspinsim.mpi import mpi_controller as mpi


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

    @mpi.execute_on_root
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
            self.experiment.set_dissipation_coupling(i, d)
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
                    len(self.experiment.weights)))

        if params.polarization == 'longitudinal':
            self.muon_axis = 'z'
        elif params.polarization == 'transverse':
            self.muon_axis = 'x'
        else:
            raise RuntimeError(
                'Invalid polarization {0}'.format(params.polarization))
        self.experiment.set_muon_polarization(self.muon_axis)
        self.log('Muon beam polarized along axis {0}\n'.format(self.muon_axis))

        # Temperature
        self.temperature = params.temperature
        self.experiment.set_temperature(self.temperature)
        self.log('Using temperature of {0} K\n'.format(self.temperature))

        # What to save
        ssys = self.experiment.spin_system
        self.observable = ssys.operator({ssys.muon_index: self.muon_axis})
        self.save = [p[0] for p in params.save]

        self.log('*'*20 + '\n')

    @mpi.execute_on_root
    def log(self, message):
        if self._log:
            self._log.write(message + '\n')

    def broadcast(self):
        # Broadcast the contents of this object to other MPI threads
        mpi.broadcast_object(self, only=[
            'experiment',
            'field_axis',
            'time_axis',
            'muon_axis',
            'temperature',
            'observable',
            'save'
        ])

    def run(self):

        self.broadcast()

        exp = self.experiment
        ssys = self.experiment.spin_system

        if ssys.is_dissipative:
            self.log('Spin system is dissipative; using Lindbladian')

        # Now slicing the values
        N_f = len(self.field_axis)   # Fields
        N_o = len(exp.weights)       # Orientations

        results = {
            'e': None,
            'i': None
        }
        if 'e' in self.save:
            results['e'] = np.zeros((N_f, len(self.time_axis)))
        if 'i' in self.save:
            results['i'] = np.zeros(N_f)

        # Split the tasks
        tasks = mpi.split_2D(range(N_f), range(N_o))
        field_scan, orient_slice = tasks[mpi.rank]

        if len(tasks) > 1:
            tsizes = [len(t[0])*len(t[1]) for t in tasks]
            self.log('Splitting jobs over {0} cores'.format(mpi.size))
            self.log('Job sizes:\n\t' + ' '.join(map(str, tsizes)) + '\n')

        # Loop over fields
        for i in field_scan:
            B = self.field_axis[i]

            self.log('Performing calculations for B = {0} T'.format(B))

            exp.set_magnetic_field(B)

            field_results = exp.run_experiment(self.time_axis,
                                               operators=[self.observable],
                                               acquire=self.save,
                                               orient_slice=orient_slice)

            if 'e' in self.save:
                results['e'][i] = field_results['e'][:, 0]
            if 'i' in self.save:
                results['i'][i] = field_results['i'][0]

        # Reduce results
        for k, v in results.items():
            if v is None:
                continue
            results[k] = mpi.sum_data(v)

        self.log('\n' + '*'*20 + '\n')

        return results


def main(use_mpi=False):

    if use_mpi:
        mpi.connect()

    if mpi.is_root:
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
            params.name = os.path.splitext(
                os.path.split(args.input_file)[1])[0]

        logfile = open(params.name + '.log', 'w')

        tstart = datetime.now()
    else:
        params = MuSpinInput()
        logfile = None

    setup = MuonExperimentalSetup(params, logfile)
    data = setup.run()

    if mpi.is_root:
        tend = datetime.now()

        simtime = (tend-tstart).total_seconds()
        logfile.write('Simulation completed in '
                      '{0:.3f} seconds\n\n'.format(simtime) +
                      '*'*20 + '\n')

        x = setup.field_axis
        t = setup.time_axis
        if 'evolution' in params.save:
            y = data['e']
            # Save evolution files
            for B, d in zip(x, y):
                fname = os.path.join(path, params.name +
                                     '_B{0}_evol.dat'.format(B))
                d = np.real(d)
                np.savetxt(fname, np.array([t, d]).T,
                           header="B field = {0} T".format(B))

        if 'integral' in params.save:
            y = data['i']
            fname = os.path.join(path, params.name +
                                 '_intgr.dat')

            np.savetxt(fname, np.array([x, y]).T)

        logfile.close()


def main_mpi():
    main(use_mpi=True)


if __name__ == '__main__':
    main()
