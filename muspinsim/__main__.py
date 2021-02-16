import os
import numpy as np
import argparse as ap
from datetime import datetime

from muspinsim.input import MuSpinInput
from muspinsim.experiment import MuonExperiment


def build_experiment(params, logfile=None):

    experiment = MuonExperiment(params.spins)
    if logfile:
        logfile.write('Hamiltonian created with spins:\n')
        logfile.write(', '.join(map(str, params.spins)) + '\n\n')

    for i, B in params.zeeman.items():
        experiment.spin_system.add_zeeman_term(i, B)
        if logfile:
            logfile.write('Added zeeman term to spin {0}\n'.format(i+1))

    for (i, j), A in params.hyperfine.items():
        experiment.spin_system.add_hyperfine_term(i, np.array(A), j)
        if logfile:
            logfile.write('Added hyperfine term to spin {0}\n'.format(i+1))

    for (i, j), r in params.dipolar.items():
        experiment.spin_system.add_dipolar_term(i, j, r)
        if logfile:
            logfile.write('Added dipolar term to spins '
                          '{0}, {1}\n'.format(i+1, j+1))

    for i, EFG in params.quadrupolar.items():
        experiment.spin_system.add_quadrupolar_term(i, EFG)
        if logfile:
            logfile.write('Added quadrupolar term to spin {0}\n'.format(i+1))

    for i, d in params.dissipation.items():
        experiment.spin_system.set_dissipation(i, d)
        if logfile:
            logfile.write('Set dissipation parameter for spin '
                          '{0} to {1} MHz\n'.format(i+1, d))

    if logfile:
        logfile.write('\n' + '*'*20 + '\n\n')

    return experiment


def run_experiment(experiment, params, logfile=None):

    trange = list(params.time)
    if len(trange) == 1:
        trange += [trange[0], 1]
    elif len(trange) == 2:
        trange += [100]
    elif len(trange) != 3:
        raise RuntimeError('Invalid time range')

    if logfile:
        logfile.write('Using time range: '
                      '{0} to {1} μs in {2} steps\n\n'.format(*trange))

    times = np.linspace(trange[0], trange[1], int(trange[2]))

    brange = list(params.field)
    if len(brange) == 1:
        brange += [brange[0], 1]
    elif len(brange) == 2:
        brange += [100]
    elif len(brange) != 3:
        raise RuntimeError('Invalid field range')

    fields = np.linspace(brange[0], brange[1], int(brange[2]))

    if logfile:
        logfile.write('Using field range: '
                      '{0} to {1} T in {2} steps\n\n'.format(*brange))

    # Powder averaging
    if params.powder is None:
        experiment.set_single_crystal(0, 0)
    else:
        scheme = params.powder[0]
        N = params.powder[1]
        experiment.set_powder_average(N, scheme)

        if logfile:
            logfile.write('Using powder averaging scheme '
                          '{0}\n'.format(scheme.upper()))
            logfile.write(
                '{0} orientations generated\n\n'.format(
                    len(experiment.weights)))

    if params.polarization == 'longitudinal':
        muaxis = 'z'
    elif params.polarization == 'transverse':
        muaxis = 'x'
    else:
        raise RuntimeError(
            'Invalid polarization {0}'.format(params.polarization))

    ssys = experiment.spin_system
    observable = ssys.operator({ssys.muon_index: muaxis})

    results = {'fields': fields, 'times': times, 'field_scan': []}

    if 'integral' in params.save:
        integrated_values = []

    # First loop: fields
    for B in fields:

        if logfile is not None:
            logfile.write('Performing calculations for B = {0} T\n'.format(B))

        experiment.set_magnetic_field(B)
        experiment.set_muon_polarization(muaxis)
        experiment.set_temperature(params.temperature)

        evol_results = []
        intgr_results = []

        acquire = [p[0] for p in params.save]

        field_results = experiment.run_experiment(times,
                                                  operators=[observable],
                                                  acquire=acquire)

        results['field_scan'].append(field_results)

        if logfile is not None:
            logfile.write('\n\n')

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

    experiment = build_experiment(params, logfile)
    expdata = run_experiment(experiment, params, logfile)

    tend = datetime.now()

    logfile.write('Simulation completed in '
                  '{0:.3f} seconds\n'.format((tend-tstart).total_seconds()) +
                  '*'*20 + '\n')

    x = expdata['fields']
    t = expdata['times']
    y = expdata['field_scan']

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

    # # Start by creating the base Hamiltonian
    # H = MuonHamiltonian(params['spins'])

    # if params['experiment'] == 'transverse':
    #     transverse_experiment(H, params)

    logfile.close()


if __name__ == '__main__':
    main()
