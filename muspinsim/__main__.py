import os
import numpy as np
import argparse as ap
from datetime import datetime

from soprano.calculate.powder import ZCW, SHREWD
from muspinsim.constants import MU_TAU
from muspinsim.spinop import SpinOperator, DensityOperator
from muspinsim.hamiltonian import MuonHamiltonian
from muspinsim.input import MuSpinInput


def build_hamiltonian(params, logfile=None):

    H = MuonHamiltonian(params.spins)
    if logfile:
        logfile.write('Hamiltonian created with spins:\n')
        logfile.write(', '.join(params.spins) + '\n\n')

    for i, A in params.hyperfine.items():
        H.add_hyperfine_term(i-1, np.array(A))
        if logfile:
            logfile.write('Added hyperfine term to spin {0}\n'.format(i))

    for (i, j), r in params.dipolar.items():
        H.add_dipolar_term((i-1, j-1), r)
        if logfile:
            logfile.write('Added dipolar term to spins '
                          '{0}, {1}\n'.format(i, j))

    for i, EFG in params.quadrupolar.items():
        H.add_quadrupolar_term(i-1, EFG)
        if logfile:
            logfile.write('Added quadrupolar term to spin {0}\n'.format(i))

    if logfile:
        logfile.write('\n' + '*'*20 + '\n\n')

    return H


def make_rotmat(theta, phi):

    ct = np.cos(theta)
    st = np.sin(theta)
    cp = np.cos(phi)
    sp = np.sin(phi)

    return np.array([
        [cp*ct, -sp, cp*st],
        [sp*ct,  cp, sp*st],
        [-st,     0,    ct]])


def run_branches(H, times, rho0, observable, params, logfile=None):

    Hs = []

    if params.branch is None or H.e is None:
        Hs.append(H)
    else:
        if logfile is not None:
            logfile.write('Using electronic branches: '
                          '{0}\n'.format(params.branch))

        for b in params.branch:
            Hs.append(H.reduced_hamiltonian(branch=b))

    evol = []
    if 'evolution' in params.save:
        for H in Hs:
            evol.append(H.evolve(rho0, times, [observable])[:, 0])

        evol = np.average(evol, axis=0)

    intgr = []
    if 'integral' in params.save:
        for H in Hs:
            intgr.append(H.integrate_decaying(rho0, MU_TAU,
                                              [observable])[0]/MU_TAU)
        intgr = np.average(intgr, axis=0)

    return np.real(evol), np.real(intgr)


def perform_experiment(H, params, logfile=None):

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
        orients, weights = np.array([[np.pi/2.0, 0]]), np.array([1.0])
    else:
        try:
            scheme = params.powder[0].lower()
            pwd = {'zcw': ZCW, 'shrewd': SHREWD}[scheme]('sphere')
        except KeyError:
            raise RuntimeError('Invalid powder averaging scheme ' +
                               params.powder[0])
        N = params.powder[1]
        orients, weights = pwd.get_orient_angles(N)

        if logfile:
            logfile.write('Using powder averaging scheme '
                          '{0}\n'.format(scheme.upper()))
            logfile.write(
                '{0} orientations generated\n\n'.format(len(weights)))

    reduced_H = (params.branch is not None) and (H.e is not None)

    Is = H.spin_system._Is
    vectors = [[1, 0, 0]]*len(Is)

    if params.polarization == 'longitudinal':
        vectors[H.mu] = [0, 0, 1]
        muaxis = 'z'
    elif params.polarization == 'transverse':
        vectors[H.mu] = [1, 0, 0]
        muaxis = 'x'
    else:
        raise RuntimeError(
            'Invalid polarization {0}'.format(params.polarization))

    observable = H.spin_system.operator({H.mu: muaxis})

    if reduced_H:
        axes = [muaxis if i == H.mu else '0' for i, s in enumerate(Is)]
        Is = np.delete(Is, H.e)
        vectors = np.delete(vectors, H.e, axis=0)
        axes = np.delete(axes, H.e)
        observable = SpinOperator.from_axes(Is, axes)

    rho0 = DensityOperator.from_vectors(Is,
                                        vectors,
                                        [0 if i == H.mu else 1
                                         for i in range(len(Is))])

    results = {'fields': fields, 'times': times, 'field_scan': []}

    if 'integral' in params.save:
        integrated_values = []

    # First loop: fields
    for B in fields:

        if logfile is not None:
            logfile.write('Performing calculations for B = {0} T\n'.format(B))

        H.set_B_field(B)

        evol_results = []
        intgr_results = []

        for o, w in zip(orients, weights):
            R = make_rotmat(*o)
            rH = H.rotate(R)
            evol, intgr = run_branches(rH, times, rho0, observable, params,
                                       logfile)
            evol_results.append(evol*w)
            intgr_results.append(intgr*w)

        field_results = {
            'evolution': np.sum(evol_results, axis=0),
            'integral': np.sum(intgr_results, axis=0)
        }

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

    H = build_hamiltonian(params, logfile)

    expdata = perform_experiment(H, params, logfile)

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
            np.savetxt(fname, np.array([t, scandata['evolution']]).T,
                       header="B field = {0} T".format(B))

    if 'integral' in params.save:

        fname = os.path.join(path, params.name +
                             '_intgr.dat')

        data = [sd['integral'] for sd in y]

        np.savetxt(fname, np.array([x, data]).T)

    # # Start by creating the base Hamiltonian
    # H = MuonHamiltonian(params['spins'])

    # if params['experiment'] == 'transverse':
    #     transverse_experiment(H, params)

    logfile.close()


if __name__ == '__main__':
    main()
