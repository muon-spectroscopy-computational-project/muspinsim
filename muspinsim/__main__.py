import os
import numpy as np
import argparse as ap
from soprano.calculate.powder import ZCW, SHREWD
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

    Is = H.spin_system._Is
    vectors = [[1, 0, 0]]*len(Is)

    if params.polarization == 'longitudinal':
        vectors[H.mu] = [0, 0, 1]
        observable = H.spin_system.operator({H.mu: 'z'})
    elif params.polarization == 'transverse':
        vectors[H.mu] = [1, 0, 0]
        observable = H.spin_system.operator({H.mu: 'x'})
    else:
        raise RuntimeError(
            'Invalid polarization {0}'.format(params.polarization))

    rho0 = DensityOperator.from_vectors(Is,
                                        vectors,
                                        [0 if i == H.mu else 1
                                         for i in range(len(Is))])

    results = {'fields': fields, 'field_scan': []}
    # First loop: fields
    for B in fields:

        H.set_B_field(B)

        for o, w in zip(orients, weights):
            pass
            
        for s in params.save:

            if s == 'evolution':
                pass
            elif s == 'integral':
                pass
            else:
                raise RuntimeError('Invalid save mode ' + s)


def main():
    # Entry point for script

    parser = ap.ArgumentParser()
    parser.add_argument('input_file', type=str, default=None, help="""YAML
                        formatted file with input parameters.""")
    args = parser.parse_args()

    seed = os.path.splitext(args.input_file)[0]

    fs = open(args.input_file)
    params = MuSpinInput(fs)

    logfile = open(seed + '.log', 'w')

    H = build_hamiltonian(params, logfile)

    expdata = perform_experiment(H, params, logfile)

    # # Start by creating the base Hamiltonian
    # H = MuonHamiltonian(params['spins'])

    # if params['experiment'] == 'transverse':
    #     transverse_experiment(H, params)

    logfile.close()


if __name__ == '__main__':
    main()
