"""experiment.py

A class defining a whole muon experiment"""

import numpy as np
from numbers import Number
from scipy import constants as cnst
from soprano.calculate.powder import ZCW, SHREWD

from muspinsim.constants import MU_TAU
from muspinsim.spinop import DensityOperator, SpinOperator, Operator
from muspinsim.spinsys import MuonSpinSystem
from muspinsim.hamiltonian import Hamiltonian


def _make_rotmat(theta, phi):

    ct = np.cos(theta)
    st = np.sin(theta)
    cp = np.cos(phi)
    sp = np.sin(phi)

    return np.array([
        [cp*ct, -sp, cp*st],
        [sp*ct,  cp, sp*st],
        [-st,     0,    ct]])


class MuonExperiment(object):

    def __init__(self, spins=['e', 'mu']):

        if len(spins) == 0:
            raise ValueError('At least one spin must be included')

        self._spin_system = MuonSpinSystem(spins)
        self._orientations = [[0.0, 0.0]]
        self._weights = [1.0]

        # Zeeman Hamiltonian?
        zops = [self._spin_system.operator({i: 'z'})*self._spin_system.gamma(i)
                for i in range(len(spins))]

        self._Hz = zops[0]
        for o in zops[1:]:
            self._Hz += o

        self._Hz = Hamiltonian.from_spin_operator(self._Hz)
        self._B = 0

        self.set_starting_state()

    @property
    def spin_system(self):
        return self._spin_system

    @property
    def orientations(self):
        return np.array(self._orientations)

    @property
    def weights(self):
        return np.array(self._weights)

    @property
    def B(self):
        return self._B

    @property
    def rho0(self):
        return self._rho0

    def set_single_crystal(self, theta=0.0, phi=0.0):

        self._orientations = np.array([[theta, phi]])
        self._weights = np.ones(1)

    def set_powder_average(self, N=20, scheme='zcw'):

        try:
            scheme = scheme.lower()
            pwd = {'zcw': ZCW, 'shrewd': SHREWD}[scheme]('sphere')
        except KeyError:
            raise RuntimeError('Invalid powder averaging scheme ' +
                               scheme)

        orients, weights = pwd.get_orient_angles(N)

        self._orientations = orients
        self._weights = weights

    def set_starting_state(self, muon_axis='x', T=np.inf):
        # We assume a thermal starting state

        if isinstance(muon_axis, str):
            try:
                muon_axis = {
                    'x': [1, 0, 0],
                    'y': [0, 1, 0],
                    'z': [0, 0, 1]
                }[muon_axis]
            except KeyError:
                raise ValueError('muon_axis must be a vector or x, y or z')

        mu_i = self.spin_system.muon_index
        rhos = []

        for i, s in enumerate(self.spin_system.spins):
            I = self.spin_system.I(i)
            if i == mu_i:
                r = DensityOperator.from_vectors(I, muon_axis, 0)
            else:
                # Get the Zeeman Hamiltonian for this field
                Sz = SpinOperator.from_axes(I, 'z')
                E = np.diag(Sz.matrix)*self.spin_system.gamma(i)*self.B*1e6
                if T > 0:
                    Z = np.exp(-cnst.h*E/(cnst.k*T))
                else:
                    Z = np.where(E == np.amin(E), 1.0, 0.0)
                if np.sum(Z) > 0:
                    Z /= np.sum(Z)
                else:
                    Z = np.ones(len(E))/len(E)
                r = DensityOperator(np.diag(Z))

            rhos.append(r)

        self._rho0 = rhos[0]
        for r in rhos[1:]:
            self._rho0 = self._rho0.kron(r)

    def set_magnetic_field(self, B=0.0):
        self._B = B

    def run_experiment(self, times=np.linspace(0, 10),
                       operators=None,
                       acquire='e'):

        if operators is None:
            operators = [self.spin_system.operator({
                self.spin_system.muon_index: 'x'
            })]

        # Generate all rotated Hamiltonians
        orients, weights = self._orientations, self._weights

        rotmats = [_make_rotmat(t, p) for (t, p) in orients]
        Hz = self._Hz*self.B
        rho0 = self.rho0
        results = {'t': times, 'e': [], 'i': []}

        for R in rotmats:

            Hint = self.spin_system.rotate(R).hamiltonian
            H = Hz + Hint  # Total Hamiltonian

            if 'e' in acquire:
                # Evolution
                evol = H.evolve(rho0, times, operators)
                results['e'].append(evol)
            if 'i' in acquire:
                intg = H.integrate_decaying(rho0, MU_TAU, operators)
                results['i'].append(intg)

        # Averaging
        for k, data in results.items():
            if k == 't' or len(data) == 0:
                continue
            results[k] = np.average(data, axis=0, weights=weights)

        return results
