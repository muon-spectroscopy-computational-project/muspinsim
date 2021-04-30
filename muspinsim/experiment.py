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
from muspinsim.lindbladian import Lindbladian


def _make_rotmat(theta, phi):

    ct = np.cos(theta)
    st = np.sin(theta)
    cp = np.cos(phi)
    sp = np.sin(phi)

    return np.array([
        [cp*ct, -sp, cp*st],
        [sp*ct,  cp, sp*st],
        [-st,     0,    ct]])


def _make_lindbladian(ssys, B, T):

    H = ssys.hamiltonian
    g = ssys.gammas

    if T > 0:
        Zu = np.exp(-cnst.h*g*B*1e6/(cnst.k*T))
    else:
        Zu = g*0

    dops = []
    for i, d in enumerate(ssys.dissipation_factors):
        if (d != 0):
            dops.append((ssys.operator({i: '-'}), d/(1+Zu[i])))
            dops.append((ssys.operator({i: '+'}), d*Zu[i]/(1+Zu[i])))

    return Lindbladian.from_hamiltonian(H, dops)


class MuonExperiment(object):

    def __init__(self, spins=['e', 'mu']):
        """Create a MuonExperiment object

        Create a "virtual spectrometer" that can be used to carry out any
        number of experiments on a given SpinSystem.

        Keyword Arguments:
            spins {list} -- The spins in the system. See SpinSystem for
                            details (default: {['e', 'mu']})

        Raises:
            ValueError -- If an empty spins list is passed
        """

        if len(spins) == 0:
            raise ValueError('At least one spin must be included')

        self._spin_system = MuonSpinSystem(spins)
        self._orientations = [[0.0, 0.0]]
        self._weights = [1.0]
        self._default_operators = [self.spin_system.operator({
            self.spin_system.muon_index: 'x'
        })]

        # Zeeman Hamiltonian?
        zops = [self._spin_system.operator({i: 'z'})*self._spin_system.gamma(i)
                for i in range(len(spins))]

        self._Hz = zops[0]
        for o in zops[1:]:
            self._Hz += o

        self._Hz = Hamiltonian.from_spin_operator(self._Hz)
        self._Lz = Lindbladian.from_hamiltonian(self._Hz)
        self._B = 0
        self._T = np.inf
        self._mupol = [1, 0, 0]
        self._rho0 = None

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
    def T(self):
        return self._T

    @property
    def muon_polarization(self):
        return np.array(self._mupol)

    def set_single_crystal(self, theta=0.0, phi=0.0):
        """Set a single crystal orientation

        Make the sample have a single specified crystallite orientation

        Keyword Arguments:
            theta {number} -- Polar angle (default: {0.0})
            phi {number} -- Azimuthal angle (default: {0.0})
        """

        self._orientations = np.array([[theta, phi]])
        self._weights = np.ones(1)

    def set_powder_average(self, N=20, scheme='zcw'):
        """Set a powder average method

        Set a scheme and a number of angles for a powder averaging algorithm,
        representing a polycrystalline or powdered sample.

        Keyword Arguments:
            N {number} -- Minimum number of orientations to use (default: {20})
            scheme {str} -- Powder averaging scheme, either 'zcw' or 'shrewd'
                            (default: {'zcw'})

        Raises:
            ValueError -- Invalid powder averaging scheme
        """

        try:
            scheme = scheme.lower()
            pwd = {'zcw': ZCW, 'shrewd': SHREWD}[scheme]('sphere')
        except KeyError:
            raise ValueError('Invalid powder averaging scheme ' +
                             scheme)

        orients, weights = pwd.get_orient_angles(N)

        self._orientations = orients
        self._weights = weights

    def set_magnetic_field(self, B=0.0):
        """Set the magnetic field

        Set the magnetic field applied to the sample, always pointing along
        the Z axis in the laboratory frame.

        Keyword Arguments:
            B {number} -- Magnetic field in Tesla (default: {0.0})
        """

        self._rho0 = None
        self._B = B

    def set_muon_polarization(self, muon_axis='x'):
        """Set muon initial polarization

        Set the direction of the muon's initial polarization.

        Keyword Arguments:
            muon_axis {str|ndarray} -- String or vector defining a direction
                                       for the starting muon polarization 
                                       (default: {'x'})        
        Raises:
            ValueError -- Invalid muon_axis
        """

        if isinstance(muon_axis, str):
            try:
                muon_axis = {
                    'x': [1, 0, 0],
                    'y': [0, 1, 0],
                    'z': [0, 0, 1],
                    '-x': [-1, 0, 0],
                    '-y': [0, -1, 0],
                    '-z': [0, 0, -1]
                }[muon_axis]
            except KeyError:
                raise ValueError('muon_axis must be a vector or x, y or z')
        else:
            muon_axis = np.array(muon_axis)
            muon_axis /= np.linalg.norm(muon_axis)

        self._rho0 = None
        self._mupol = muon_axis

    def set_temperature(self, T=np.inf):
        """Sets the temperature of the experiment. 

        Sets the temperature of the experiment in Kelvin. Determines the
        initial thermal starting state and the temperature of the thermal bath
        if the system contains dissipation terms.

        Keyword Arguments:
            T {Number} -- Temperature in Kelvin (default: {np.inf})

        Raises:
            ValueError -- Invalid temperature
        """

        if (T < 0):
            raise ValueError('Can not set a negative temperature')

        self._rho0 = None
        self._T = T

    def get_starting_state(self):
        """Return the starting quantum state for the system

        Build the starting quantum state for the system as a coherently 
        polarized muon + a thermal density matrix (using only the Zeeman 
        terms) for every other spin for the current magnetic field,
        temperature, and muon polarization axis.

        Returns:
            DensityOperator -- The starting density matrix
        """

        if self._rho0 is None:
            T = self._T
            muon_axis = self._mupol

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

        return self._rho0

    def run_experiment(self, times=[0],
                       operators=None,
                       acquire='e',
                       orient_slice=None):
        """Run an experiment

        Run an experiment by evolving or integrating the starting state under
        the Hamiltonian of the system for the given times, and measuring the
        given quantity.

        Keyword Arguments:
            times {list} -- Times to sample evolution at (default: {[0]})
            operators {list} -- List of operators to measure expectation values 
                                of (default: {None})
            acquire {str} -- Whether to record the evolution ('e') of the 
                             expectation values, or their integral ('i') 
                             convolved with the muon's exponential decay, 
                             or both ('ei', 'ie') (default: {'e'})
            orient_slice {slice} -- Slice of orientations to run the
                                    experiment on. Useful for parallelisation.
                                    If None, use all of them (default: None)
        Returns:
            dict -- Dictionary of results.
        """

        if operators is None:
            operators = self._default_operators

        # Generate all rotated Hamiltonians
        orients, weights = self._orientations, self._weights

        if orient_slice is None:
            orient_slice = slice(0, None)
        orients = np.array(orients[orient_slice])
        weights = np.array(weights[orient_slice])

        rotmats = [_make_rotmat(t, p) for (t, p) in orients]
        
        Hz = self._Hz*self.B
        rho0 = self.get_starting_state()

        results = {'e': [], 'i': []}

        use_dissipation = self.spin_system.is_dissipative

        if use_dissipation:
            Hz = self._Lz*self.B
            Ls = []

        for R in rotmats:

            rotsys = self.spin_system.rotate(R.T)

            if use_dissipation:
                Hint = _make_lindbladian(rotsys, self.B, self.T)
            else:
                Hint = rotsys.hamiltonian

            H = Hz + Hint  # Total Hamiltonian/Lindbladian

            if 'e' in acquire:
                # Evolution
                evol = H.evolve(rho0, times, operators)
                results['e'].append(evol)
            if 'i' in acquire:
                intg = H.integrate_decaying(rho0, MU_TAU,
                                            operators)/MU_TAU
                results['i'].append(intg)

        # Averaging
        for k, data in results.items():
            if len(data) == 0:
                continue
            data = np.array(data)
            results[k] = np.real(np.sum(data*weights[:, None, None], axis=0))

        return results
