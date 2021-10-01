"""experiment.py

Classes and functions to perform actual experiments"""

import numpy as np
import scipy.constants as cnst

from muspinsim.mpi import mpi_controller as mpi
from muspinsim.simconfig import MuSpinConfig, ConfigSnapshot
from muspinsim.input import MuSpinInput
from muspinsim.spinsys import MuonSpinSystem
from muspinsim.spinop import DensityOperator, SpinOperator

"""Calculate a thermal density matrix in which the system is prepared

Calculate an approximate thermal density matrix to prepare the system in,
with the muon polarised along a given direction and every other spin in a
thermal equilibrium decohered state.

Arguments:
    B {np.ndarray} -- Applied magnetic field in T
    p {np.ndarray} -- Muon polarization direction
    T {float} -- Temperature in K
    system {MuonSpinSystem} -- System for which to prepare the matrix

Returns:
    rho0 {DensityOperator} -- Density matrix at t=0
"""


class ExperimentRunner(object):
    """A class meant to run experiments. Its main purpose as an object is to
    provide caching for any quantities that might not need to be recalculated 
    between successive snapshots."""

    def __init__(self, infile: MuSpinInput, variables: dict = {}):
        """Set up an experiment as defined by a MuSpinInput object

        Prepare a set of calculations (for multiple files and averages) as
        defined by a MuSpinInput object and a set of variable values. Takes
        care of parallelism, splitting calculations across nodes etc.

        Arguments:
            infile {MuSpinInput} -- The input file object defining the 
                                    calculations we need to perform.
            variables {dict} -- The values of any variables appearing in the input
                                file
        """

        if mpi.is_root:
            # On root, we run the evaluation that gives us the actual possible
            # values for simulation configurations. These are then broadcast
            # across all nodes, each of which runs its own slice of them, and
            # finally gathered back together
            config = MuSpinConfig(infile.evaluate(**variables))
        else:
            config = MuSpinConfig()

        mpi.broadcast_object(config)

        self._config = config
        self._system = config.system

        # Parameters
        self._B = None
        self._p = None
        self._T = None

        # Basic Hamiltonian
        self._Hsys = self._system.hamiltonian.matrix

        # Derived quantities
        self._rho0 = None
        self._Hz = None
        self._Ld = None

    @property
    def B(self):
        return self._B

    @B.setter
    def B(self, x):
        self._B = x
        self._rho0 = None
        self._Hz = None
        self._Ld = None

    @property
    def p(self):
        return self._p

    @p.setter
    def p(self, x):
        self._p = x
        self._rho0 = None

    @property
    def T(self):
        return self._T

    @T.setter
    def T(self, x):
        self._T = x
        self._rho0 = None
        self._Ld = None

    @property
    def rho0(self):

        if self._rho0 is None:
            T = self._T
            muon_axis = self._p
            B = self._B

            mu_i = self._system.muon_index
            rhos = []

            for i, s in enumerate(self._system.spins):
                I = self._system.I(i)
                if i == mu_i:
                    r = DensityOperator.from_vectors(I, muon_axis, 0)
                else:
                    # Get the Zeeman Hamiltonian for this field
                    Hz = np.sum([B[j]*SpinOperator.from_axes(I, e).matrix
                                 for j, e in enumerate('xyz')],
                                axis=0)

                    evals, evecs = np.linalg.eigh(Hz)
                    E = evals*1e6*self._system.gamma(i)

                    if T > 0:
                        Z = np.exp(-cnst.h*E/(cnst.k*T))
                    else:
                        Z = np.where(E == np.amin(E), 1.0, 0.0)
                    if np.sum(Z) > 0:
                        Z /= np.sum(Z)
                    else:
                        Z = np.ones(len(E))/len(E)

                    rhoI = np.sum(evecs[:, None, :] *
                                  evecs[None, :, :].conj() *
                                  Z[None, None, :],
                                  axis=-1)

                    r = DensityOperator(rhoI)

                rhos.append(r)

            self._rho0 = rhos[0]
            for r in rhos[1:]:
                self._rho0 = self._rho0.kron(r)

        return self._rho0

    @property
    def Hz(self):
        if self._Hz is None:
            # Recalculate
            pass
        return self._Hz

    @property
    def Ld(self):
        if self._Ld is None:
            pass
        return self._Ld

    def run_all(self):
        """Run the experiment

        Run all calculations in the configuration set, gather the results and
        return them.

        Returns:
            results (np.ndarray) -- An array of results, gathered on the root
                                    node.
        """

        for cfg in self._config[mpi.rank::mpi.size]:
            dataslice = run_experiment(cfg, self._config.system)
            config.store_time_slice(cfg.id, dataslice)

        results = mpi.sum_data(config.results)

        return results

    def run_single(self, cfg_snap: ConfigSnapshot):
        pass


def run_configuration(infile: MuSpinInput, variables: dict = {}):
    """Run a whole set of calculations as defined by a MuSpinInput object

    Run a set of calculations (for multiple files and averages) as defined by
    a MuSpinInput object and a set of variable values. Takes care of 
    parallelism, splitting calculations across nodes etc.

    Arguments:
        infile {MuSpinInput} -- The input file object defining the 
                                calculations we need to perform.
        variables {dict} -- The values of any variables appearing in the input
                            file

    Returns:
        results (np.ndarray) -- An array of results, gathered on the root node.
    """

    if mpi.is_root:
        # On root, we run the evaluation that gives us the actual possible
        # values for simulation configurations. These are then broadcast
        # across all nodes, each of which runs its own slice of them, and
        # finally gathered back together
        config = MuSpinConfig(infile.evaluate(**variables))
    else:
        config = MuSpinConfig()

    mpi.broadcast_object(config)

    for cfg in config[mpi.rank::mpi.size]:
        dataslice = run_experiment(cfg, config.system)
        config.store_time_slice(cfg.id, dataslice)

    results = mpi.sum_data(config.results)

    return results


def run_experiment(cfg_snap: ConfigSnapshot, system: MuonSpinSystem):
    """Run a muon experiment from a configuration snapshot

    Run a muon experiment using the specific parameters and time axis given in
    a configuration snapshot.

    Arguments: 
        cfg_snap {ConfigSnapshot} -- A named tuple defining the values of all
                                     parameters to be used in this calculation.
        system {MuonSpinSystem} -- The spin system on which to perform the
                                   calculation

    Returns:
        result {np.ndarray} -- A 1D array containing the time series of 
                               required results, or a single value.
    """

    # Let's gather the important stuff
    B = cfg_snap.B          # Magnetic field
    p = cfg_snap.mupol      # Muon polarization
    q, w = cfg_snap.orient  # Quaternion and Weight for orientation
    T = cfg_snap.T          # Temperature

    # Let's start by rotating things
    B = q.rotate(B)
    p = q.rotate(p)

    # Measurement operator?
    S = system.muon_operator(p)

    # Base Hamiltonian
    H = system.hamiltonian

    return np.array(cfg_snap.t)*0+mpi.rank
