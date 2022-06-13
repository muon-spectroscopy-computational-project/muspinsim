"""experiment.py

Classes and functions to perform actual experiments"""

import logging
import numpy as np
import scipy.constants as cnst

from muspinsim.constants import MU_TAU
from muspinsim.utils import get_xy
from muspinsim.mpi import mpi_controller as mpi
from muspinsim.simconfig import MuSpinConfig, ConfigSnapshot
from muspinsim.input import MuSpinInput
from muspinsim.spinop import DensityOperator, SpinOperator
from muspinsim.hamiltonian import Hamiltonian
from muspinsim.lindbladian import Lindbladian


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
        # Store single spin operators
        self._single_spinops = np.array(
            [
                [self._system.operator({i: a}).matrix for a in "xyz"]
                for i in range(len(self._system))
            ]
        )

        # Parameters
        self._B = np.zeros(3)
        self._p = np.array([1.0, 0, 0])
        self._T = np.inf

        # Basic Hamiltonian
        self._Hsys = self._system.hamiltonian

        # Derived quantities
        self._rho0 = None
        self._Hz = None
        self._dops = None

    @property
    def config(self):
        return self._config

    @property
    def system(self):
        return self._system

    @property
    def B(self):
        return self._B

    @B.setter
    def B(self, x):
        x = np.array(x)
        if (x != self._B).any():
            self._B = x
            self._rho0 = None
            self._Hz = None
            self._dops = None

    @property
    def p(self):
        return self._p

    @p.setter
    def p(self, x):
        x = np.array(x)
        if (x != self._p).any():
            self._p = x
            self._rho0 = None

    @property
    def T(self):
        return self._T

    @T.setter
    def T(self, x):
        if x != self._T:
            self._T = x
            self._rho0 = None
            self._dops = None

    @property
    def rho0(self):
        """Calculate a thermal density matrix in which the system is prepared

        Calculate an approximate thermal density matrix to prepare the system in,
        with the muon polarised along a given direction and every other spin in a
        thermal equilibrium decohered state.

        Returns:
            rho0 {DensityOperator} -- Density matrix at t=0
        """

        if self._rho0 is None:
            T = self._T
            muon_axis = self._p
            B = self._B

            if np.isclose(np.linalg.norm(B), 0.0) and T < np.inf:
                logging.warning(
                    "WARNING: initial density matrix is computed"
                    " with an approximation that can fail"
                    " at low fields and finite temperature"
                )

            mu_i = self._system.muon_index
            rhos = []

            for i, s in enumerate(self._system.spins):
                I = self._system.I(i)
                if i == mu_i:
                    r = DensityOperator.from_vectors(I, muon_axis, 0)
                else:
                    # Get the Zeeman Hamiltonian for this field
                    Hz = np.sum(
                        [
                            B[j] * SpinOperator.from_axes(I, e).matrix
                            for j, e in enumerate("xyz")
                        ],
                        axis=0,
                    )

                    evals, evecs = np.linalg.eigh(Hz)
                    E = evals * 1e6 * self._system.gamma(i)

                    if T > 0:
                        Z = np.exp(-cnst.h * E / (cnst.k * T))
                    else:
                        Z = np.where(E == np.amin(E), 1.0, 0.0)
                    if np.sum(Z) > 0:
                        Z /= np.sum(Z)
                    else:
                        Z = np.ones(len(E)) / len(E)

                    rhoI = np.sum(
                        evecs[:, None, :] * evecs[None, :, :].conj() * Z[None, None, :],
                        axis=-1,
                    )

                    r = DensityOperator(rhoI)

                rhos.append(r)

            self._rho0 = rhos[0]
            for r in rhos[1:]:
                self._rho0 = self._rho0.kron(r)

        return self._rho0

    @property
    def Hz(self):

        if self._Hz is None:
            B = self._B
            g = self._system.gammas
            Hz = np.sum(
                B[None, :, None, None] * g[:, None, None, None] * self._single_spinops,
                axis=(0, 1),
            )
            self._Hz = Hamiltonian(Hz, dim=self._system.dimension)

        return self._Hz

    @property
    def dissipation_operators(self):
        if self._dops is None:

            # Create a copy of the system
            sys = self._system.clone()

            # Clean it up of all terms
            sys.clear_terms()
            sys.clear_dissipative_terms()

            T = self._T
            # We only go by the intensity of the field
            B = np.linalg.norm(self._B)
            g = sys.gammas
            if T > 0:
                Zu = np.exp(-cnst.h * g * B * 1e6 / (cnst.k * T))
                if np.isclose(B, 0.0) and T < np.inf:
                    logging.warning(
                        "WARNING: dissipation effects are computed"
                        " with an approximation that can fail"
                        " at low fields and finite temperature"
                    )
            else:
                Zu = g * 0.0

            if B == 0:
                x, y = np.array([1.0, 0, 0]), np.array([0, 1.0, 0])
            else:
                z = self._B / B
                x, y = get_xy(z)

            self._dops = []
            for i, a in self._config.dissipation_terms.items():

                op_x = np.sum(self._single_spinops[i, :] * x[:, None, None], axis=0)
                op_y = np.sum(self._single_spinops[i, :] * y[:, None, None], axis=0)
                op_p = SpinOperator(op_x + 1.0j * op_y, dim=self.system.dimension)
                op_m = SpinOperator(op_x - 1.0j * op_y, dim=self.system.dimension)

                # The 1/pi factor here makes sure that the convention is that when
                # a user inputs a certain value of dissipation for a single spin,
                # that is exactly the exponential decay factor that is observed.

                self._dops.append((op_p, 1 / np.pi * a * Zu[i] / (1 + Zu[i])))
                self._dops.append((op_m, 1 / np.pi * a / (1 + Zu[i])))

        return self._dops

    @property
    def Hsys(self):
        return self._Hsys

    @property
    def Htot(self):
        # Build the total Hamiltonian
        H = self.Hsys + self.Hz

        # Do we have dissipation?
        if len(self._config.dissipation_terms) > 0:
            # Actually use a Lindbladian
            H = Lindbladian.from_hamiltonian(H, self.dissipation_operators)

        return H

    @property
    def p_operator(self):
        return self._system.muon_operator(self.p)

    def run(self):
        """Run the experiment

        Run all calculations in the configuration set, gather the results and
        return them.

        Returns:
            results (np.ndarray) -- An array of results, gathered on the root
                                    node.
        """

        for cfg in self._config[mpi.rank :: mpi.size]:
            dataslice = self.run_single(cfg)
            self._config.store_time_slice(cfg.id, dataslice)

        self._config.results = mpi.sum_data(self._config.results)

        return self._config.results

    def load_config(self, cfg_snap: ConfigSnapshot):
        """Load a configuration snapshot in this ExperimentRunner

        Load a configuration snapshot in the ExperimentRunner, assigning field,
        polarisation and temperature.

        Arguments:
            cfg_snap {ConfigSnapshot} -- A named tuple defining the values of all
                                         parameters to be used in this calculation.

        Returns:
            weight {float} -- The weight to assign to this specific simulation
        """

        # Let's gather the important stuff
        B = cfg_snap.B  # Magnetic field
        p = cfg_snap.mupol  # Muon polarization
        T = cfg_snap.T  # Temperature
        q, w = cfg_snap.orient  # Quaternion and Weight for orientation

        # Let's start by rotating things
        self.B = q.rotate(B)
        self.p = q.rotate(p)
        self.T = T

        return w

    def run_single(self, cfg_snap: ConfigSnapshot):
        """Run a muon experiment from a configuration snapshot

        Run a muon experiment using the specific parameters and time axis given in
        a configuration snapshot.

        Arguments:
            cfg_snap {ConfigSnapshot} -- A named tuple defining the values of all
                                         parameters to be used in this calculation.

        Returns:
            result {np.ndarray} -- A 1D array containing the time series of
                                   required results, or a single value.
        """

        w = self.load_config(cfg_snap)

        # Measurement operator?
        S = self.p_operator

        H = self.Htot

        if cfg_snap.y == "asymmetry":
            data = H.evolve(self.rho0, cfg_snap.t, operators=[S])[:, 0]
        elif cfg_snap.y == "integral":
            data = H.integrate_decaying(self.rho0, MU_TAU, operators=[S])[0] / MU_TAU

        return np.real(data) * w
