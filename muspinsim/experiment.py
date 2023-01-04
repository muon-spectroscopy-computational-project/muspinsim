"""experiment.py

Classes and functions to perform actual experiments"""

import logging
import numpy as np
import scipy.constants as cnst

from muspinsim.celio import CelioHamiltonian
from muspinsim.constants import MU_TAU
from muspinsim.utils import get_xy
from muspinsim.mpi import mpi_controller as mpi
from muspinsim.simconfig import MuSpinConfig, ConfigSnapshot
from muspinsim.spinsys import MuonSpinSystem, SingleTerm
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

        # broadcast config object without _system attribute
        attrs = list(config.__dict__.keys())
        for x in ["_system", "system"]:
            if x in attrs:
                attrs.remove(x)
        mpi.broadcast_object(config, attrs)

        # broadcast _system attribute without _terms attribute
        system = config.__dict__.get("_system", None)

        # Create default system only if none found
        if system is None:
            system = MuonSpinSystem()

        attrs = list(system.__dict__.keys())
        if "_terms" in attrs:
            attrs.remove("_terms")
        mpi.broadcast_object(system, attrs)

        # broadcast _terms attribute sequentially
        terms = system.__dict__.get("_terms", [])
        terms = mpi.broadcast_terms(terms)

        for i in terms:
            i.__setattr__("_spinsys", system)
        system.__setattr__("_terms", terms)
        config.__setattr__("_system", system)

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
                    evals, evecs = np.linalg.eigh(Hz.toarray())
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
            # Compute Zeeman Hamiltonian contribution
            if not self._config.celio_k:
                B = self._B
                g = self._system.gammas
                Bg = B[None, :] * g[:, None]

                Hz_sp_list = (self._single_spinops * Bg).flatten().tolist()
                Hz = np.sum(Hz_sp_list)

                self._Hz = Hamiltonian(Hz, dim=self._system.dimension)
            else:
                extra_terms = []
                # Add zeeman terms only if there is a field present to avoid
                # making Celio's method unnecessarily expensive
                if not np.array_equal(self._B, [0, 0, 0]):
                    for i in range(len(self._system.spins)):
                        extra_terms.append(
                            SingleTerm(
                                self._system,
                                i,
                                self._B * self._system.gammas[i],
                                label="Zeeman",
                            )
                        )
                self._Hz = CelioHamiltonian(
                    extra_terms, self.config.celio_k, self._system
                )

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

            def sparse_sum(sp_mat_list):
                sp_mat_list = sp_mat_list.flatten().tolist()
                return np.sum(sp_mat_list)

            self._dops = []
            for i, a in self._config.dissipation_terms.items():

                op_x = sparse_sum(self._single_spinops[i, :, None] * x[:, None])
                op_y = sparse_sum(self._single_spinops[i, :, None] * y[:, None])
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

        # Magnetic field is sum of external field (rotated with angular
        # averages) and the intrinsic field that should not be rotated
        self.B += cfg_snap.intrinsic_B

        # Figure out if a speedup is suitable
        B = np.linalg.norm(self.B)
        check_result = (cnst.e * (cnst.hbar**2) * B) / (2 * cnst.m_p * cnst.k * T)

        # For now only use when exactly 0 (i.e. when T -> inf, or B = 0)
        self._T_inf_speedup = check_result == 0

        # Due to the method of computation we also require that muon is first in
        # the system so we check this is the case here, otherwise we need to
        # change the order of kronecker products when computing the sigma_mu for
        # the system and this would be slower anyway
        if self._T_inf_speedup and self._system.muon_index != 0:
            self._T_inf_speedup = False

            # Add a message to the log to notify there is a speedup available
            # if the system is reordered
            logging.info(
                "The system is suitable for a speedup if the muon is defined first."
            )
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
            # Use faster more approximate method of Celio's if requested
            if isinstance(H, CelioHamiltonian) and self.config.celio_averages:
                # Ensure the system is valid for its use
                if self.T != np.inf:
                    raise ValueError(
                        "The fast version of Celio's method requires T -> inf. "
                        "Either remove the number of averages or don't use "
                        "celio."
                    )

                data = H.fast_evolve(
                    self._system.sigma_mu(self.p),
                    cfg_snap.t,
                    self.config.celio_averages,
                    True,
                )
            else:
                # Use faster evolution if able to (Doesn't exist for Linbladian)
                if self._T_inf_speedup and not isinstance(H, Lindbladian):
                    other_spins = list(range(0, len(self._system.spins)))
                    other_spins.remove(self._system.muon_index)
                    other_dimension = np.prod(
                        [self._system.dimension[i] for i in other_spins]
                    )

                    data = H.fast_evolve(
                        self._system.sigma_mu(self.p), cfg_snap.t, other_dimension
                    )
                else:
                    data = H.evolve(self.rho0, cfg_snap.t, operators=[S])[:, 0]
        elif cfg_snap.y == "integral":
            data = H.integrate_decaying(self.rho0, MU_TAU, operators=[S])[0] / MU_TAU

        return np.real(data) * w
