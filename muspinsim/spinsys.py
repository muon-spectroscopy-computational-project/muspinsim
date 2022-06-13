"""spinsys.py

A class to hold a given spin system, defined by specific nuclei
"""

import logging

import numpy as np
from numbers import Number
import scipy.constants as cnst

from muspinsim.utils import Clonable
from muspinsim.spinop import SpinOperator
from muspinsim.hamiltonian import Hamiltonian
from muspinsim.lindbladian import Lindbladian
from muspinsim.constants import gyromagnetic_ratio, spin, quadrupole_moment, EFG_2_MHZ


class InteractionTerm(Clonable):
    def __init__(self, spinsys, indices=[], tensor=0, label=None):

        self._spinsys = spinsys
        self._indices = np.array(indices)
        self._tensor = np.array(tensor)
        self._label = "Term" if label is None else label

        if np.any(np.array(self._tensor.shape) != 3):
            raise ValueError("Tensor is not fully three-dimensional")

        self._recalc_operator()

    def _recalc_operator(self):

        total_op = None
        d = len(self._tensor.shape)

        if d > 0:
            index_tuples = np.indices(self._tensor.shape).reshape((d, -1)).T
        else:
            index_tuples = [[]]

        for ii in index_tuples:
            op = (
                self._spinsys.operator(
                    {ind: "xyz"[ii[i]] for i, ind in enumerate(self._indices)}
                )
                * self._tensor[tuple(ii)]
            )
            if total_op is None:
                total_op = op
            else:
                total_op += op

        self._operator = total_op

    @property
    def label(self):
        return self._label

    @property
    def indices(self):
        return tuple(self._indices)

    @property
    def tensor(self):
        return np.array(self._tensor)

    @property
    def operator(self):
        return self._operator.clone()

    @property
    def matrix(self):
        return self._operator.matrix

    def __repr__(self):
        return self.label


class SingleTerm(InteractionTerm):
    def __init__(self, spinsys, i, vector, label="Single"):

        super(SingleTerm, self).__init__(spinsys, [i], vector, label)

    @property
    def i(self):
        return self._indices[0]

    def rotate(self, rotmat):

        R = np.array(rotmat)
        v = self._tensor
        v = np.dot(v, R.T)

        rt = SingleTerm(self._spinsys, self.i, v, self._label)

        return rt

    def __repr__(self):
        return "{0} {{ S_{1} * {2} }}".format(self._label, self.i, self._tensor)


class DoubleTerm(InteractionTerm):
    def __init__(self, spinsys, i, j, matrix, label="Double"):

        super(DoubleTerm, self).__init__(spinsys, [i, j], matrix, label)

    @property
    def i(self):
        return self._indices[0]

    @property
    def j(self):
        return self._indices[1]

    def rotate(self, rotmat):

        R = np.array(rotmat)
        M = self._tensor
        M = np.linalg.multi_dot([R, M, R.T])

        rt = DoubleTerm(self._spinsys, self.i, self.j, M, self._label)

        return rt

    def __repr__(self):
        return "{0} {{ S_{1} * [{2} {3} {4}] * S_{5} }}".format(
            self._label, self.i, *self._tensor, self.j
        )


class DissipationTerm(Clonable):
    def __init__(self, operator, gamma=0.0):

        self._op = operator
        self._g = gamma

    @property
    def operator(self):
        return self._op

    @property
    def gamma(self):
        return self._g

    @property
    def tuple(self):
        return (self._op, self._g)


class SpinSystem(Clonable):
    def __init__(self, spins=[]):
        """Create a SpinSystem object

        Create an object representing a system of particles with spins (muons,
        electrons and atomic nuclei) and holding their operators.

        Keyword Arguments:
            spins {list} -- List of symbols representing the various particles.
                            Each element can be 'e' (electron), 'mu' (muon) a
                            chemical symbol, or a (str, int) tuple with a
                            chemical symbol and an isotope (default: {[]})
        """

        gammas = []
        Qs = []
        Is = []
        operators = []

        for s in spins:
            if isinstance(s, tuple):
                el, iso = s
            else:
                el, iso = s, None

            gammas.append(gyromagnetic_ratio(el, iso))
            Qs.append(quadrupole_moment(el, iso))
            Is.append(spin(el, iso))

            opdict = {a: SpinOperator.from_axes(Is[-1], a) for a in "xyz+-0"}

            operators.append(opdict)

        self._spins = list(spins)
        self._gammas = np.array(gammas)
        self._Qs = np.array(Qs)
        self._Is = np.array(Is)
        self._dim = tuple((2 * self._Is + 1).astype(int))

        self._operators = operators

        self._terms = []
        self._dissip_terms = []

        snames = [
            "{1}{0}".format(*s) if (type(s) == tuple) else str(s) for s in self._spins
        ]
        logging.info("Created spin system with spins:")
        logging.info("\t\t{0}".format(" ".join(snames)))

    @property
    def spins(self):
        return list(self._spins)

    @property
    def gammas(self):
        return self._gammas.copy()

    @property
    def Qs(self):
        return self._Qs.copy()

    @property
    def Is(self):
        return self._Is.copy()

    @property
    def dimension(self):
        return self._dim

    @property
    def is_dissipative(self):
        return (np.array(self._dissip_terms) != 0.0).any()

    def add_term(self, indices, tensor, label="Term"):
        """Add to the spin system a generic interaction term

        Add a term of the form T*S_i*S_j*S_k*..., where S_i is the vector of
        the three spin operators:

        [S_x, S_y, S_z]

        for spin of index i.

        Arguments:
            indices {[int]} -- Indices of spins appearing in the term
            tensor {ndarray} -- Tensor with n dimensions (n = len(indices)),
                                each of length 3, describing the interaction.

        Keyword Arguments:
            label {str} -- A label to name the term (default: {'Term'})

        Returns:
            term {InteractionTerm} -- The term just created

        Raises:
            ValueError -- Invalid index or vector
        """

        for i in indices:
            if i < 0 or i >= len(self._spins):
                raise ValueError("Invalid index i")

        tensor = np.array(tensor)

        term = InteractionTerm(self, indices, tensor, label=label)
        self._terms.append(term)

        return term

    def add_linear_term(self, i, vector, label="Single"):
        """Add to the spin system a term linear in one spin

        Add a term of the form v*S_i, where S_i is the vector of the three
        spin operators:

        [S_x, S_y, S_z]

        for spin of index i.

        Arguments:
            i {int} -- Index of the spin
            vector {ndarray} -- Vector v

        Keyword Arguments:
            label {str} -- A label to name the term (default: {'Single'})

        Returns:
            SingleTerm -- The term just created

        Raises:
            ValueError -- Invalid index or vector
        """

        if i < 0 or i >= len(self._spins):
            raise ValueError("Invalid index i")

        vector = np.array(vector)

        term = SingleTerm(self, i, vector, label=label)
        self._terms.append(term)

        return term

    def add_bilinear_term(self, i, j, matrix, label="Double"):
        """Add to the spin system a term bilinear in two spins

        Add a term of the form S_i*M*S_j, where S_i is the vector of the three
        spin operators:

        [S_x, S_y, S_z]

        for spin of index i, and same for S_j.

        Arguments:
            i {int} -- Index of first spin
            j {int} -- Index of second spin
            matrix {ndarray} -- Matrix M

        Keyword Arguments:
            label {str} -- A label to name the term (default: {'Double'})

        Returns:
            DoubleTerm -- The term just created

        Raises:
            ValueError -- Invalid index or vector
        """

        if i < 0 or i >= len(self._spins):
            raise ValueError("Invalid index i")

        if j < 0 or j >= len(self._spins):
            raise ValueError("Invalid index j")

        matrix = np.array(matrix)

        term = DoubleTerm(self, i, j, matrix, label=label)
        self._terms.append(term)

        return term

    def add_zeeman_term(self, i, B):
        """Add a zeeman term

        Add a single term coupling a given spin to a magnetic field

        Arguments:
            i {int} -- Index of the spin
            B {ndarray | number} -- Magnetic field vector, in Tesla. If just a
                                    scalar is assumed to be along z

        Returns:
            SingleTerm -- The term just created
        """

        if isinstance(B, Number):
            B = [0, 0, B]  # Treat it as along z by default

        B = np.array(B)

        logging.info("Adding Zeeman term to spin {0}".format(i + 1))

        return self.add_linear_term(i, B * self.gamma(i), "Zeeman")

    def add_dipolar_term(self, i, j, r):
        """Add a dipolar term

        Add a spin-spin dipolar coupling between two distinct spins. The
        coupling is calculated geometrically from the vector connecting them,
        in Angstrom.

        Arguments:
            i {int} -- Index of the first spin
            j {int} -- Index of the second spin
            r {ndarray} -- Vector connecting the two spins (in Angstrom)

        Returns:
            DoubleTerm -- The term just created

        Raises:
            ValueError -- Raised if i == j
        """

        if i == j:
            raise ValueError("Can not set up dipolar coupling with itself")

        r = np.array(r)

        g_i = self.gamma(i)
        g_j = self.gamma(j)

        rnorm = np.linalg.norm(r)
        D = -(np.eye(3) - 3.0 / rnorm**2.0 * r[:, None] * r[None, :])
        dij = -(cnst.mu_0 * cnst.hbar * (g_i * g_j * 1e6)) / (
            2 * (rnorm * 1e-10) ** 3
        )  # MHz
        D *= dij

        logging.info("Adding dipolar term to spins {0}-{1}".format(i + 1, j + 1))

        return self.add_bilinear_term(i, j, D, "Dipolar")

    def add_quadrupolar_term(self, i, EFG):
        """Add a quadrupolar term

        Add a quadrupolar term to a nucleus with I >= 1 from its Electric
        Field Gradient tensor.

        Arguments:
            i {int} -- Index of the spin
            EFG {ndarray} --  Electric Field Gradient tensor

        Returns:
            DoubleTerm -- The term just created
        """

        EFG = np.array(EFG)
        Q = self.Q(i)
        I = self.I(i)

        if I == 0.5:
            raise ValueError(
                "Can not set up quadrupolar coupling for " "spin 1/2 particle"
            )

        Qtens = EFG_2_MHZ * Q / (2 * I * (2 * I - 1)) * EFG

        logging.info("Adding quadrupolar term to spin {0}".format(i + 1))

        return self.add_bilinear_term(i, i, Qtens, "Quadrupolar")

    def remove_term(self, term):
        """Remove a term from the spin system

        Remove an interaction term from this spin system.

        Arguments:
            term {InteractionTerm} -- Term to remove

        Raises:
            ValueError -- The term is not contained in this system
        """

        self._terms.remove(term)

    def clear_terms(self):
        """Remove all terms

        Remove all interaction terms from this spin system.
        """

        terms = list(self._terms)
        for t in terms:
            self.remove_term(t)

    def add_dissipative_term(self, op, d=0.0):
        """Set a dissipation operator for the system.

        Set a dissipation operator for this system, representing its coupling
        (in MHz) with an external heat bath to include in the Lindbladian of
        the system.

        Arguments:
            op {SpinOperator} -- Operator for the dissipation term

        Keyword Arguments:
            d {number} -- Dissipation coupling in MHz (default: {0.0})
        """

        term = DissipationTerm(op, d)
        self._dissip_terms.append(term)

        return term

    def remove_dissipative_term(self, term):
        """Remove a dissipation term from the system.

        Remove a dissipation term from this spin system.

        Arguments:
            term {DissipationTerm} -- Term to remove

        Raises:
            ValueError -- The term is not contained in this system
        """

        self._dissip_terms.remove(term)

    def clear_dissipative_terms(self):
        """Remove all terms

        Remove all dissipative terms from this spin system.
        """

        dterms = list(self._dissip_terms)
        for t in dterms:
            self.remove_dissipative_term(t)

    def gamma(self, i):
        """Returns the gyromagnetic ratio of a given particle

        Arguments:
            i {int} -- Index of the particle

        Returns:
            float -- Gyromagnetic ratio in MHz/T
        """
        return self._gammas[i]

    def Q(self, i):
        """Returns the quadrupole moment of a given particle

        Arguments:
            i {int} -- Index of the particle

        Returns:
            float -- Quadrupole moment in Barn
        """
        return self._Qs[i]

    def I(self, i):
        """Returns the spin of a given particle

        Arguments:
            i {int} -- Index of the particle

        Returns:
            float -- Spin in units of hbar
        """

        return self._Is[i]

    def operator(self, terms={}):
        """Return an operator for this spin system

        Return a SpinOperator for this system containing the specified terms.

        Keyword Arguments:
            terms {dict} -- A dictionary of terms to include. The keys should
                            indices of particles and the values should be
                            symbols indicating one spin operator (either x, y,
                            z, +, - or 0). Wherever not specified, the identity
                            operaror is applied (default: {{}})

        Returns:
            SpinOperator -- The requested operator
        """

        ops = [self._operators[i][terms.get(i, "0")] for i in range(len(self))]

        M = ops[0]

        for i in range(1, len(ops)):
            M = M.kron(ops[i])

        return M

    def rotate(self, rotmat=np.eye(3)):

        # Trying to avoid pointlessly cloning the terms
        terms = self._terms
        self._terms = []

        # Make a clone
        rssys = self.clone()
        self._terms = terms

        # Edit the terms
        try:
            rssys._terms = [t.rotate(rotmat) for t in terms]
        except AttributeError:
            raise RuntimeError(
                "Can only rotate SpinSystems containing Single" " or Double terms"
            )

        return rssys

    @property
    def hamiltonian(self):

        if len(self._terms) == 0:
            n = np.prod(self.dimension)
            H = np.zeros((n, n))
        else:
            H = np.sum([t.matrix for t in self._terms], axis=0)
        H = Hamiltonian(H, dim=self.dimension)

        return H

    @property
    def lindbladian(self):

        H = self.hamiltonian
        dops = [t.tuple for t in self._dissip_terms]
        L = Lindbladian.from_hamiltonian(H, dops)

        return L

    def __len__(self):
        return len(self._gammas)


class MuonSpinSystem(SpinSystem):
    def __init__(self, spins=["mu", "e"]):

        super(MuonSpinSystem, self).__init__(spins)

        # Identify the muon index
        if self._spins.count("mu") != 1:
            raise ValueError(
                "Spins passed to MuonSpinSystem must contain" " exactly one muon"
            )

        self._mu_i = self._spins.index("mu")
        self._e_i = set([i for i, s in enumerate(self.spins) if s == "e"])

        # For convenience, store the operators for the muon
        self._mu_ops = [self.operator({self._mu_i: e}) for e in "xyz"]

    @property
    def muon_index(self):
        return self._mu_i

    @property
    def elec_indices(self):
        return self._e_i

    def add_hyperfine_term(self, i, A, j=None):
        """Add a hyperfine term

        Add a hyperfine term for a given spin, provided that an electron is
        present.

        Arguments:
            i {int} -- Index of the spin (must be different from electron)
            A {[type]} -- Hyperfine tensor (in MHz)
            j {int} -- Index of the electron spin. If not specified uses the
                       one that is present, if there is one (default: None)

        Returns:
            DoubleTerm -- The term just created

        Raises:
            ValueError -- Invalid index
        """

        elec_i = self.elec_indices

        if j is None:
            if len(elec_i) > 1:
                raise ValueError(
                    "Must specify an electron index in system "
                    "with multiple electrons"
                )
            else:
                j = list(elec_i)[0]
        else:
            if j not in elec_i:
                raise ValueError(
                    "Second index in hyperfine coupling must" " refer to an electron"
                )
        if i in elec_i:
            raise ValueError(
                "First index in hyperfine coupling must" " not refer to an electron"
            )

        logging.info("Adding hyperfine term to spins {0}-{1}".format(i + 1, j + 1))

        return self.add_bilinear_term(i, j, A, "Hyperfine")

    def muon_operator(self, v):
        """Get a muon operator

        Get a single operator for the muon, given a vector representing its
        direction. Uses precalculated operators for speed.

        Arguments:
            v {[float]} -- 3-dimensional vector representing the directions of
                           the desired operator

        Returns:
            mu_op {SpinOperator} -- Requested operator

        Raises:
            ValueError -- Invalid length of v
        """

        if len(v) != 3:
            raise ValueError(
                "Vector passed to muon_operator must be three" " dimensional"
            )

        op = [x * self._mu_ops[i] for i, x in enumerate(v)]
        op = sum(op[1:], op[0])

        return op
