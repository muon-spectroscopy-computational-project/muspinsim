import unittest
import numpy as np
from scipy import sparse

from muspinsim import constants
from muspinsim.spinop import SpinOperator, SuperOperator
from muspinsim.spinsys import (
    SpinSystem,
    InteractionTerm,
    SingleTerm,
    DoubleTerm,
    MuonSpinSystem,
)


class TestSpinSystem(unittest.TestCase):
    def test_create(self):

        ssys = SpinSystem(["mu", ("H", 2)])

        # Test cloning
        ssys2 = ssys.clone()

        self.assertEqual(ssys.dimension, ssys2.dimension)

        # Check that the copy is deep
        ssys2._spins[0] = "C"
        self.assertEqual(ssys.spins, ["mu", ("H", 2)])

    def test_terms(self):

        ssys = SpinSystem(["mu", "e"])

        # A scalar term
        term = InteractionTerm(ssys, [], 2.0)
        self.assertTrue(
            np.allclose(term.operator.matrix.data, (ssys.operator({}) * 2).matrix.data)
        )

        # Linear term
        term = SingleTerm(ssys, 0, [0, 0, 1])
        self.assertTrue(
            np.allclose(term.operator.matrix.data, ssys.operator({0: "z"}).matrix.data)
        )

        # Test copy
        copy = term.clone()
        copy._tensor[0] = 1

        self.assertTrue(isinstance(copy, SingleTerm))
        self.assertFalse(np.all(copy.tensor == term.tensor))

        # Bilinear term
        term = DoubleTerm(ssys, 0, 1, np.diag([1, 1, 2]))
        self.assertTrue(
            np.allclose(
                term.operator.matrix.data,
                (
                    ssys.operator({0: "x", 1: "x"})
                    + ssys.operator({0: "y", 1: "y"})
                    + 2 * ssys.operator({0: "z", 1: "z"})
                ).matrix.data,
            )
        )

        # Rotation matrix
        R = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
        rotterm = term.rotate(R)
        self.assertTrue(
            np.allclose(
                rotterm.operator.matrix.data,
                (
                    ssys.operator({0: "x", 1: "x"})
                    + 2 * ssys.operator({0: "y", 1: "y"})
                    + ssys.operator({0: "z", 1: "z"})
                ).matrix.data,
            )
        )

    def test_check(self):

        ssys = SpinSystem(["mu", "e"])

        self.assertEqual(ssys.gamma(0), constants.MU_GAMMA)
        self.assertEqual(ssys.gamma(1), constants.ELEC_GAMMA)

        self.assertEqual(len(ssys), 2)
        self.assertEqual(ssys.dimension, (2, 2))

    def test_operator(self):

        ssys = SpinSystem(["mu", "e"])

        self.assertTrue(
            np.allclose(
                ssys.operator({0: "x"}).matrix.toarray(),
                SpinOperator.from_axes([0.5, 0.5], "x0").matrix.toarray(),
            )
        )

        self.assertTrue(
            np.allclose(
                ssys.operator({0: "z", 1: "y"}).matrix.toarray(),
                SpinOperator.from_axes([0.5, 0.5], "zy").matrix.toarray(),
            )
        )

        self.assertEqual(ssys.dimension, (2, 2))

    def test_operator_include_only_given(self):

        ssys = SpinSystem(["mu", "e"])

        print(ssys.operator({0: "x"}, True).matrix.toarray())
        print(SpinOperator.from_axes([0.5, 0.5], "x0").matrix.toarray())

        self.assertTrue(
            np.allclose(
                sparse.kron(
                    ssys.operator({0: "x"}, True).matrix, sparse.identity(2)
                ).toarray(),
                SpinOperator.from_axes([0.5, 0.5], "x0").matrix.toarray(),
            )
        )

        self.assertTrue(
            np.allclose(
                ssys.operator({0: "z", 1: "y"}, True).matrix.toarray(),
                SpinOperator.from_axes([0.5, 0.5], "zy").matrix.toarray(),
            )
        )

        self.assertEqual(ssys.dimension, (2, 2))

    def test_addterms(self):

        ssys = SpinSystem(["mu", "e"])

        t1 = ssys.add_linear_term(0, [1, 0, 1])

        self.assertEqual(t1.label, "Single")

        self.assertTrue(
            np.allclose(
                t1.operator.matrix.data,
                (ssys.operator({0: "x"}) + ssys.operator({0: "z"})).matrix.data,
            )
        )

        t2 = ssys.add_bilinear_term(0, 1, np.eye(3))

        self.assertEqual(t2.label, "Double")
        self.assertTrue(
            np.allclose(
                t2.operator.matrix.data,
                (
                    ssys.operator({0: "x", 1: "x"})
                    + ssys.operator({0: "y", 1: "y"})
                    + ssys.operator({0: "z", 1: "z"})
                ).matrix.data,
            )
        )

        H = ssys.hamiltonian

        self.assertTrue(
            np.all(
                np.isclose(
                    H.matrix.toarray(),
                    np.array(
                        [
                            [0.75, 0, 0.5, 0.0],
                            [0, 0.25, 0.5, 0.5],
                            [0.5, 0.5, -0.75, 0],
                            [0.0, 0.5, 0, -0.25],
                        ]
                    ),
                )
            )
        )

        ssys.add_dissipative_term(ssys.operator({0: "x"}), 1.0)
        self.assertTrue(ssys.is_dissipative)

        # Now test clearing them
        ssys.clear_terms()

        H = ssys.hamiltonian

        self.assertTrue(np.all(np.isclose(H.matrix.toarray(), np.zeros((4, 4)))))

    def test_addterms_celio(self):

        ssys = SpinSystem(["mu", "e"], celio=10)

        t1 = ssys.add_linear_term(0, [1, 0, 1])

        self.assertEqual(t1.label, "Single")

        self.assertTrue(
            np.allclose(
                t1.operator.matrix.data,
                (
                    ssys.operator({0: "x"}, True) + ssys.operator({0: "z"}, True)
                ).matrix.data,
            )
        )

        t2 = ssys.add_bilinear_term(0, 1, np.eye(3))

        self.assertEqual(t2.label, "Double")
        self.assertTrue(
            np.allclose(
                t2.operator.matrix.data,
                (
                    ssys.operator({0: "x", 1: "x"}, True)
                    + ssys.operator({0: "y", 1: "y"}, True)
                    + ssys.operator({0: "z", 1: "z"}, True)
                ).matrix.data,
            )
        )

        evol_op = ssys.hamiltonian._calc_trotter_evol_op(1)

        self.assertTrue(
            np.all(
                np.isclose(
                    evol_op.toarray(),
                    np.array(
                        [
                            [
                                0.57470166 + 0.35450902j,
                                0.66911902 - 0.01630097j,
                                0.22622372 - 0.16020638j,
                                0.10688036 - 0.08825368j,
                            ],
                            [
                                0.66911902 - 0.16020638j,
                                -0.33601494 + 0.53101637j,
                                -0.32064108 + 0.05565173j,
                                0.012463 + 0.16020638j,
                            ],
                            [
                                0.22622372 - 0.01630097j,
                                -0.32064108 - 0.23215908j,
                                0.54977566 + 0.53101637j,
                                0.4553583 + 0.01630097j,
                            ],
                            [
                                0.10688036 - 0.08825368j,
                                0.012463 + 0.01630097j,
                                0.4553583 + 0.16020638j,
                                -0.78846238 + 0.35450902j,
                            ],
                        ]
                    ),
                )
            )
        )

        # Now test clearing them
        ssys.clear_terms()

        evol_op = ssys.hamiltonian._calc_trotter_evol_op(1)

        self.assertTrue(np.isclose(evol_op, 1))

    def test_lindbladian(self):

        ssys = SpinSystem(["mu"])
        ssys.add_linear_term(0, [0, 0, 1])

        L = ssys.lindbladian
        L0 = -1.0j * SuperOperator.commutator(ssys.operator({0: "z"}))

        self.assertTrue(np.all(np.isclose(L.matrix.toarray(), L0.matrix.toarray())))

        sx = ssys.operator({0: "x"})
        d = 2.0
        ssys.add_dissipative_term(sx, d)

        L1 = L0 + d * (
            SuperOperator.bracket(sx) - 0.5 * SuperOperator.anticommutator(sx * sx)
        )

        L = ssys.lindbladian
        self.assertTrue(np.all(np.isclose(L.matrix.toarray(), L1.matrix.toarray())))

    def test_lindbladian_celio(self):

        ssys = SpinSystem(["mu"], 10)
        ssys.add_linear_term(0, [0, 0, 1])

        with self.assertRaises(NotImplementedError):
            L = ssys.lindbladian


class TestMuonSpinSystem(unittest.TestCase):
    def test_terms(self):

        mSsys = MuonSpinSystem(["e", "mu"])

        self.assertEqual(mSsys.dimension, (2, 2))
        self.assertEqual(mSsys.elec_indices, {0})
        self.assertEqual(mSsys.muon_index, 1)

        gmu = constants.MU_GAMMA
        ge = constants.ELEC_GAMMA

        z0 = mSsys.add_zeeman_term(0, 1.0)
        z1 = mSsys.add_zeeman_term(1, 1.0)
        h01 = mSsys.add_hyperfine_term(1, np.eye(3) * 100)

        H = mSsys.hamiltonian.matrix.toarray()

        self.assertTrue(
            np.all(
                np.isclose(
                    np.diag(H),
                    [
                        0.5 * (ge + gmu) + 25,
                        0.5 * (ge - gmu) - 25,
                        0.5 * (-ge + gmu) - 25,
                        0.5 * (-ge - gmu) + 25,
                    ],
                )
            )
        )

        mSsys.remove_term(h01)

        H = mSsys.hamiltonian.matrix.toarray()

        self.assertTrue(
            np.all(
                np.isclose(
                    np.diag(H),
                    [
                        0.5 * (ge + gmu),
                        0.5 * (ge - gmu),
                        0.5 * (-ge + gmu),
                        0.5 * (-ge - gmu),
                    ],
                )
            )
        )

        self.assertEqual(z0.label, "Zeeman")
        self.assertEqual(h01.label, "Hyperfine")

        mSsys = MuonSpinSystem(["e", "e", "mu"])

        self.assertEqual(mSsys.elec_indices, {0, 1})
        with self.assertRaises(ValueError):
            mSsys.add_hyperfine_term(2, np.eye(3))  # Must specify electron
