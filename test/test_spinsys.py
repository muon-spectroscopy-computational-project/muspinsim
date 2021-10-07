import unittest
import numpy as np

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

        ssys = SpinSystem(["mu", "e"])

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
        self.assertEqual(term.operator, ssys.operator({}) * 2)

        # Linear term
        term = SingleTerm(ssys, 0, [0, 0, 1])
        self.assertEqual(term.operator, ssys.operator({0: "z"}))

        # Test copy
        copy = term.clone()
        copy._tensor[0] = 1

        self.assertTrue(isinstance(copy, SingleTerm))
        self.assertFalse(np.all(copy.tensor == term.tensor))

        # Bilinear term
        term = DoubleTerm(ssys, 0, 1, np.diag([1, 1, 2]))
        self.assertEqual(
            term.operator,
            ssys.operator({0: "x", 1: "x"})
            + ssys.operator({0: "y", 1: "y"})
            + 2 * ssys.operator({0: "z", 1: "z"}),
        )

        # Rotation matrix
        R = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
        rotterm = term.rotate(R)
        self.assertEqual(
            rotterm.operator,
            ssys.operator({0: "x", 1: "x"})
            + 2 * ssys.operator({0: "y", 1: "y"})
            + ssys.operator({0: "z", 1: "z"}),
        )

    def test_check(self):

        ssys = SpinSystem(["mu", "e"])

        self.assertEqual(ssys.gamma(0), constants.MU_GAMMA)
        self.assertEqual(ssys.gamma(1), constants.ELEC_GAMMA)

        self.assertEqual(len(ssys), 2)
        self.assertEqual(ssys.dimension, (2, 2))

    def test_operator(self):

        ssys = SpinSystem(["mu", "e"])

        self.assertEqual(
            ssys.operator({0: "x"}), SpinOperator.from_axes([0.5, 0.5], "x0")
        )
        self.assertEqual(
            ssys.operator({0: "z", 1: "y"}), SpinOperator.from_axes([0.5, 0.5], "zy")
        )
        self.assertEqual(ssys.dimension, (2, 2))

    def test_addterms(self):

        ssys = SpinSystem(["mu", "e"])

        t1 = ssys.add_linear_term(0, [1, 0, 1])

        self.assertEqual(t1.label, "Single")
        self.assertEqual(t1.operator, ssys.operator({0: "x"}) + ssys.operator({0: "z"}))

        t2 = ssys.add_bilinear_term(0, 1, np.eye(3))

        self.assertEqual(t2.label, "Double")
        self.assertEqual(
            t2.operator,
            ssys.operator({0: "x", 1: "x"})
            + ssys.operator({0: "y", 1: "y"})
            + ssys.operator({0: "z", 1: "z"}),
        )

        H = ssys.hamiltonian

        self.assertTrue(
            np.all(
                np.isclose(
                    H.matrix,
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

        self.assertTrue(np.all(np.isclose(H.matrix, np.zeros((4, 4)))))

    def test_lindbladian(self):

        ssys = SpinSystem(["mu"])
        ssys.add_linear_term(0, [0, 0, 1])

        H = ssys.hamiltonian
        L = ssys.lindbladian
        L0 = -1.0j * SuperOperator.commutator(ssys.operator({0: "z"}))

        self.assertTrue(np.all(np.isclose(L.matrix, L0.matrix)))

        sx = ssys.operator({0: "x"})
        d = 2.0
        ssys.add_dissipative_term(sx, d)

        L1 = L0 + d * (
            SuperOperator.bracket(sx) - 0.5 * SuperOperator.anticommutator(sx * sx)
        )

        L = ssys.lindbladian
        self.assertTrue(np.all(np.isclose(L.matrix, L1.matrix)))


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

        H = mSsys.hamiltonian.matrix

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

        H = mSsys.hamiltonian.matrix

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
