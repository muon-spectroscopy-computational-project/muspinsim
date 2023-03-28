import unittest

import numpy as np
from muspinsim.spinop import DensityOperator, SpinOperator
from muspinsim.spinsys import InteractionTerm, MuonSpinSystem

from muspinsim.validation import (
    validate_celio_params,
    validate_evolve_params,
    validate_integrate_decaying_params,
    validate_times,
)


class TestValidation(unittest.TestCase):
    def test_validate_times(self):
        # Should work
        validate_times(np.array([0, 1]))

        # Invalid
        with self.assertRaises(ValueError):
            validate_times(np.eye(2))
        with self.assertRaises(TypeError):
            validate_times(
                [10],
            )

    def test_validate_evolve_params(self):
        # Should work
        validate_evolve_params(
            DensityOperator(np.eye(2)), np.array([0, 1]), [SpinOperator.from_axes()]
        )

        # Invalid rho0
        with self.assertRaises(TypeError):
            validate_evolve_params(10, np.array([0, 1]), [SpinOperator.from_axes()])

        # Invalid times
        with self.assertRaises(ValueError):
            validate_evolve_params(
                DensityOperator(np.eye(2)), np.eye(2), [SpinOperator.from_axes()]
            )

        # Invalid SpinOperators
        with self.assertRaises(ValueError):
            validate_evolve_params(
                DensityOperator(np.eye(2)),
                np.array([0, 1]),
                [SpinOperator.from_axes(), 2],
            )

    def test_validate_celio(self):
        # Should work
        spin_system = MuonSpinSystem(["e", "mu"])
        interaction_terms = [InteractionTerm(spin_system)]
        validate_celio_params(interaction_terms, np.array([0, 1, 2]))

        # No terms
        with self.assertRaises(ValueError):
            validate_celio_params([], np.array([0, 1, 2]))

        # Invalid initial time
        with self.assertRaises(ValueError):
            validate_celio_params(interaction_terms, np.array([1, 2, 3]))

        # Uneven spacing between times
        with self.assertRaises(ValueError):
            validate_celio_params(interaction_terms, np.array([0, 1, 2.5, 3]))

    def test_validate_integrate_decaying_params(self):
        # Should work
        validate_integrate_decaying_params(
            DensityOperator(np.eye(2)), 10, [SpinOperator.from_axes()]
        )

        # Invalid rho0
        with self.assertRaises(TypeError):
            validate_integrate_decaying_params(10, 10, [SpinOperator.from_axes()])

        # Invalid tau
        with self.assertRaises(ValueError):
            validate_integrate_decaying_params(
                DensityOperator(np.eye(2)), "test", [SpinOperator.from_axes()]
            )

        # Invalid SpinOperators
        with self.assertRaises(ValueError):
            validate_integrate_decaying_params(
                DensityOperator(np.eye(2)),
                10,
                [SpinOperator.from_axes(), 2],
            )

        # No SpinOperators
        with self.assertRaises(TypeError):
            validate_integrate_decaying_params(10, 10, [])
