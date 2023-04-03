from numbers import Number

import numpy as np

from muspinsim.spinop import DensityOperator, SpinOperator


def validate_times(times):
    """Validates the 'times' parameter for 'evolve' and 'integrate_decaying'
       methods in the  Hamiltonian, CelioHamiltonian and Linbladian classes

    Arguments:
       times {ndarray} -- Times to compute the evolution for, in microseconds

    Raises:
       ValueError -- Invalid values of times
    """
    if not isinstance(times, np.ndarray):
        raise TypeError("times must be an array of values in microseconds")
    if len(times.shape) != 1:
        raise ValueError("times must be an array of values in microseconds")


def validate_evolve_params(rho0, times, operators):
    """Validates the required parameters for 'evolve' methods
       in the Hamiltonian, CelioHamiltonian and Linbladian classes

    Arguments:
        rho0 {DensityOperator} -- Initial state
        times {ndarray} -- Times to compute the evolution for, in microseconds
        operators {[SpinOperator]} -- List of SpinOperators to compute the
                                      expectation values of at each step.
    Raises:
        TypeError -- Invalid operators
        ValueError -- Invalid values of times or operators
    """
    if not isinstance(rho0, DensityOperator):
        raise TypeError("rho0 must be a valid DensityOperator")

    validate_times(times)

    if not all([isinstance(o, SpinOperator) for o in operators]):
        raise ValueError(
            "operators must be a SpinOperator or a list of SpinOperator objects"
        )


def validate_celio_params(terms, times):
    """Validates the required parameters for any method using Celio's

    Arguments:
        terms {[InteractionTerm]} -- Interaction terms that will form part of the
                                     Trotter expansion
        times {ndarray} -- Times to compute the evolution for, in microseconds

    Raises:
        ValueError -- If there are no terms to evolve with
        ValueError -- If the first time does not start at 0
        ValueError -- If the times have uneven spacing
    """

    if len(terms) == 0:
        raise ValueError("No interaction terms to evolve")

    if times[0] != 0:
        raise ValueError("Cannot use Celio's method with a non-zero start time")

    # Ensure the spacing is identical between all the values
    differences = np.diff(times)
    if not np.isclose(differences, differences[0]).all():
        raise ValueError("Cannot use Celio's method with uneven spacing between times")


def validate_integrate_decaying_params(rho0, tau, operators):
    """Validates the required parameters for 'integrate_decaying' methods
       in the Hamiltonian and Linbladian classes

    Arguments:
        rho0 {DensityOperator} -- Initial state
        tau {float} -- Decay time, in microseconds
        operators {list} -- Operators to compute the expectation values
                            of

    Raises:
        TypeError -- Invalid operators
        ValueError -- Invalid values of tau or operators
        ValueError -- If 'operators' is empty
    """
    if not isinstance(rho0, DensityOperator):
        raise TypeError("'rho0' must be a valid DensityOperator")

    if not (isinstance(tau, Number) and np.isreal(tau) and tau > 0):
        raise ValueError("'tau' must be a real number > 0")

    if not operators:
        raise ValueError("At least one SpinOperator must be present in 'operators'")

    if not all([isinstance(o, SpinOperator) for o in operators]):
        raise ValueError(
            "'operators' must be a SpinOperator or a list of SpinOperator objects"
        )
