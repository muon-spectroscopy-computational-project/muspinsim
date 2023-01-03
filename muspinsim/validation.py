import numpy as np

from numbers import Number
from muspinsim.spinop import DensityOperator, SpinOperator


def validate_evolve_params(rho0, times, operators):
    """Validates the required parameters for 'evolve' methods
       in Hamiltonian, CelioHamiltonian and Linbladian classes

    Arguments:
        rho0 {DensityOperator} -- Initial state
        times {ndarray} -- Times to compute the evolution for, in microseconds

    Keyword Arguments:
        operators {[SpinOperator]} -- List of SpinOperators to compute the
                                        expectation values of at each step.
                                        If omitted, the states' density
                                        matrices will be returned instead
                                        (default: {[]})
    Raises:
        TypeError -- Invalid operators
        ValueError -- Invalid values of times or operators
    """
    if not isinstance(rho0, DensityOperator):
        raise TypeError("rho0 must be a valid DensityOperator")

    if len(times.shape) != 1:
        raise ValueError("times must be an array of values in microseconds")

    if not all([isinstance(o, SpinOperator) for o in operators]):
        raise ValueError(
            "operators must be a SpinOperator or a list of SpinOperator objects"
        )


def validate_integrate_decaying_params(rho0, tau, operators):
    """Validates the required parameters for 'integrate_decaying' methods
       in Hamiltonian and Linbladian classes

    Arguments:
        rho0 {DensityOperator} -- Initial state
        tau {float} -- Decay time, in microseconds

    Keyword Arguments:
        operators {list} -- Operators to compute the expectation values
                            of

    Raises:
        TypeError -- Invalid operators
        ValueError -- Invalid values of tau or operators
    """
    if not isinstance(rho0, DensityOperator):
        raise TypeError("rho0 must be a valid DensityOperator")

    if not (isinstance(tau, Number) and np.isreal(tau) and tau > 0):
        raise ValueError("tau must be a real number > 0")

    if not all([isinstance(o, SpinOperator) for o in operators]):
        raise ValueError(
            "operators must be a SpinOperator or a list of SpinOperator objects"
        )
