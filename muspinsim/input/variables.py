"""variables.py

Class defining a fitting variable with a starting value and bounds"""

import numpy as np


class FittingVariable(object):
    def __init__(self, name, value=0.0, minv=-np.inf, maxv=np.inf):

        self._name = name
        self._value = float(value)
        self._min = float(minv)
        self._max = float(maxv)

        invalid = ""
        if self._max <= self._min:
            invalid += (
                "Variable {0} has invalid range: "
                "(max value {1} cannot be less than or equal to min value {2}"
                ")\n".format(name, self._max, self._min)
            )
        if self._value > self._max:
            invalid += (
                "Variable {0} has invalid starting value: "
                "(starting value {1} cannot be greater than max value {2})".format(
                    name, self._value, self._max
                )
            )
        if self._value < self._min:
            invalid += (
                "Variable {0} has invalid starting value: "
                "(starting value {1} cannot be less than min value {2})".format(
                    name, self._value, self._min
                )
            )

        if invalid != "":
            raise ValueError(invalid)

    @property
    def name(self):
        return self._name

    @property
    def value(self):
        return self._value

    @property
    def bounds(self):
        return (self._min, self._max)
