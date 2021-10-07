"""variables.py

Class defining a fitting variable with a starting value and bounds"""

import numpy as np


class FittingVariable(object):
    def __init__(self, name, value=0.0, minv=-np.inf, maxv=np.inf):

        self._name = name
        self._value = float(value)
        self._min = float(minv)
        self._max = float(maxv)

        if self._max <= self._min:
            raise ValueError("Variable {0} has invalid range".format(name))
        elif self._value > self._max or self._value < self._min:
            raise ValueError("Variable {0} has invalid starting " "value".format(name))

    @property
    def name(self):
        return self._name

    @property
    def value(self):
        return self._value

    @property
    def bounds(self):
        return (self._min, self._max)
