"""variables.py

Class defining a fitting variable with a starting value and bounds"""

import numpy as np


class FittingVariable(object):

    def __init__(self, name, value=0.0, minv=-np.inf, maxv=np.inf):

        self._name = name
        self._value = value
        self._min = minv
        self._max = maxv

    @property
    def name(self):
        return self._name
    
    @property
    def value(self):
        return self._value
    
    @property
    def bounds(self):
        return (self._min, self._max) 
