"""simconfig.py

Classes to generate and distribute configurations for simulations
"""

from collections.abc import Iterable
from numbers import Real

import numpy as np

from muspinsim.input.keyword import InputKeywords


# A dictionary of correspondence between keyword names and config parameters
_CDICT = {
    'name': 'name',
    'spins': 'spins',
    'polarization': 'mupol',
    'field': 'B',
    'time': 't',
    'x_axis': 'x',
    'y_axis': 'y',
    'orientation': 'orient',
    'temperature': 'T'
}


class MuSpinConfigError(Exception):
    pass


class MuSpinConfigRange(object):

    def __init__(self, values):

        if not isinstance(values, Iterable):
            raise ValueError('MuSpinConfigRange needs an iterable argument')

        self._values = values

    @property
    def values(self):
        return self._values


class MuSpinConfig(object):
    """A class to store a configuration for a MuSpinSim simulation, including
    any ranges of parameters as determined by the input file."""

    def __init__(self, params={}):
        """Initialise a MuSpinConfig object

        Initialise a MuSpinConfig object from values produced by the 
        .evaluate method of a MuSpinInput object.

        Arguments:
            params {dict} -- Dictionary of parameters as returned by 
                             MuSpinInput.evaluate

        """

        self._parameters = {}

        for iname, cname in _CDICT.items():
            self.set(cname, params[iname].value)

        # Systems

    def set(self, name, value):
        """Set the value of a configuration parameter

        Set the value of a configuration parameter, with preliminary use of 
        validation and conversion to MuSpinConfigRange if applicable.

        Arguments:
            name {str} -- Name of the parameter to set
            value {any} -- Parameter value

        """

        vname = '_validate_{0}'.format(name)

        if hasattr(self, vname):
            vfun = getattr(self, vname)
            value = list(map(vfun, value))

        if len(value) > 1:
            value = MuSpinConfigRange(value)
        elif len(value) == 1:
            value = value[0]

        self._parameters[name] = value

    def get(self, name):
        """Get the value of a configuration parameter

        Get the value of a configuration parameter, given its name.

        Arguments:
            name {str} -- Name of the parameter to get

        Returns:
            value {any} -- Parameter value
        """

        return self._parameters.get(name)

    def _validate_name(self, x):
        return x[0]

    def _validate_B(self, x):

        if len(x) == 1:
            x = np.array([0, 0, x[0]]) # The default direction is Z
        elif len(x) != 3:
            raise MuSpinConfigError('Invalid magnetic field value')

        return x

    def _validate_x(self, x):
        # Check that it's valid
        try:
            kw = InputKeywords[x[0]]
            if not kw.accept_as_x:
                raise KeyError()
        except KeyError:
            raise MuSpinConfigError('Invalid choice of X axis for simulation')

        return _CDICT[x[0]]

    def _validate_T(self, x):
        return x[0]
