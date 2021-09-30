"""simconfig.py

Classes to generate and distribute configurations for simulations
"""

from collections.abc import Iterable
from numbers import Real

import numpy as np

from muspinsim.input.keyword import InputKeywords
from muspinsim.spinsys import MuonSpinSystem


# A dictionary of correspondence between keyword names and config parameters
_CDICT = {
    'name': 'name',
    'spins': 'spins',
    'polarization': 'mupol',
    'field': 'B',
    'time': 't',
    'orientation': 'orient',
    'temperature': 'T'
}

# Same but specifically for coupling types
_CSDICT = {
    'zeeman': 'zmn',
    'dipolar': 'dip',
    'quadrupolar': 'quad',
    'hyperfine': 'hfc',
    'dissipation': 'dsp'
}

# A useful decorator


def _validate_coupling_args(fun):
    def decorated(self, value, args={}):
        ij = np.array(list(args.values()))
        if (ij < 1).any() or (ij > len(self.system)).any():
            raise MuSpinConfigError('Out of range indices for coupling')

        return fun(self, value, args)

    return decorated


def _validate_vector(v, name='vector'):
    v = np.array(v)
    if v.shape != (3,):
        raise MuSpinConfigError('Invalid shape for '
                                '{0} coupling term'.format(name))
    return v


def _validate_tensor(v, name='tensor'):
    v = np.array(v)
    if v.shape != (3, 3):
        raise MuSpinConfigError('Invalid shape for '
                                '{0} coupling term'.format(name))
    return v


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

    def __len__(self):
        return len(self._values)

    def __getitem__(self, i):
        return self._values[i]


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
        self._arguments = {}

        for iname, cname in _CDICT.items():
            try:
                p = params[iname]
                self.set(cname, p.value, p.args)
            except KeyError:
                raise MuSpinConfigError('Invalid params object passed to '
                                        'MuSpinConfig: '
                                        'missing {0}'.format(iname))

        # A bit of a special treatment for axes stuff
        self._x_axis = _CDICT[params['x_axis'].value[0][0]]
        if not isinstance(self.get(self._x_axis), MuSpinConfigRange):
            raise MuSpinConfigError('Designated x axis does not have a range '
                                    'of values')
        self._y_axis = params['y_axis'].value[0][0]
        self._avg_axes = map(
            _CDICT.get, params['average_axes'].value.reshape((-1,)))

        # A special case for the Y-axis
        if self._y_axis == 'integral':
            # Then time doesn't matter
            self._parameters['t'] = None
            if self._x_axis == 't':
                raise MuSpinConfigError('Can not use time as x axis for '
                                        'integrated asymmetry')

        # Now make a spin system
        self._system = MuonSpinSystem(self.get('spins'))
        self._dissip_terms = {}

        for iid, idata in params['couplings'].items():
            iname = idata.name
            try:
                cname = _CSDICT[iname]
            except KeyError:
                raise MuSpinConfigError('Invalid params object passed to '
                                        'MuSpinConfig: '
                                        'unknown {0} coupling'.format(iname))

            cval = self.validate(cname, idata.value, idata.args)[0]

            i = idata.args.get('i')
            j = idata.args.get('j')

            # Move back from 1-based to 0-based indexing
            i = i-1 if i is not None else None
            j = j-1 if j is not None else None

            if cname == 'zmn':
                # Zeeman coupling
                self._system.add_zeeman_term(i, cval)
            elif cname == 'dip':
                # Dipolar coupling
                self._system.add_dipolar_term(i, j, cval)
            elif cname == 'hfc':
                # Hyperfine coupling
                self._system.add_hyperfine_term(i, cval, j)
            elif cname == 'quad':
                # Quadrupolar coupling
                self._system.add_quadrupolar_term(i, cval)
            elif cname == 'dsp':
                # Dissipation. Special case, this is temperature dependent and
                # must be set individually
                self._dissip_terms[i] = cval

        # Compile which specific words have ranges
        self._range_axes = {k for k, p in self._parameters.items()
                            if isinstance(p, MuSpinConfigRange)}

        self._avg_axes = set(self._avg_axes).intersection(self._range_axes)
        self._file_axes = self._range_axes.difference(self._avg_axes)
        self._file_axes = self._file_axes.difference({self._x_axis})

        # Turn them into tuples to preserve a fixed order
        self._range_axes = tuple(sorted(self._range_axes))
        self._avg_axes = tuple(sorted(self._avg_axes))
        self._file_axes = tuple(sorted(self._file_axes))

    def validate(self, name, value, args={}):

        vname = '_validate_{0}'.format(name)

        if hasattr(self, vname):
            vfun = getattr(self, vname)
            value = [vfun(v, args) for v in value]

        return value

    def set(self, name, value, args={}):
        """Set the value of a configuration parameter

        Set the value of a configuration parameter, with preliminary use of
        validation and conversion to MuSpinConfigRange if applicable.

        Arguments:
            name {str} -- Name of the parameter to set
            value {any} -- Parameter value
            args {dict} -- Dictionary of arguments

        """

        value = self.validate(name, value, args)

        if len(value) > 1:
            value = MuSpinConfigRange(value)
        elif len(value) == 1:
            value = value[0]

        self._parameters[name] = value
        self._arguments[name] = args

    def get(self, name):
        """Get the value of a configuration parameter

        Get the value of a configuration parameter, given its name

        Arguments:
            name {str} -- Name of the parameter to get

        Returns:
            value {any} -- Parameter value
        """

        return self._parameters.get(name)

    def get_args(self, name):
        """Get the arguments of a configuration parameter

        Get the arguments of a configuration parameter, given its name

        Arguments:
            name {str} -- Name of the parameter to get

        Returns:
            args {dict} -- Parameter arguments
        """

        return self._arguments.get(name)

    def get_frange_params(self, indices=[]):

        if len(indices) != len(self._file_axes):
            raise MuSpinConfigError('Indices must match file axes when '
                                    'fetching file parameters')

        fpars = {}

        for i, fi in enumerate(indices):
            key = self._file_axes[i]
            fpars[key] = self._parameters[key][fi]

        return fpars

    @property
    def params(self):
        return {**self._parameters}

    @property
    def args(self):
        return {**self._arguments}

    @property
    def system(self):
        return self._system

    @property
    def constants(self):
        # All parameters that are not in a range
        cnst = {}
        for k, v in self._parameters.items():
            if not (k in self._range_axes):
                cnst[k] = v

        return cnst

    def _validate_name(self, v, a={}):
        return v[0]

    def _validate_t(self, v, a={}):
        if len(v) != 1:
            raise MuSpinConfigError('Invalid line in time range')
        return v[0]

    def _validate_B(self, v, a={}):

        if len(v) == 1:
            v = np.array([0, 0, v[0]])  # The default direction is Z
        elif len(v) != 3:
            raise MuSpinConfigError('Invalid magnetic field value')

        return v

    def _validate_T(self, v, a={}):
        return v[0]

    @_validate_coupling_args
    def _validate_zmn(self, v, a={}):
        return _validate_vector(v, 'Zeeman')

    @_validate_coupling_args
    def _validate_dip(self, v, a={}):
        return _validate_vector(v, 'dipolar')

    @_validate_coupling_args
    def _validate_hfc(self, v, a={}):
        return _validate_tensor(v, 'hyperfine')

    @_validate_coupling_args
    def _validate_quad(self, v, a={}):
        return _validate_tensor(v, 'quadrupolar')


class MuSpinFileSim(object):

    def __init__(self, config, file_indices):

        self._cfg = config
        self._cnst = config.constants
        self._fvals = config.get_frange_params(file_indices)

        print(self._fvals)
