"""simconfig.py

Classes to generate and distribute configurations for simulations
"""

import logging
from collections import OrderedDict, namedtuple
from collections.abc import Iterable
from itertools import product
from numbers import Real

import numpy as np

from muspinsim.input.keyword import InputKeywords
from muspinsim.spinsys import MuonSpinSystem


# A dictionary of correspondence between keyword names and config parameters
_CDICT = {
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


def _validate_shape(v, target_shape=(3,), name='vector'):
    v = np.array(v)
    if v.shape != target_shape:
        raise MuSpinConfigError('Invalid shape for '
                                '{0} coupling term'.format(name))
    return v


# Another utility function

def _elems_from_arrayodict(inds, odict):
    return {k: v[inds[i]] for i, (k, v) in enumerate(odict.items())}


class MuSpinConfigError(Exception):
    pass


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

        self._constants = {}
        self._file_ranges = OrderedDict()
        self._avg_ranges = OrderedDict()
        self._x_range = OrderedDict()

        # Basic parameters
        self._name = self.validate('name', params['name'].value)[0]
        self._spins = self.validate('spins', params['spins'].value[0])
        self._y_axis = self.validate('y', params['y_axis'].value[0])[0]

        # Identify ranges
        try:
            x = _CDICT[params['x_axis'].value[0][0]]
        except KeyError:
            raise MuSpinConfigError('Invalid x axis name in input file')
        self._x_range[x] = None

        for a in params['average_axes'].value.reshape((-1,)):
            try:
                self._avg_ranges[_CDICT[a]] = None
            except KeyError:
                raise MuSpinConfigError('Invalid average axis name in '
                                        'input file')

        self._time_N = 0  # Number of time points. This is special
        self._time_isavg = ('t' in self._avg_ranges)  # Is time averaged over?

        # Now inspect all parameters
        for iname, cname in _CDICT.items():

            try:
                p = params[iname]
            except KeyError:
                raise MuSpinConfigError('Invalid params object passed to '
                                        'MuSpinConfig: '
                                        'missing {0}'.format(iname))

            v = self.validate(cname, p.value, p.args)

            if cname == 't':
                if self._y_axis == 'integral':
                    # Time is useless, might as well remove it
                    logging.warning('Ignoring time axis since Y = integral')
                    v = [np.inf]

                self._time_N = len(v)

            if len(v) > 1:
                # It's a range
                if cname in self._x_range:
                    self._x_range[cname] = v
                elif cname in self._avg_ranges:
                    self._avg_ranges[cname] = v
                else:
                    self._file_ranges[cname] = v
            else:
                # It's a constant
                self._constants[cname] = v[0]

        # Check that the Y axis and time are consistent
        if self._y_axis == 'integral':
            if 't' in self._x_range:
                raise MuSpinConfigError('Can not use time as X axis when '
                                        'evaluating integral of signal')            

        # Check that a X axis was found
        if None in self._x_range.values():
            raise MuSpinConfigError('Specified x axis is not a range')

        # Remove anything that is not a range in averages
        self._avg_ranges = OrderedDict(**{k: v
                                          for k, v in self._avg_ranges.items()
                                          if v is not None})

        # Now make the spin system
        self._system = MuonSpinSystem(self._spins)
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

        # Now for results, use the shapes of only file and x ranges
        res_shape = [len(v) for v in self._file_ranges.values()]
        res_shape += [len(v) for v in self._x_range.values()]

        self._results = np.zeros(res_shape)

        # And define the configurations, individually and collectively
        def make_configs(od):
            cfg = []
            for k, v in od.items():
                if k == 't':
                    cfg.append([slice(None)])
                else:
                    cfg.append(np.arange(len(v)))
            return list(product(*cfg))

        fconfigs = make_configs(self._file_ranges)
        aconfigs = make_configs(self._avg_ranges)
        xconfigs = make_configs(self._x_range)

        # Total configurations
        self._configurations = list(product(*[fconfigs, aconfigs, xconfigs]))
        # Size of the average
        self._avg_N = len(aconfigs)

        # Define a namedtuple for configurations
        cfg_keys = list(self._constants.keys())
        cfg_keys += list(self._file_ranges.keys())
        cfg_keys += list(self._avg_ranges.keys())
        cfg_keys += list(self._x_range.keys())

        self._cfg_tuple = namedtuple('ConfigSnapshot', ['id'] + cfg_keys)

    def validate(self, name, value, args={}):

        vname = '_validate_{0}'.format(name)

        if hasattr(self, vname):
            vfun = getattr(self, vname)
            value = [vfun(v, args) for v in value]

        return value

    def store_time_slice(self, config_id, tslice):
        # Check the shape
        if len(tslice) != self._time_N:
            raise ValueError('Time slice has invalid length')

        if self._time_isavg:
            tslice = np.average(tslice)

        ii = tuple(list(config_id[0]) + list(config_id[2]))
        self._results[ii] += tslice/self._avg_N

    @property
    def name(self):
        return self._name

    @property
    def spins(self):
        return list(self._spins)

    @property
    def system(self):
        return self._system.clone()

    @property
    def constants(self):
        return {**self._constants}

    @property
    def results(self):
        return np.array(self._results)

    def __len__(self):
        return len(self._configurations)

    def __getitem__(self, i):

        isint = type(i) == int
        if isint:
            i = slice(i, i+1)
        elif type(i) != slice:
            raise TypeError('Indices must be integer or slices, '
                            'not {0}'.format(type(i)))

        ans = []

        for (fc, ac, xc) in self._configurations[i]:

            fd = _elems_from_arrayodict(fc, self._file_ranges)
            ad = _elems_from_arrayodict(ac, self._avg_ranges)
            xd = _elems_from_arrayodict(xc, self._x_range)

            tp = self._cfg_tuple(id=(fc, ac, xc),
                                 **self._constants, **fd, **ad, **xd)
            ans.append(tp)

        if isint:
            ans = ans[0]

        return ans

    def _validate_name(self, v, a={}):
        if len(v) > 1:
            raise MuSpinConfigError('Name must be a word without spaces')
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
        return _validate_shape(v, (3,), 'Zeeman')

    @_validate_coupling_args
    def _validate_dip(self, v, a={}):
        return _validate_shape(v, (3,), 'dipolar')

    @_validate_coupling_args
    def _validate_hfc(self, v, a={}):
        return _validate_shape(v, (3, 3), 'hyperfine')

    @_validate_coupling_args
    def _validate_quad(self, v, a={}):
        return _validate_shape(v, (3, 3), 'quadrupolar')

    @_validate_coupling_args
    def _validate_dsp(self, v, a={}):
        return _validate_shape(v, (1,), 'dissipation')