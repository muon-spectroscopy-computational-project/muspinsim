"""simconfig.py

Classes to generate and distribute configurations for simulations
"""

import re
import os
import logging
import datetime
from collections import OrderedDict, namedtuple
from collections.abc import Iterable
from itertools import product

import numpy as np
from ase.quaternions import Quaternion

from muspinsim.spinsys import MuonSpinSystem
from muspinsim.utils import quat_from_polar


# A dictionary of correspondence between keyword names and config parameters
_CDICT = {
    "polarization": "mupol",
    "field": "B",
    "time": "t",
    "orientation": "orient",
    "temperature": "T",
}

# Same but specifically for coupling types
_CSDICT = {
    "zeeman": "zmn",
    "dipolar": "dip",
    "quadrupolar": "quad",
    "hyperfine": "hfc",
    "dissipation": "dsp",
}

# Named tuple for snapshots
ConfigSnapshot = namedtuple("ConfigSnapshot", ["id", "y"] + list(_CDICT.values()))


# A useful decorator
def _validate_coupling_args(fun):
    def decorated(self, value, **args):
        ij = np.array([v for v in args.values() if v is not None])
        if (ij < 1).any() or (ij > len(self.system)).any():
            raise MuSpinConfigError("Out of range indices for coupling")

        return fun(self, value, **args)

    return decorated


def _validate_shape(v, target_shape=(3,), name="vector"):
    v = np.array(v)
    if v.shape != target_shape:
        raise MuSpinConfigError("Invalid shape for " "{0} coupling term".format(name))
    return v


# Another utility function


def _elems_from_arrayodict(inds, odict):
    return {k: v[inds[i]] for i, (k, v) in enumerate(odict.items())}


def _log_dictranges(rdict):

    for k, r in rdict.items():
        logging.info("\t\t{k} => {n} points".format(k=k, n=len(r)))


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

        if len(params) == 0:
            return

        # Basic parameters
        self._name = self.validate("name", params["name"].value)[0]
        self._spins = self.validate("spins", params["spins"].value[0])
        self._y_axis = self.validate("y", params["y_axis"].value[0])[0]

        # Identify ranges
        try:
            x = _CDICT[params["x_axis"].value[0][0]]
        except KeyError:
            raise MuSpinConfigError("Invalid x axis name in input file")
        self._x_range[x] = None

        for a in params["average_axes"].value.reshape((-1,)):
            if a.lower() == "none":
                # Special case
                continue
            try:
                self._avg_ranges[_CDICT[a]] = None
            except KeyError:
                raise MuSpinConfigError("Invalid average axis name in " "input file")

        self._time_N = 0  # Number of time points. This is special
        self._time_isavg = "t" in self._avg_ranges  # Is time averaged over?

        # Now inspect all parameters
        for iname, cname in _CDICT.items():

            try:
                p = params[iname]
            except KeyError:
                raise MuSpinConfigError(
                    "Invalid params object passed to "
                    "MuSpinConfig: "
                    "missing {0}".format(iname)
                )

            v = self.validate(cname, p.value, p.args)

            # Some additional special case treatment
            if cname == "t":
                if self._y_axis == "integral":
                    # Time is useless, might as well remove it
                    logging.warning("Ignoring time axis since Y = integral")
                    v = [np.inf]

                self._time_N = len(v)
            elif cname == "orient":
                if cname in self._avg_ranges:
                    # We need to normalize the weights so that they sum to N
                    norm = len(v) / np.sum(np.array([w for (q, w) in v]))
                    v = np.array([(q, w * norm) for (q, w) in v])
                else:
                    # Ignore the weights
                    v = np.array([(q, 1.0) for (q, w) in v])

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
        if self._y_axis == "integral":
            if "t" in self._x_range:
                raise MuSpinConfigError(
                    "Can not use time as X axis when " "evaluating integral of signal"
                )

        # If we're fitting, we can't have file ranges
        finfo = params["fitting_info"]
        if finfo["fit"]:
            if len(self._file_ranges) > 0:
                raise MuSpinConfigError("Can not have file ranges when " "fitting")
            # The x axis is overridden, whatever it is
            xname = list(self._x_range.keys())[0]
            self._constants.pop(xname, None)  # Just in case it was here
            self._x_range[xname] = finfo["data"][:, 0]
            if xname == "t":
                # Special case
                self._time_N = len(self._x_range[xname])

        # Check that a X axis was found
        if list(self._x_range.values())[0] is None:
            raise MuSpinConfigError("Specified x axis is not a range")

        # Remove anything that is not a range in averages
        self._avg_ranges = OrderedDict(
            **{k: v for k, v in self._avg_ranges.items() if v is not None}
        )

        logging.info("Using X axis:")
        _log_dictranges(self._x_range)
        if len(self._avg_ranges) > 0:
            logging.info("Averaging over:")
            _log_dictranges(self._avg_ranges)
        if len(self._file_ranges) > 0:
            logging.info("Scanning over:")
            _log_dictranges(self._file_ranges)

        # Now make the spin system
        self._system = MuonSpinSystem(self._spins)
        self._dissip_terms = {}

        for iid, idata in params["couplings"].items():
            iname = idata.name
            try:
                cname = _CSDICT[iname]
            except KeyError:
                raise MuSpinConfigError(
                    "Invalid params object passed to "
                    "MuSpinConfig: "
                    "unknown {0} coupling".format(iname)
                )

            cval = self.validate(cname, idata.value, idata.args)[0]

            i = idata.args.get("i")
            j = idata.args.get("j")

            # Move back from 1-based to 0-based indexing
            i = i - 1 if i is not None else None
            j = j - 1 if j is not None else None

            if cname == "zmn":
                # Zeeman coupling
                self._system.add_zeeman_term(i, cval)
            elif cname == "dip":
                # Dipolar coupling
                self._system.add_dipolar_term(i, j, cval)
            elif cname == "hfc":
                # Hyperfine coupling
                self._system.add_hyperfine_term(i, cval, j)
            elif cname == "quad":
                # Quadrupolar coupling
                self._system.add_quadrupolar_term(i, cval)
            elif cname == "dsp":
                # Dissipation. Special case, this is temperature dependent and
                # must be set individually
                self._dissip_terms[i] = cval[0]

        # Now for results, use the shapes of only file and x ranges
        res_shape = [len(v) for v in self._file_ranges.values()]
        res_shape += [len(v) for v in self._x_range.values()]

        self._results = np.zeros(res_shape)

        # And define the configurations, individually and collectively
        def make_configs(od):
            cfg = []
            for k, v in od.items():
                if k == "t":
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

        logging.info(
            "Total number of configurations to simulate: "
            "{0}".format(len(self._configurations))
        )
        logging.info(
            "Total number of configurations to average: " "{0}".format(self._avg_N)
        )

        # Define a namedtuple for configurations
        cfg_keys = list(self._constants.keys())
        cfg_keys += list(self._file_ranges.keys())
        cfg_keys += list(self._avg_ranges.keys())
        cfg_keys += list(self._x_range.keys())

    def validate(self, name, value, args={}):
        """Validate an input parameter with a custom method, if present.

        Arguments:
            name {str} -- Name of the input parameter
            value {any} -- Value to validate

        Keyword arguments:
            args {dict} -- Arguments, if present

        Returns:
            value -- Validated and normalised value
        """

        vname = "_validate_{0}".format(name)

        if hasattr(self, vname):
            vfun = getattr(self, vname)
            value = [vfun(v, **args) for v in value]

        return value

    def store_time_slice(self, config_id, tslice):
        """Store a time slice of data in this configuration's results.

        Store a time slice of data, given the configuration snapshot ID, and
        sum it to pre-existing data according to averaging rules.


        Arguments:
            config_id {tuple} -- ID of the ConfigurationSnapshot with which
                                 the data was calculated
            tslice {np.ndarray} -- Time slice of data
        """

        # Check the shape
        if isinstance(tslice, Iterable) and len(tslice) != self._time_N:
            raise ValueError("Time slice has invalid length")

        if self._time_isavg:
            tslice = np.average(tslice)

        ii = tuple(list(config_id[0]) + list(config_id[2]))
        self._results[ii] += tslice / self._avg_N

    def save_output(self, name=None, path=".", extension=".dat"):
        """Save all output files for the gathered results

        Save all output files for the gathered results, using an appropriate
        output path and seed name.

        Keyword arguments:
            name {str} -- Root name to use for the files
            path {str} -- Folder path to save the files in
            extension {str} -- Extension to save the files with
        """

        from muspinsim import __version__

        if name is None:
            name = self.name

        # Header format
        file_header = """MUSPINSIM v.{version}
Output file written on {date}
Parameters used:
""".format(
            version=__version__, date=datetime.datetime.now().ctime()
        )

        for fn in self._file_ranges:
            file_header = file_header + "\t{0:<20} = {{{0}}}\n".format(fn)

        indices = [range(len(rng)) for rng in self._file_ranges.values()]
        indices = product(*indices)

        # File name
        fid_pattern = "_{}" * len(self._file_ranges)
        fname_pattern = "{name}{id}{ext}"

        x = self.x_axis_values

        if "B" in self._x_range.keys():
            x = np.linalg.norm(x, axis=-1)

        # Actually save the files
        for inds in indices:
            fid = fid_pattern.format(*inds)
            fname = fname_pattern.format(name=name, id=fid, ext=extension)
            fname = os.path.join(path, fname)

            vdict = {}
            for i, (key, val) in enumerate(self._file_ranges.items()):
                pname = "_print_{0}".format(key)
                v = val[inds[i]]
                if hasattr(self, pname):
                    v = getattr(self, pname)(v)

                vdict[key] = v

            header = file_header.format(**vdict)

            data = np.zeros((len(x), 2))
            data[:, 0] = x
            data[:, 1] = self._results[inds]

            np.savetxt(fname, data, header=header)

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
    def dissipation_terms(self):
        return {**self._dissip_terms}

    @property
    def results(self):
        return np.array(self._results)

    @results.setter
    def results(self, r):
        r = np.array(r)
        if r.shape != self._results.shape:
            raise ValueError("Trying to set an invalid results array")
        self._results = r

    @property
    def x_axis(self):
        return [*self._x_range.keys()][0]

    @property
    def x_axis_values(self):
        return np.array([*self._x_range.values()][0])

    @property
    def y_axis(self):
        return self._y_axis

    def __len__(self):
        return len(self._configurations)

    def __getitem__(self, i):

        isint = type(i) == int
        if isint:
            i = slice(i, i + 1)
        elif type(i) != slice:
            raise TypeError(
                "Indices must be integer or slices, " "not {0}".format(type(i))
            )

        ans = []

        for (fc, ac, xc) in self._configurations[i]:

            fd = _elems_from_arrayodict(fc, self._file_ranges)
            ad = _elems_from_arrayodict(ac, self._avg_ranges)
            xd = _elems_from_arrayodict(xc, self._x_range)

            tp = ConfigSnapshot(
                id=(fc, ac, xc), y=self._y_axis, **self._constants, **fd, **ad, **xd
            )
            ans.append(tp)

        if isint:
            ans = ans[0]

        return ans

    def _validate_name(self, v):
        if len(v) > 1:
            raise MuSpinConfigError("Name must be a word without spaces")
        return v[0]

    def _validate_spins(self, v):

        # Find isotopes
        isore = re.compile("([0-9]+)([A-Z][a-z]*|e)")
        m = isore.match(v)
        if m is None:
            return v
        else:
            A, el = m.groups()
            A = int(A)
            return (el, A)

    def _validate_t(self, v):
        if len(v) != 1:
            raise MuSpinConfigError("Invalid line in time range")
        return v[0]

    def _validate_B(self, v):

        if len(v) == 1:
            v = np.array([0, 0, v[0]])  # The default direction is Z
        elif len(v) != 3:
            raise MuSpinConfigError("Invalid magnetic field value")

        return v

    def _validate_mupol(self, v):
        v = np.array(v, dtype=float)
        if len(v) != 3:
            raise MuSpinConfigError("Invalid muon polarization direction")

        v /= np.linalg.norm(v)

        return v

    def _validate_orient(self, v, mode):

        q = None
        w = 1.0  # Weight
        if len(v) == 2:
            # Interpret them as polar angles
            q = quat_from_polar(*v)
        elif len(v) == 3:
            q = Quaternion.from_euler_angles(*v, mode=mode)
        elif len(v) == 4:
            q = Quaternion.from_euler_angles(*v[:3], mode=mode)
            w = v[3]

        # After computing the rotation, we store the conjugate because it's a
        # lot cheaper, instead of rotating the whole system by q, to rotate
        # only the magnetic field and the polarization (lab frame) by the
        # inverse of q

        return (q.conjugate(), w)

    def _validate_T(self, v):
        return v[0]

    @_validate_coupling_args
    def _validate_zmn(self, v, i):
        return _validate_shape(v, (3,), "Zeeman")

    @_validate_coupling_args
    def _validate_dip(self, v, i, j):
        return _validate_shape(v, (3,), "dipolar")

    @_validate_coupling_args
    def _validate_hfc(self, v, i, j=None):
        return _validate_shape(v, (3, 3), "hyperfine")

    @_validate_coupling_args
    def _validate_quad(self, v, i):
        return _validate_shape(v, (3, 3), "quadrupolar")

    @_validate_coupling_args
    def _validate_dsp(self, v, i):
        return _validate_shape(v, (1,), "dissipation")

    def _print_B(self, v):
        return "{0} T".format(v)

    def _print_orient(self, v):
        o, w = v
        abc = o.euler_angles("zyz") * 180 / np.pi
        ostr = (
            "[ZYZ] a = {0:.1f} deg, b = {1:.1f} deg, " "c = {2:.1f} deg, weight = {3}"
        ).format(*abc, w)
        return ostr
