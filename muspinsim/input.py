"""input.py

Class to read in input files for the muspinsim script
"""

import re
import numpy as np


def _read_list(raw, convert=str):
    return list(map(convert, raw.strip().split()))


def _read_tensor(raw, convert=float):
    data = []
    for l in raw:
        data.append(list(map(convert, l.strip().split())))
    return data


def _has_data_size(l):
    def decorator(function):
        kw = function.__name__.split('_', 1)[1]

        def wrapper(self, *args):

            if len(args[0]) != l:
                raise RuntimeError('Invalid block length for keyword ' + kw)

            return function(self, *args)

        return wrapper

    return decorator

class MuSpinInput(object):

    def __init__(self, fs):
        """Read in an input file

        Read in an input file from an opened file stream        

        Arguments:
            fs {TextIOBase} -- I/O stream (should be file, can be StringIO)
        """

        lines = fs.readlines()

        # Split lines in blocks
        raw_blocks = {}
        curr_block = None

        indre = re.compile('(\\s+)[^\\s]')
        indent = None

        for l in lines:
            if l.strip() == '' or l[0] == '#':
                continue  # It's a comment
            m = indre.match(l)
            if m:
                if indent is None:
                    indent = m.groups()[0]
                if m.groups()[0] != indent:
                    raise RuntimeError('Invalid indent in input file')
                else:
                    try:
                        raw_blocks[curr_block].append(l.strip())
                    except KeyError:
                        raise RuntimeError('Badly formatted input file')
            else:
                curr_block = l.strip()
                raw_blocks[curr_block] = []
                indent = None  # Reset for each block

        # Defaults
        self.name = None
        self.spins = ['mu', 'e']
        self.polarization = 'transverse'
        self.field = [0.0]
        self.time = [0.0, 10.0, 100]
        self.save = {'evolution'}
        self.powder = None
        self.temperature = np.inf

        # Couplings
        self.zeeman = {}
        self.hyperfine = {}
        self.dipolar = {}
        self.quadrupolar = {}
        self.dissipation = {}

        for line, value in raw_blocks.items():
            key = line.split()[0]
            args = line.split()[1:]
            try:
                getattr(self, 'read_{0}'.format(key))(value, *args)
            except AttributeError:
                raise RuntimeError(
                    'Invalid keyword {0} in input file'.format(key))
            except TypeError:
                raise RuntimeError('Invalid arguments for keyword '
                                   '{0}'.format(key))

    @_has_data_size(1)
    def read_spins(self, data):
        self.spins = []
        spins = _read_list(data[0])

        # Find isotopes
        isore = re.compile('([0-9]+)([A-Z][a-z]*|e)')
        for s in spins:
            m = isore.match(s)
            if m is None:
                self.spins.append(s)
            else:
                A, el = m.groups()
                A = int(A)
                self.spins.append((el, A))

    @_has_data_size(1)
    def read_name(self, data):
        self.name = data[0].strip().lower()

    @_has_data_size(1)
    def read_polarization(self, data):
        self.polarization = data[0].strip().lower()

    @_has_data_size(1)
    def read_field(self, data):
        self.field = _read_list(data[0], float)

    @_has_data_size(1)
    def read_time(self, data):
        self.time = _read_list(data[0], float)

    def read_save(self, data):
        self.save = set(sum([_read_list(d) for d in data], []))

    @_has_data_size(1)
    def read_powder(self, data, method):
        self.powder = (method, int(data[0]))

    @_has_data_size(1)
    def read_temperature(self, data):
        if data[0].lower() in ('inf', 'infinity'):
            self.temperature = np.inf
        else:
            self.temperature = float(data[0])

    @_has_data_size(1)
    def read_zeeman(self, data, i):
        i = int(i)-1
        self.zeeman[i] = _read_list(data[0], float)

    @_has_data_size(3)
    def read_hyperfine(self, data, i, j=None):
        i = int(i)-1
        j = int(j)-1 if j else None
        self.hyperfine[(i, j)] = _read_tensor(data)

    @_has_data_size(1)
    def read_dipolar(self, data, i, j):
        i = int(i)-1
        j = int(j)-1
        self.dipolar[(i, j)] = _read_list(data[0], float)

    @_has_data_size(3)
    def read_quadrupolar(self, data, i):
        i = int(i)-1
        self.quadrupolar[i] = _read_tensor(data)

    @_has_data_size(1)
    def read_dissipation(self, data, i):
        i = int(i)-1
        self.dissipation[i] = float(data[0])

    @_has_data_size(1)
    def read_experiment(self, data):
        exptype = data[0].strip().lower()

        # Set different parameters depending on experiment
        if exptype == 'zero_field':
            self.field = [0.0]
            self.polarization = 'transverse'
            self.save = {'evolution'}
        elif exptype == 'longitudinal':
            self.polarization = 'longitudinal'
            self.save = {'evolution'}
        elif exptype == 'alc':
            self.polarization = 'longitudinal'
            self.save = {'integral'}
            self.time = [0.0]
