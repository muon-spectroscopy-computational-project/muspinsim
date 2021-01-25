"""input.py

Class to read in input files for the muspinsim script
"""

import re


def _read_list(raw, convert=str):
    return list(map(convert, raw.strip().split()))


def _read_tensor(raw, convert=float):
    data = []
    for l in raw:
        data.append(list(map(float, l.strip().split())))
    return data


def _has_data_size(l):
    def decorator(function):
        kw = function.__name__.split('_', 1)[1]

        def wrapper(self, *args):

            if len(args[-1]) != l:
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
        i = 0
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

        # Defaults
        self.spins = ['mu', 'e']
        self.polarization = 'transverse'
        self.field = [0.0]
        self.time = [0.0, 10.0, 100]
        self.save = {'evolution'}
        self.powder = None
        self.branch = None

        # Couplings
        self.hyperfine = {}
        self.dipolar = {}
        self.quadrupolar = {}

        for line, value in raw_blocks.items():
            key = line.split()[0]
            args = line.split()[1:]
            try:
                getattr(self, 'read_{0}'.format(key))(*args, value)
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
        isore = re.compile('([0-9]*)([A-Z][a-z]*)')
        for s in spins:
            m = isore.match(s)
            if m is None:
                self.spins.append(s)
            else:
                A, el = m.groups()
                A = int(A)
                self.spins.append((el, A))

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
    def read_powder(self, method, data):
        self.powder = (method, int(data[0]))

    def read_branch(self, data):
        self.branch = set(sum([_read_list(d) for d in data], []))

    @_has_data_size(3)
    def read_hyperfine(self, i, data):
        self.hyperfine[int(i)] = _read_tensor(data)

    @_has_data_size(1)
    def read_dipolar(self, i, j, data):
        self.dipolar[(int(i), int(j))] = _read_list(data[0], float)

    @_has_data_size(3)
    def read_quadrupolar(self, i, data):
        self.quadrupolar[int(i)] = _read_tensor(data)

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
