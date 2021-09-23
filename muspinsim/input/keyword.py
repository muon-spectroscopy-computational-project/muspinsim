"""keyword.py

Classes to define and read conveniently individual keywords of an input file
"""

import re
import sys
import inspect
import numpy as np

from muspinsim.input.asteval import ASTExpression, ast_tokenize
from muspinsim.utils import deepmap

# Supported math functions
_math_functions = {
    'sin': np.sin,
    'cos': np.cos,
    'tan': np.tan,
    'arcsin': np.arcsin,
    'arccos': np.arccos,
    'arctan': np.arctan,
    'arctan2': np.arctan2,
    'exp': np.exp,
    'sqrt': np.sqrt
}

# And math constants
_math_constants = {
    'pi': np.pi,
    'e': np.exp(1)
}

# Expansion functions


def _range(x1, x2, n=100):
    return np.linspace(x1, x2, n)[:, None]


class MuSpinKeyword(object):
    """Generic class used to parse a keyword from a MuSpinSim input file"""

    name = 'keyword'
    block_size = 1
    block_size_bounds = (1, np.inf)
    args_signature = []
    accept_range = True
    default = None

    def __init__(self, block=[], args=[]):
        """Create an instance of a given keyword, passing the raw block of 
        text as well as the arguments.

        Arguments:
            block {[str]} -- Lines of text defining the value of the keyword
            args {[any]} -- Any arguments appearing after the keyword

        """

        # Sanity check
        if (self.block_size < self.block_size_bounds[0] or
                self.block_size > self.block_size_bounds[1]):
            raise RuntimeError(
                'Invalid block_size for keyword {0}'.format(self.name))

        self._store_args(args)

        # Reshape block
        block = np.array(block)
        try:
            block = block.reshape((-1, self.block_size))
        except ValueError:
            raise RuntimeError('Invalid block length for '
                               'keyword {0}'.format(self.name))
        if not self.accept_range and len(block) > 1:
            raise RuntimeError('Can not accept range of values for '
                               'keyword {0}'.format(self.name))

        if len(block) == 0:
            block = np.array([[self.default]])

        self._store_values(block)

    def _store_args(self, args):
        # Check that the arguments are correct
        self._args = []
        for i, a in enumerate(args):
            try:
                self._args.append(self.args_signature[i](a))
            except IndexError:
                raise RuntimeError('Too many arguments passed to '
                                   'keyword {0}'.format(self.name))
            except ValueError:
                raise RuntimeError('Invalid argument type passed to '
                                   'keyword {0}'.format(self.name))

    def _store_values(self, block):
        # Parse and store each value separately
        self._values = []
        for v in block:
            b = [l.split() for l in v]
            if len(b) == 1:
                b = b[0]
            self._values.append(b)
        self._values = np.array(self._values)

    @property
    def arguments(self):
        return tuple(self._args)

    def evaluate(self):
        return self._values.copy()

    def __len__(self):
        return len(self._values)


class MuSpinEvaluateKeyword(MuSpinKeyword):
    """Specialised class for keywords with values that need evaluating"""

    name = 'evaluate_keyword'
    _functions = {**_math_functions}
    _constants = {**_math_constants}

    def __init__(self, block=[], args=[], variables=[]):

        cnames = set(self._constants.keys())
        vnames = set(variables)

        conflicts = cnames.intersection(vnames)
        if len(conflicts) > 0:
            raise ValueError('Variable names {0}'.format(conflicts) +
                             ' conflict with existing constants')

        self._variables = list(cnames.union(vnames))
        super(MuSpinEvaluateKeyword, self).__init__(block, args)

    def _store_values(self, block):
        self._values = []
        for v in block:
            b = [[ASTExpression(tk,
                                variables=self._variables,
                                functions=self._functions)
                  for tk in ast_tokenize(l)] for l in v]
            if len(b) == 1:
                b = b[0]
            self._values.append(b)

    def evaluate(self, **variables):

        allvars = {**variables, **self._constants}

        def expreval(expr):
            return expr.evaluate(**allvars)

        return np.array(deepmap(expreval, self._values))


class MuSpinExpandKeyword(MuSpinEvaluateKeyword):
    """Specialised class for keywords whose lines can be expanded with
    particular functions"""

    name = 'expand_keyword'
    block_size_bounds = (1, 1)
    _functions = {**_math_functions,
                  'range': _range}

    def evaluate(self, **variables):

        allvars = {**variables, **self._constants}

        if self.block_size != 1:
            raise RuntimeError(
                'MuSpinExpandKeyword can not have block_size > 1')

        def expreval(expr):
            return expr.evaluate(**allvars)

        eval_values = []
        for line in self._values:
            eval_line = np.array([expreval(expr) for expr in line])

            if len(line) == 1 and len(eval_line.shape) == 3:
                eval_values += list(eval_line[0])
            elif len(line) == 1 and len(eval_line.shape) == 2:
                eval_values += [eval_line[0]]
            elif len(eval_line.shape) == 1:
                eval_values += [eval_line]
            else:
                raise RuntimeError('Invalid line for '
                                   'keyword {0}'.format(self.name))

        return eval_values


class MuSpinTensorKeyword(MuSpinEvaluateKeyword):

    name = 'tensor_keyword'
    block_size = 3
    block_size_bounds = (3, 3)


# Now on to defining the actual keywords that are admitted in input files
class KWName(MuSpinKeyword):

    name = 'name'
    block_size = 1
    accept_range = False
    default = ''


class KWSpins(MuSpinKeyword):

    name = 'spins'
    block_size = 1
    accept_range = False
    default = 'mu e'


class KWPolarization(MuSpinExpandKeyword):

    name = 'polarization'
    block_size = 1
    default = 'transverse'
    _constants = {**_math_constants,
                  'longitudinal': np.array([0, 0, 1]),
                  'transverse':   np.array([1, 0, 0])
                  }
    _functions = {**_math_functions}


# Compile all KW classes into a single dictionary automatically
InputKeywords = {
    obj.name: obj
    for name, obj in inspect.getmembers(sys.modules[__name__])
    if (inspect.isclass(obj) and re.match('KW[0-9a-zA-Z]+', name))
}
