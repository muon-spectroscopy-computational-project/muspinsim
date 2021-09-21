"""keyword.py

Classes to define and read conveniently individual keywords of an input file
"""

import numpy as np

from muspinsim.input.asteval import ASTExpression
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


class MuSpinKeyword(object):
    """Generic class used to parse a keyword from a MuSpinSim input file"""

    name = 'keyword'
    block_size = 1
    args_signature = []
    accept_range = True

    def __init__(self, block, args=[]):
        """Create an instance of a given keyword, passing the raw block of 
        text as well as the arguments.

        Arguments:
            block {[str]} -- Lines of text defining the value of the keyword
            args {[any]} -- Any arguments appearing after the keyword

        """

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


class MuSpinNumericalKeyword(MuSpinKeyword):
    """Specialised class for keywords with numerical values"""

    name = 'numerical_keyword'
    _functions = {**_math_functions}

    def __init__(self, block, args=[], variables=[]):

        self._variables = variables
        super(MuSpinNumericalKeyword, self).__init__(block, args)

    def _store_values(self, block):
        self._values = []
        for v in block:
            b = [[ASTExpression(x,
                                variables=self._variables,
                                functions=self._functions)
                  for x in l.split()] for l in v]
            if len(b) == 1:
                b = b[0]
            self._values.append(b)

    def evaluate(self, **variables):

        def expreval(expr):
            return expr.evaluate(**variables)

        return np.array(deepmap(expreval, self._values))

class MuSpinTensorKeyword(MuSpinNumericalKeyword):

    name = 'tensor_keyword'
    block_size = 3

