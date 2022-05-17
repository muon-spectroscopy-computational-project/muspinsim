"""keyword.py

Classes to define and read conveniently individual keywords of an input file
"""

import re
import sys
import inspect
import numpy as np

from muspinsim.constants import MU_GAMMA
from muspinsim.input.larkeval import LarkExpression, lark_tokenize
from muspinsim.input.variables import FittingVariable
from muspinsim.utils import deepmap, zcw_gen, eulrange_gen

# Supported math functions
_math_functions = {
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "arcsin": np.arcsin,
    "arccos": np.arccos,
    "arctan": np.arctan,
    "arctan2": np.arctan2,
    "exp": np.exp,
    "log": np.log,
    "sqrt": np.sqrt,
}

# And math constants
_math_constants = {"pi": np.pi, "deg": np.pi / 180.0, "e": np.exp(1), "inf": np.inf}

_phys_constants = {"muon_gyr": MU_GAMMA, "MHz": 1.0 / (2 * MU_GAMMA)}

# Functions for powder orientation

_pwd_functions = {"zcw": zcw_gen, "eulrange": eulrange_gen}


# Expansion functions
def _range(x1, x2, n=100):
    return np.linspace(x1, x2, int(n))[:, None]


class MuSpinKeyword(object):
    """Generic class used to parse a keyword from a MuSpinSim input file"""

    name = "keyword"
    block_size = 1
    block_size_bounds = (1, np.inf)
    accept_range = True
    accept_as_x = False
    default = None
    _validators = {}

    def __init__(self, block=[], args=[]):
        """Create an instance of a given keyword, passing the raw block of
        text as well as the arguments.

        Arguments:
            block {[str]} -- Lines of text defining the value of the keyword
            args {[any]} -- Any arguments appearing after the keyword

        """

        # Sanity check
        if (
            self.block_size < self.block_size_bounds[0]
            or self.block_size > self.block_size_bounds[1]
        ):
            raise RuntimeError("Invalid block_size for keyword {0}".format(self.name))

        self._store_args(args)

        # Reshape block
        block = np.array(block)
        try:
            block = block.reshape((-1, self.block_size))
        except ValueError:
            raise RuntimeError(
                "Invalid block length for " "keyword {0}".format(self.name)
            )
        if not self.accept_range and len(block) > 1:
            raise RuntimeError(
                "Can not accept range of values for " "keyword {0}".format(self.name)
            )

        if len(block) == 0 and self.has_default:
            block = np.array([self.default.split("\n")])

        self._store_values(block)

        self._validate_values()

    def _default_args(self):
        # Dummy function, used for type signature and processing of arguments
        return {}

    def _store_args(self, args):
        try:
            self._args = self._default_args(*args)
        except TypeError:
            raise RuntimeError(
                "Wrong number of arguments passed to " "keyword {0}".format(self.name)
            )
        except ValueError:
            raise RuntimeError(
                "Invalid argument type passed to " "keyword {0}".format(self.name)
            )

    def _store_values(self, block):
        # Parse and store each value separately
        self._values = []
        for v in block:
            b = [l.split() for l in v]
            if len(b) == 1:
                b = b[0]
            self._values.append(b)
        self._values = np.array(self._values)

    def _validate_values(self):

        for rule, vfunc in self._validators.items():
            ans = all([vfunc(b) for b in self._values])

            if not ans:
                raise ValueError(
                    "Invalid block for " "keyword {0}: {1}".format(self.name, rule)
                )

    @property
    def arguments(self):
        return {**self._args}

    @property
    def id(self):
        return self.name

    @property
    def has_default(self):
        return self.default is not None

    def evaluate(self):
        return self._values.copy()

    def __len__(self):
        return len(self._values)


class MuSpinEvaluateKeyword(MuSpinKeyword):
    """Specialised class for keywords with values that need evaluating"""

    name = "evaluate_keyword"
    _functions = {**_math_functions}
    _constants = {**_math_constants}

    def __init__(self, block=[], args=[], variables=[]):

        cnames = set(self._constants.keys())
        fnames = set(self._functions.keys())
        vnames = set(variables)

        conflicts = cnames.intersection(vnames)
        conflicts = conflicts.union(fnames.intersection(vnames))
        if len(conflicts) > 0:
            raise ValueError(
                "Variable names {0}".format(conflicts)
                + " conflict with existing constants"
            )

        self._variables = list(cnames.union(vnames))
        super(MuSpinEvaluateKeyword, self).__init__(block, args)

    def _store_values(self, block):
        self._values = []
        for v in block:
            b = [
                [
                    LarkExpression(
                        tk, variables=self._variables, functions=self._functions
                    )
                    for tk in lark_tokenize(l)
                ]
                for l in v
            ]
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

    name = "expand_keyword"
    block_size_bounds = (1, 1)
    _functions = {**_math_functions, "range": _range}

    def evaluate(self, **variables):

        allvars = {**variables, **self._constants}

        if self.block_size != 1:
            raise RuntimeError("MuSpinExpandKeyword can not have block_size > 1")

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
                raise RuntimeError(
                    "Unable to evaluate expression for " "keyword {0}".format(self.name)
                )

        return eval_values


class MuSpinCouplingKeyword(MuSpinEvaluateKeyword):

    name = "coupling_keyword"
    block_size = 1
    default = "0 0 0"

    def _default_args(self, i=None, j=None):
        args = {
            "i": int(i) if (i is not None) else None,
            "j": int(j) if (j is not None) else None,
        }
        return args

    @property
    def accept_range(self):
        return False  # Never ranges! These keywords define the system!

    @property
    def id(self):
        id_str = ""
        i = self._args.get("i")
        j = self._args.get("j")
        if i is not None:
            id_str += "_{0}".format(i)
            if j is not None:
                id_str += "_{0}".format(j)
        return "{0}{1}".format(self.name, id_str)


# Now on to defining the actual keywords that are admitted in input files
class KWName(MuSpinKeyword):

    name = "name"
    block_size = 1
    accept_range = False
    default = "muspinsim"


class KWSpins(MuSpinKeyword):

    name = "spins"
    block_size = 1
    accept_range = False
    default = "mu e"


class KWPolarization(MuSpinExpandKeyword):

    name = "polarization"
    block_size = 1
    default = "transverse"
    _constants = {
        **_math_constants,
        "longitudinal": np.array([0, 0, 1.0]),
        "transverse": np.array([1.0, 0, 0]),
    }
    _functions = {**_math_functions}


class KWField(MuSpinExpandKeyword):

    name = "field"
    block_size = 1
    accept_range = True
    accept_as_x = True
    default = "0.0"
    _constants = {**_math_constants, **_phys_constants}


class KWTime(MuSpinExpandKeyword):

    name = "time"
    block_size = 1
    accept_range = True
    accept_as_x = True
    default = "range(0, 10, 101)"


class KWXAxis(MuSpinKeyword):

    name = "x_axis"
    block_size = 1
    accept_range = False
    default = "time"
    _validators = {
        "Invalid value": lambda s: (
            s[0] in InputKeywords and InputKeywords[s[0]].accept_as_x
        )
    }


class KWYAxis(MuSpinKeyword):

    name = "y_axis"
    block_size = 1
    accept_range = False
    default = "asymmetry"
    _validators = {"Invalid value": lambda s: s in ("asymmetry", "integral")}


class KWAverageAxes(MuSpinKeyword):

    name = "average_axes"
    block_size = 1
    accept_range = True
    default = "orientation"
    _validators = {
        "Invalid value": lambda s: all(
            (w in InputKeywords or w.lower() == "none") for w in s
        )
    }


class KWOrientation(MuSpinExpandKeyword):

    name = "orientation"
    block_size = 1
    accept_range = True
    default = "0 0 0"
    _functions = {**_math_functions, **_pwd_functions}

    def _default_args(self, mode="zyz"):
        args = {"mode": mode}
        return args


class KWTemperature(MuSpinExpandKeyword):

    name = "temperature"
    block_size = 1
    accept_range = True
    accept_as_x = True
    default = "inf"


# Couplings
class KWZeeman(MuSpinCouplingKeyword):

    name = "zeeman"
    block_size = 1
    _constants = {**_math_constants, **_phys_constants}

    def _default_args(self, i):
        args = {"i": int(i)}
        return args


class KWDipolar(MuSpinCouplingKeyword):

    name = "dipolar"
    block_size = 1

    def _default_args(self, i, j):
        args = {"i": int(i), "j": int(j)}
        return args


class KWHyperfine(MuSpinCouplingKeyword):

    name = "hyperfine"
    block_size = 3

    def _default_args(self, i, j=None):
        args = {"i": int(i), "j": int(j) if j is not None else None}
        return args


class KWQuadrupolar(MuSpinCouplingKeyword):

    name = "quadrupolar"
    block_size = 3

    def _default_args(self, i):
        args = {
            "i": int(i),
        }
        return args


class KWDissipation(MuSpinCouplingKeyword):

    name = "dissipation"
    block_size = 1

    def _default_args(self, i):
        args = {"i": int(i)}
        return args


# Fitting variables. This is a special case
class KWFittingVariables(MuSpinKeyword):

    name = "fitting_variables"
    block_size = 1
    accept_range = True
    default = ""
    _constants = {**_math_constants, **_phys_constants}

    def _store_values(self, block):

        variables = list(self._constants.keys())

        self._values = []

        for v in block:
            v = v[0].split(maxsplit=1)
            if len(v) == 0:
                return
            b = [v[0]]

            if len(v) == 2:
                b += [
                    LarkExpression(tk, variables=variables)
                    for tk in lark_tokenize(v[1])
                ]
                b[1:] = [expr.evaluate(**self._constants) for expr in b[1:]]

            b = FittingVariable(*b)

            self._values.append(b)


# Other fitting-related keywords
class KWFittingData(MuSpinExpandKeyword):

    name = "fitting_data"
    block_size = 1
    accept_range = True
    default = ""
    _constants = {}
    _functions = {"load": np.loadtxt}


class KWFittingMethod(MuSpinKeyword):

    name = "fitting_method"
    block_size = 1
    accept_range = False
    default = "nelder-mead"
    _validators = {
        "Invalid value": lambda s: (
            (s[0].lower() in ("nelder-mead", "lbfgs")) and len(s) == 1
        )
    }


class KWFittingTolerance(MuSpinKeyword):

    name = "fitting_tolerance"
    block_size = 1
    accept_range = False
    default = "1e-3"
    _validators = {"Invalid value": lambda s: (float(s[0]) and len(s) == 1)}


# Special configuration keyword
class KWExperiment(MuSpinKeyword):

    name = "experiment"
    block_size = 1
    accept_range = False
    default = ""


# Compile all KW classes into a single dictionary automatically
InputKeywords = {
    obj.name: obj
    for name, obj in inspect.getmembers(sys.modules[__name__])
    if (inspect.isclass(obj) and re.match("KW[0-9a-zA-Z]+", name))
}
