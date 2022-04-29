"""input.py

Class to read in input files for the muspinsim script
"""

import re
from io import StringIO

import numpy as np
from collections import namedtuple

from muspinsim.input.keyword import (
    InputKeywords,
    MuSpinEvaluateKeyword,
    MuSpinCouplingKeyword,
)


class MuSpinInputError(Exception):
    pass


MuSpinInputValue = namedtuple("MuSpinInputValue", ["name", "args", "value"])

# Experiment defaults as .in files
_exp_defaults = {
    "alc": """
polarization
    longitudinal
y_axis
    integral
x_axis
    field
""",
    "zero_field": """
field
    0.0
polarization
    transverse
x_axis
    time
y_axis
    asymmetry
""",
}


class MuSpinInput(object):
    def __init__(self, fs=None):
        """Read in an input file

        Read in an input file from an opened file stream

        Arguments:
            fs {TextIOBase} -- I/O stream (should be file, can be StringIO)
        """

        self._keywords = {}
        self._variables = {}
        self._fitting_info = {"fit": False, "data": None, "method": None, "rtol": None}

        if fs is not None:

            lines = fs.readlines()

            # Split lines in blocks
            raw_blocks = {}
            curr_block = None

            indre = re.compile("(\\s+)[^\\s]")
            indent = None

            for l in lines:

                # Remove any comments
                l = l.split("#", 1)[0]

                if l.strip() == "":
                    continue  # It's a comment
                m = indre.match(l)
                if m:
                    if indent is None:
                        indent = m.groups()[0]
                    if m.groups()[0] != indent:
                        raise RuntimeError("Invalid indent in input file")
                    else:
                        try:
                            raw_blocks[curr_block].append(l.strip())
                        except KeyError:
                            raise RuntimeError("Badly formatted input file")
                else:
                    curr_block = l.strip()
                    raw_blocks[curr_block] = []
                    indent = None  # Reset for each block

            # A special case: if there are fitting variables, we need to know
            # right away
            self._load_fitting_kw(raw_blocks)

            # Another special case: if the "experiment" keyword is present,
            # use it to set some defaults
            try:
                block = raw_blocks.pop("experiment")
                kw = InputKeywords["experiment"](block)
                exptype = kw.evaluate()[0]
                if len(exptype) > 1:
                    raise MuSpinInputError(
                        "Can not define more than one experiment type"
                    )
                elif len(exptype) == 1:
                    try:
                        mock_i = MuSpinInput(StringIO(_exp_defaults[exptype[0]]))
                        self._keywords.update(mock_i._keywords)
                    except KeyError:
                        raise MuSpinInputError("Invalid experiment type defined")
            except KeyError:
                pass

            # Now parse
            for header, block in raw_blocks.items():

                hsplit = header.split()
                name = hsplit[0]
                args = hsplit[1:]

                try:
                    KWClass = InputKeywords[name]
                except KeyError:
                    raise MuSpinInputError(
                        "Invalid keyword " "{0} ".format(name) + "found in input file"
                    )

                if issubclass(KWClass, MuSpinEvaluateKeyword):
                    kw = KWClass(block, args=args, variables=self._variables)
                else:
                    kw = KWClass(block, args=args)

                kwid = kw.id

                if name != kwid:
                    self._keywords[name] = self._keywords.get(name, {})
                    self._keywords[name][kwid] = kw
                else:
                    self._keywords[name] = kw

    @property
    def variables(self):
        return {**self._variables}

    @property
    def fitting_info(self):
        return {**self._fitting_info}

    def evaluate(self, **variables):
        """Produce a full dictionary with a value for every input keyword,
        interpreted given the variable values that have been passed."""

        result = {"couplings": {}, "fitting_info": self.fitting_info}

        for name, KWClass in InputKeywords.items():

            if issubclass(KWClass, MuSpinCouplingKeyword):
                if name in self._keywords:
                    for kwid, kw in self._keywords[name].items():
                        val = MuSpinInputValue(
                            name, kw.arguments, kw.evaluate(**variables)
                        )
                        result["couplings"][kwid] = val
            else:
                if name in self._keywords:
                    kw = self._keywords[name]
                    v = variables if issubclass(KWClass, MuSpinEvaluateKeyword) else {}
                    val = kw.evaluate(**v)

                    result[name] = MuSpinInputValue(name, kw.arguments, val)

                elif KWClass.default is not None:
                    kw = KWClass()
                    val = np.array(kw.evaluate())

                    result[name] = MuSpinInputValue(name, kw.arguments, val)

        return result

    def _load_fitting_kw(self, raw_blocks):
        """Special case: handling of all the fitting related keywords and
        information."""

        try:
            block = raw_blocks.pop("fitting_variables")
            kw = InputKeywords["fitting_variables"](block)
            self._variables = {v.name: v for v in kw.evaluate()}
        except KeyError:
            pass

        if len(self._variables) == 0:
            return

        self._fitting_info["fit"] = True

        try:
            block = raw_blocks.pop("fitting_data")
            kw = InputKeywords["fitting_data"](block)
            self._fitting_info["data"] = np.array(kw.evaluate())
        except KeyError:
            raise MuSpinInputError(
                "Fitting variables defined without defining" " a set of data to fit"
            )

        block = raw_blocks.pop("fitting_tolerance", [])
        kw = InputKeywords["fitting_tolerance"](block)
        self._fitting_info["rtol"] = float(kw.evaluate()[0][0])

        block = raw_blocks.pop("fitting_method", [])
        kw = InputKeywords["fitting_method"](block)
        self._fitting_info["method"] = kw.evaluate()[0][0]
