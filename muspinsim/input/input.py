"""input.py

Class to read in input files for the muspinsim script
"""

import re
from io import StringIO
from collections import namedtuple

import numpy as np

from muspinsim.input.keyword import (
    InputKeywords,
    MuSpinEvaluateKeyword,
    MuSpinCouplingKeyword,
)

from muspinsim.input.larkeval import LarkExpression, LarkExpressionError


class MuSpinInputError(Exception):
    """Custom exception for MuSpinSim input parsing"""


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


def write_error(keyword, block_line_num, err):
    return (
        f"Error occurred when parsing keyword '{keyword}' "
        f"(block starting at line {block_line_num}):\n{str(err)}"
    )


def _make_blocks(file_stream):
    lines = file_stream.readlines()

    # Split lines in blocks
    raw_blocks = {}
    curr_block = None
    block_line_nums = {}

    indre = re.compile("(\\s+)[^\\s]")
    indent = None

    for i, l in enumerate(lines):

        # Remove any comments
        l = l.split("#", 1)[0]

        if l.strip() == "":
            continue  # It's a comment
        m = indre.match(l)
        if m:
            if indent is None:
                indent = m.groups()[0]
            if m.groups()[0] != indent:
                raise MuSpinInputError("Invalid indent in input file")
            else:
                try:
                    raw_blocks[curr_block].append(l.strip())
                except KeyError as exc:
                    raise MuSpinInputError("Badly formatted input file") from exc
        else:
            curr_block = l.strip()

            # Check if the block has already been defined before
            if curr_block in raw_blocks:
                raise MuSpinInputError(
                    f"Redefinition of '{curr_block}' found in input file"
                )

            raw_blocks[curr_block] = []
            block_line_nums[curr_block] = i + 1
            indent = None  # Reset for each block

    return raw_blocks, block_line_nums


class MuSpinInput:
    def __init__(self, file_stream=None):
        """Read in an input file

        Read in an input file from an opened file stream

        Arguments:
            file_stream {TextIOBase} -- I/O stream (should be file, can be StringIO)
        """

        self._keywords = {}
        self._variables = {}
        self._fitting_info = {
            "fit": False,
            "data": None,
            "method": None,
            "rtol": None,
            "function": None,
            # When true indicates all fitting can be done after the simulation
            "single_simulation": True,
        }

        if file_stream is not None:

            raw_blocks, block_line_nums = _make_blocks(file_stream)

            # if we find errors when parsing fitting variables, we post an error
            # so we don't propagate invalid variables when parsing keywords later
            failed_status, errors = self._load_fitting_kw(raw_blocks, block_line_nums)
            if failed_status:
                raise MuSpinInputError(
                    "Found {0} Error(s) whilst trying to parse fitting keywords: "
                    "\n\n{1}".format(len(errors), "\n\n".join(errors))
                )

            # Another special case: if the "experiment" keyword is present,
            # use it to set some defaults
            errors_found = []
            try:
                block = raw_blocks.pop("experiment")
                kw = InputKeywords["experiment"](block)
                exptype = kw.evaluate()[0]
                try:
                    exp_kw, _ = _make_blocks(StringIO(_exp_defaults[exptype[0]]))
                    raw_blocks.update(exp_kw)
                except KeyError:
                    err = (
                        f"Invalid experiment type '{exptype[0]}' defined, "
                        f"possible types include {list(_exp_defaults.keys())}"
                    )
                    errors_found += [
                        write_error("experiment", block_line_nums["experiment"], err)
                    ]
            except RuntimeError as exc:
                errors_found += [
                    write_error("experiment", block_line_nums["experiment"], str(exc))
                ]
            except KeyError:
                pass

            # Now parse
            for header, block in raw_blocks.items():
                hsplit = header.split()
                name = hsplit[0]
                args = hsplit[1:]
                try:
                    try:
                        KWClass = InputKeywords[name]
                    except KeyError as exc:
                        raise RuntimeError(
                            f"Invalid keyword '{name}' found in input file"
                        ) from exc
                    if issubclass(KWClass, MuSpinEvaluateKeyword):
                        kw = KWClass(block, args=args, variables=self._variables)

                        # In cases where fitting parameters are only used
                        # as variables for 'results_function' and no where
                        # else we only need to run the simulation once
                        if self._fitting_info["single_simulation"]:
                            # Flatten here as may have array of arrays e.g.
                            # hyperfine
                            for value in np.array(kw._values[0]).flatten():
                                if name != "results_function" and isinstance(
                                    value, LarkExpression
                                ):
                                    if np.any(
                                        np.in1d(
                                            list(self._variables.keys()),
                                            list(value._variables),
                                        )
                                    ):
                                        self._fitting_info["single_simulation"] = False
                    else:
                        kw = KWClass(block, args=args)

                    kwid = kw.id

                    if name != kwid:
                        self._keywords[name] = self._keywords.get(name, {})
                        self._keywords[name][kwid] = kw
                    else:
                        self._keywords[name] = kw

                except (ValueError, LarkExpressionError, RuntimeError) as exc:
                    errors_found += [
                        write_error(name, block_line_nums[header], str(exc))
                    ]

            if errors_found:
                raise MuSpinInputError(
                    "Found {0} error(s) whilst trying to parse keywords: "
                    "\n\n{1}".format(len(errors_found), "\n\n".join(errors_found))
                )

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
                # remove unnecessary keywords - stored in fitting_info
                if name in [
                    "fitting_data",
                    "fitting_tolerance",
                    "fitting_variables",
                    "fitting_method",
                ]:
                    pass
                # Special case where we don't want to evaluate the expression
                # yet
                elif name in ["results_function"]:
                    if name in self._keywords:
                        result[name] = self._keywords[name]
                    else:
                        # Default
                        result[name] = KWClass(variables=variables)
                elif name in self._keywords:
                    kw = self._keywords[name]
                    v = variables if issubclass(KWClass, MuSpinEvaluateKeyword) else {}
                    val = kw.evaluate(**v)

                    result[name] = MuSpinInputValue(name, kw.arguments, val)

                elif KWClass.default is not None:
                    kw = KWClass()
                    val = np.array(kw.evaluate())

                    result[name] = MuSpinInputValue(name, kw.arguments, val)

        return result

    def _load_fitting_kw(self, raw_blocks, block_line_nums):
        """Special case: handling of all the fitting related keywords and
        information."""
        errors_found = []
        try:
            block = raw_blocks.pop("fitting_variables")
            kw = InputKeywords["fitting_variables"](block)
            self._variables = {v.name: v for v in kw.evaluate()}
        except KeyError:
            pass
        except (RuntimeError, ValueError, LarkExpressionError) as exc:
            errors_found += [
                write_error(
                    "fitting_variables", block_line_nums["fitting_variables"], str(exc)
                )
            ]

        if errors_found:
            return 1, errors_found

        if len(self._variables) == 0:
            return 0, None

        self._fitting_info["fit"] = True

        try:
            block = raw_blocks.pop("fitting_data")
            kw = InputKeywords["fitting_data"](block)
            self._fitting_info["data"] = np.array(kw.evaluate())
        except KeyError:
            errors_found += [
                write_error(
                    "fitting_variables",
                    block_line_nums["fitting_variables"],
                    "Fitting variables defined without defining any data to fit",
                )
            ]
        except (RuntimeError, ValueError, LarkExpressionError, IOError) as exc:
            errors_found += [
                write_error("fitting_data", block_line_nums["fitting_data"], str(exc))
            ]

        try:
            block = raw_blocks.pop("fitting_tolerance", [])
            kw = InputKeywords["fitting_tolerance"](block)
            self._fitting_info["rtol"] = float(kw.evaluate()[0][0])
        except (RuntimeError, ValueError) as exc:
            errors_found += [
                write_error(
                    "fitting_tolerance", block_line_nums["fitting_tolerance"], str(exc)
                )
            ]

        try:
            block = raw_blocks.pop("fitting_method", [])
            kw = InputKeywords["fitting_method"](block)
            self._fitting_info["method"] = kw.evaluate()[0][0]
        except (RuntimeError, ValueError) as exc:
            errors_found += [
                write_error(
                    "fitting_method", block_line_nums["fitting_method"], str(exc)
                )
            ]

        if errors_found:
            return 1, errors_found
        return 0, None
