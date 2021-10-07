"""fitting.py

A class that takes care of runs where the goal is to fit some given data"""

import os
import numpy as np
from scipy.optimize import minimize

from muspinsim.input import MuSpinInput
from muspinsim.simconfig import MuSpinConfig
from muspinsim.experiment import ExperimentRunner
from muspinsim.mpi import mpi_controller as mpi


class FittingRunner(object):
    def __init__(self, inpfile: MuSpinInput):
        """Initialise a FittingRunner object

        Initialise an object to run a parallelised fitting calculation.
        The object only needs a MuSpinInput object defining the input file,
        and only on the root node it's important that this object has the file
        contents loaded. Everything else will be synchronised by broadcasting.

        Arguments:
            inpfile {MuSpinInput} -- Input file contents

        """

        self._input = inpfile
        self._runner = None

        # Identify variables
        self._fitinfo = inpfile.fitting_info

        if mpi.is_root:
            self._ytarg = self._fitinfo["data"][:, 1]

            variables = inpfile.variables

            if len(variables) == 0:
                # Should never happen, but...
                raise ValueError(
                    "MuSpinInput passed to FittingRunner has no " "variables to fit"
                )

            self._xnames = tuple(sorted(variables.keys()))  # Order is important!
            self._x = np.array([variables[n].value for n in self._xnames])
            self._xbounds = [variables[n].bounds for n in self._xnames]

            mpi.broadcast_object(self, ["_ytarg", "_x", "_xbounds", "_xnames"])
        else:
            mpi.broadcast_object(self, ["_ytarg", "_x", "_xbounds", "_xnames"])

        self._done = False
        self._sol = None

    def run(self):
        """Run a fitting calculation using Scipy, and returns the solution

        Returns:
            sol {scipy.OptimizeResult} -- The result of the optimisation.
        """

        if mpi.is_root:

            # Get the correct string for the method
            method = {"nelder-mead": "nelder-mead", "lbfgs": "L-BFGS-B"}[
                self._fitinfo["method"].lower()
            ]

            self._sol = minimize(
                self._targfun,
                self._x,
                method=method,
                tol=self._fitinfo["rtol"],
                bounds=self._xbounds,
            )

            self._done = True
            mpi.broadcast_object(self, ["_x", "_done"])

            mpi.broadcast_object(self, ["_sol"])

            # And now save the last result
            self._runner.config.save_output()
        else:
            while not self._done:
                self._targfun(self._x)
            # Receive solution
            mpi.broadcast_object(self, ["_sol"])

        return self._sol

    def _targfun(self, x):

        self._x = x
        # Synchronize across nodes
        mpi.broadcast_object(self, ["_x", "_done"])

        if self._done:
            # Child nodes will learn it here
            return

        vardict = dict(zip(self._xnames, self._x))
        self._runner = ExperimentRunner(self._input, variables=vardict)
        y = self._runner.run()

        if mpi.is_root:
            # Compare with target data
            err = np.average(np.abs(y - self._ytarg))
            return err

    def write_report(self, fname=None, path="./"):
        """Write a report file with the contents of the fitting optimisation.

        Write a human readable report summing up the results of the
        optimisation, including values found for the variables and tolerance
        achieved.

        Keyword Arguments:
            fname {str} -- Name to give to the report. If None, use the input
                           file's given name as base (default: None)
            path {str} -- Path at which to write the report (default: ./)
        """

        if mpi.is_root:

            # Final variable values
            variables = dict(zip(self._xnames, self._sol["x"]))
            config = MuSpinConfig(self._input.evaluate(**variables))

            if fname is None:
                fname = config.name + "_fitreport.txt"

            with open(os.path.join(path, fname), "w") as f:
                f.write("Fitting process for {0} completed\n".format(config.name))
                f.write("Success achieved: {0}\n".format(self._sol["success"]))
                if not self._sol["success"]:
                    f.write("   Message: {0}\n".format(self._sol["message"]))

                f.write(
                    "Final absolute error <|f-f_targ|>: "
                    "{0}\n".format(self._sol["fun"])
                )
                f.write("Number of simulations: " "{0}\n".format(self._sol["nfev"]))
                f.write("Number of iterations: {0}\n".format(self._sol["nit"]))

                f.write("\n" + "=" * 20 + "\n")
                f.write("\nValues found for fitting variables:\n\n")
                for name, val in variables.items():
                    f.write("\t{0} = {1}\n".format(name, val))

    @property
    def solution(self):
        return self._sol
