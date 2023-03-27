"""fitting.py

A class that takes care of runs where the goal is to fit some given data"""

import os
import numpy as np
from scipy.optimize import minimize, least_squares

from muspinsim.input import MuSpinInput
from muspinsim.simconfig import MuSpinConfig
from muspinsim.experiment import ExperimentRunner
from muspinsim.mpi import mpi_controller as mpi


class FittingRunner:
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
        self._fitinfo["method"] = self._fitinfo["method"].lower()

        if mpi.is_root:
            self._ytarg = self._fitinfo["data"][:, 1]

            variables = inpfile.variables

            if len(variables) == 0:
                # Should never happen, but...
                raise ValueError(
                    "MuSpinInput passed to FittingRunner has no variables to fit"
                )

            self._xnames = tuple(sorted(variables.keys()))  # Order is important!
            self._x = np.array([variables[n].value for n in self._xnames])
            self._xbounds = [variables[n].bounds for n in self._xnames]

            mpi.broadcast_object(self, ["_ytarg", "_x", "_xbounds", "_xnames"])
        else:
            mpi.broadcast_object(self, ["_ytarg", "_x", "_xbounds", "_xnames"])

        self._done = False
        self._sol = None
        self._cached_results = None

    def run(self, name=None, path="."):
        """Run a fitting calculation using Scipy, and returns the solution

        Returns:
            sol {scipy.OptimizeResult} -- The result of the optimisation.
        """

        if mpi.is_root:

            # Get the correct string for the method
            method = {
                "nelder-mead": "nelder-mead",
                "lbfgs": "L-BFGS-B",
                "least-squares": "trf",
            }[self._fitinfo["method"]]

            # SciPy minimise
            if self._fitinfo["method"] == "least-squares":
                # Bounds used here are different to minimize, need all lower
                # value and all upper values in separate arrays

                # lbfgs uses gtol when using tol, so will do the same here
                xbounds = np.array(self._xbounds)
                self._sol = least_squares(
                    self._targfun,
                    self._x,
                    method="trf",
                    gtol=self._fitinfo["rtol"],
                    bounds=(xbounds[:, 0], xbounds[:, 1]),
                )
            else:
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

            # As we are fitting, the config will only contain the results
            # prior to the applying results_function, so update the actual
            # values now
            self._runner.config.results = self._runner.apply_results_function(
                self._runner.config.results, dict(zip(self._xnames, self._sol["x"]))
            )

            # And now save the last result
            self._runner.config.save_output(name=name, path=path)
        else:
            while not self._done:
                self._targfun(self._x)
            # Receive solution
            mpi.broadcast_object(self, ["_sol"])

        return self._sol

    def _obtain_results(self, vardict: dict):
        # Obtain cached results if available
        if self._fitinfo["single_simulation"]:
            if self._cached_results is None:
                self._runner = ExperimentRunner(self._input, variables=vardict)
                self._cached_results = self._runner.run()
            return self._cached_results
        else:
            self._runner = ExperimentRunner(self._input, variables=vardict)
            return self._runner.run()

    def _targfun(self, x):

        self._x = x
        # Synchronize across nodes
        mpi.broadcast_object(self, ["_x", "_done"])

        if self._done:
            # Child nodes will learn it here
            return

        vardict = dict(zip(self._xnames, self._x))
        y = self._obtain_results(vardict)

        # Apply the results function
        y = self._runner.apply_results_function(y, vardict)

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
                fname = config.name + "_fit_report.txt"

            with open(os.path.join(path, fname), "w", encoding="utf-8") as file:
                file.write(f"Fitting process for {config.name} completed\n")
                file.write(f"Success achieved: {self._sol['success']}\n")
                if not self._sol["success"]:
                    file.write(f"   Message: {self._sol['message']}\n")

                file.write(f"Final absolute error <|f-f_targ|>: {self._sol['fun']}\n")
                num_simulations = self._sol["nfev"]
                if self._fitinfo["single_simulation"]:
                    num_simulations = 1
                    file.write(
                        "Number of 'results_function' evaluations: "
                        f"{self._sol['nfev']}\n"
                    )
                file.write(f"Number of simulations: {num_simulations}\n")

                # Not relevant when using least-squares
                if self._fitinfo["method"] != "least-squares":
                    file.write(f"Number of iterations: {self._sol['nit']}\n")

                file.write("\n" + "=" * 20 + "\n")
                file.write("\nValues found for fitting variables:\n\n")
                for name, val in variables.items():
                    file.write(f"\t{name} = {val}\n")

    @property
    def solution(self):
        return self._sol
