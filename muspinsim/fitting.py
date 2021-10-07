"""fitting.py

A class that takes care of runs where the goal is to fit some given data"""

import numpy as np
from scipy.optimize import minimize

from muspinsim.input import MuSpinInput
from muspinsim.experiment import ExperimentRunner
from muspinsim.mpi import mpi_controller as mpi


class FittingRunner(object):

    def __init__(self, inpfile: MuSpinInput):

        self._input = inpfile

        # Identify variables
        self._fitinfo = inpfile.fitting_info

        if mpi.is_root:
            self._ytarg = self._fitinfo['data'][:, 1]

            variables = inpfile.variables

            if len(variables) == 0:
                # Should never happen, but...
                raise ValueError('MuSpinInput passed to FittingRunner has no '
                                 'variables to fit')

            self._xnames = tuple(sorted(variables.keys())
                                 )  # Order is important!
            self._x = np.array([variables[n].value for n in self._xnames])
            self._xbounds = [variables[n].bounds for n in self._xnames]

            mpi.broadcast_object(self, ['_ytarg', '_x', '_xbounds', '_xnames'])
        else:
            mpi.broadcast_object(self, ['_ytarg', '_x', '_xbounds', '_xnames'])

        self._done = False
        self._sol = None

    def run(self):

        if mpi.is_root:
            self._sol = minimize(self._targfun, self._x,
                                 method=self._fitinfo['method'].lower(),
                                 tol=self._fitinfo['rtol'])

            self._done = True
            mpi.broadcast_object(self, ['_x', '_done'])

            mpi.broadcast_object(self, ['_sol'])
        else:
            while not self._done:
                self._targfun(self._x)
            # Receive solution
            mpi.broadcast_object(self, ['_sol'])

        return self._sol

    def _targfun(self, x):

        self._x = x
        # Synchronize across nodes
        mpi.broadcast_object(self, ['_x', '_done'])

        if self._done:
            # Child nodes will learn it here
            return

        vardict = dict(zip(self._xnames, self._x))
        runner = ExperimentRunner(self._input, variables=vardict)
        y = runner.run_all()

        if mpi.is_root:
            # Compare with target data
            err = np.sum((y-self._ytarg)**2)
            return err

    @property
    def solution(self):
        return self._sol
