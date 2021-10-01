import os
import logging
import numpy as np
import argparse as ap
from datetime import datetime

from muspinsim.mpi import mpi_controller as mpi
from muspinsim.input import MuSpinInput
from muspinsim.simconfig import MuSpinConfig
from muspinsim.experiment import run_configuration

def main(use_mpi=False):

    if use_mpi:
        mpi.connect()

    if mpi.is_root:
        # Entry point for script
        parser = ap.ArgumentParser()
        parser.add_argument('input_file', type=str, default=None, help="""YAML
                            formatted file with input parameters.""")
        args = parser.parse_args()

        fs = open(args.input_file)
        infile = MuSpinInput(fs)
        is_fitting = len(infile.variables) > 0
    else:
        infile = MuSpinInput()
        is_fitting = False

    is_fitting = mpi.comm.bcast(is_fitting)

    if not is_fitting:
        # No fitting
        results = run_configuration(infile, {})

        if mpi.is_root:
            # Output
            print(results)

    else:
        raise NotImplementedError('Fitting still not implemented')

def main_mpi():
    main(use_mpi=True)


if __name__ == '__main__':
    main()
