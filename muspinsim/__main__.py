"""
MuSpinSim

A software to simulate the quantum dynamics of muon spin systems

Author: Simone Sturniolo

Copyright 2022 Science and Technology Facilities Council
This software is distributed under the terms of the MIT License
Please refer to the file LICENSE for the text of the license

"""

import os
import logging
import argparse as ap
from datetime import datetime

from muspinsim.mpi import mpi_controller as mpi
from muspinsim.input import MuSpinInput
from muspinsim.experiment import ExperimentRunner
from muspinsim.fitting import FittingRunner

LOGFORMAT = "[%(levelname)s] [%(threadName)s] [%(asctime)s] %(message)s"


def main(use_mpi=False):

    if use_mpi:
        mpi.connect()

    if mpi.is_root:
        # Entry point for script
        parser = ap.ArgumentParser()
        parser.add_argument(
            "input_file",
            type=str,
            default=None,
            help="""YAML
                            formatted file with input parameters.""",
        )
        args = parser.parse_args()

        fs = open(args.input_file)
        infile = MuSpinInput(fs)
        is_fitting = len(infile.variables) > 0

        # Open logfile
        logfile = "{0}.log".format(os.path.splitext(args.input_file)[0])
        logging.basicConfig(
            filename=logfile,
            filemode="w",
            level=logging.INFO,
            format=LOGFORMAT,
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        logging.info(
            "Launching MuSpinSim calculation " "from file: {0}".format(args.input_file)
        )

        if is_fitting:
            logging.info(
                "Performing fitting in variables: "
                "{0}".format(", ".join(infile.variables))
            )

        tstart = datetime.now()
    else:
        infile = MuSpinInput()
        is_fitting = False

    is_fitting = mpi.broadcast(is_fitting)

    if not is_fitting:
        # No fitting
        runner = ExperimentRunner(infile, {})
        runner.run()

        if mpi.is_root:
            # Output
            runner.config.save_output()
    else:
        fitter = FittingRunner(infile)
        fitter.run()

        if mpi.is_root:
            fitter.write_report()

    if mpi.is_root:
        tend = datetime.now()
        simtime = (tend - tstart).total_seconds()
        logging.info("Simulation completed in " "{0:.3f} seconds".format(simtime))


def main_mpi():
    main(use_mpi=True)


if __name__ == "__main__":
    main()
