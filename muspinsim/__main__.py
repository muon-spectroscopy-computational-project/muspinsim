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


def check_dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        try:
            os.makedirs(string)
        except OSError:
            raise NotADirectoryError(string)
    return string


def main(use_mpi=False):

    if use_mpi:
        mpi.connect()

    if mpi.is_root:
        # Entry point for script
        parser = ap.ArgumentParser(
            description="Muspinsim - A program designed to carry out "
            "spin dynamics calculations for muon science experiments"
        )
        parser.add_argument(
            "-o",
            "--out-dir",
            type=str,
            default=None,
            help="""folder to store the output .dat files""",
        )
        parser.add_argument(
            "-l",
            "--log-path",
            type=str,
            default=None,
            help="""filepath to store simulation logs """,
        )
        parser.add_argument(
            "-f",
            "--fitreport-path",
            type=str,
            default=None,
            help="""filepath to store fit report if fitting parameters given""",
        )
        parser.add_argument(
            "input_file",
            type=ap.FileType("r"),
            default=None,
            help="""filepath to muspinsim specially formatted text
            file specifying simulation input parameters.""",
        )
        args = parser.parse_args()
        inp_filepath = args.input_file.name
        inp_dir = "{0}.log".format(os.path.splitext(inp_filepath)[0])

        fs = open(inp_filepath)
        infile = MuSpinInput(fs)
        is_fitting = len(infile.variables) > 0

        # Open logfile
        logfile = inp_dir
        if args.log_path:
            # check if directory exists, if not create it
            logfile = "{0}/{1}".format(
                check_dir_path(os.path.dirname(args.log_path)),
                os.path.basename(args.log_path),
            )

        logging.basicConfig(
            filename=logfile,
            filemode="w",
            level=logging.INFO,
            format=LOGFORMAT,
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        logging.info(
            "Launching MuSpinSim calculation " "from file: {0}".format(inp_filepath)
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

    if args.out_dir:
        out_path = check_dir_path(args.out_dir)
    else:
        out_path = inp_dir

    if not is_fitting:
        # No fitting
        runner = ExperimentRunner(infile, {})
        runner.run()

        if mpi.is_root:
            # Output
            runner.config.save_output(name=None, path=out_path)
    else:
        fitter = FittingRunner(infile)
        fitter.run(name=None, path=out_path)

        if mpi.is_root:
            rep_path = inp_dir
            rep_fname = None
            if args.fitreport_path:
                rep_path = check_dir_path(os.path.dirname(args.fitreport_path))
                # default to creating it with outputs
                rep_fname = os.path.basename(args.fitreport_path)

            fitter.write_report(fname=rep_fname, path=rep_path)

    if mpi.is_root:
        tend = datetime.now()
        simtime = (tend - tstart).total_seconds()
        logging.info("Simulation completed in " "{0:.3f} seconds".format(simtime))


def main_mpi():
    main(use_mpi=True)


if __name__ == "__main__":
    main()
