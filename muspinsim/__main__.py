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

LOG_FORMAT = "[%(levelname)s] [%(threadName)s] [%(asctime)s] %(message)s"


def ensure_dir_path_exists(path_string):
    """Checks if a directory exists, and if not attempts to create it"""
    if not os.path.isdir(path_string):
        try:
            os.makedirs(path_string)
        except OSError as exc:
            raise NotADirectoryError(path_string) from exc
    return path_string


def main(use_mpi=False):
    """Entrypoint for MuSpinSim"""

    if use_mpi:
        mpi.connect()

    if mpi.is_root:
        # Entry point for script
        parser = ap.ArgumentParser(
            description="MuSpinSim - A program designed to carry out "
            "spin dynamics calculations for muon science experiments"
        )
        parser.add_argument(
            "-f",
            "--fit-report-path",
            type=str,
            default=None,
            help="Filepath to store fit report if fitting parameters given",
        )
        parser.add_argument(
            "-l",
            "--log-path",
            type=str,
            default=None,
            help="Filepath to store simulation logs",
        )
        parser.add_argument(
            "-o",
            "--out-dir",
            type=str,
            default=None,
            help="Folder to store the output .dat files",
        )
        parser.add_argument(
            "input_file",
            type=ap.FileType("r"),
            default=None,
            help="Path to file specifying simulation input parameters. "
            "For formatting and keywords, see https://"
            "muon-spectroscopy-computational-project.github.io"
            "/muspinsim/input/.",
        )
        args = parser.parse_args()
        inp_filepath = args.input_file.name

        # get input file name and directory - to be used as defaults
        inp_file_name = os.path.basename(inp_filepath).split(".")[0]
        inp_dir = os.path.dirname(inp_filepath)

        with open(inp_filepath, encoding="utf-8") as file:
            in_file = MuSpinInput(file)
        is_fitting = len(in_file.variables) > 0

        out_path = inp_dir
        if args.out_dir:
            out_path = ensure_dir_path_exists(args.out_dir)

        # Open logfile
        logfile = os.path.join(inp_dir, inp_file_name + ".log")
        if args.log_path:
            try:
                # check if directory exists, if not create it
                log_path = ensure_dir_path_exists(os.path.dirname(args.log_path))
                log_fname = os.path.basename(args.log_path)
            except NotADirectoryError:
                # if log_path was just a filename use inp_dir as a default
                log_path = inp_dir
                log_fname = args.log_path
            logfile = os.path.join(log_path, log_fname)

        logging.basicConfig(
            filename=logfile,
            filemode="w",
            level=logging.INFO,
            format=LOG_FORMAT,
            datefmt="%Y-%m-%d %H:%M:%S",
            force=True,
        )

        logging.info("Launching MuSpinSim calculation from file: %s", inp_filepath)

        if is_fitting:
            logging.info(
                "Performing fitting in variables: %s",
                ", ".join(in_file.variables),
            )

        t_start = datetime.now()
    else:
        in_file = MuSpinInput()
        is_fitting = False

    is_fitting = mpi.broadcast(is_fitting)

    if not is_fitting:
        # No fitting
        runner = ExperimentRunner(in_file, {})
        runner.run()

        if mpi.is_root:
            # Output
            runner.config.save_output(name=None, path=out_path)
    else:
        fitter = FittingRunner(in_file)
        fitter.run(name=None, path=out_path)

        if mpi.is_root:
            rep_dname = inp_dir
            rep_fname = f"{inp_file_name}_fit_report.txt"
            if args.fit_report_path:
                try:
                    # check if directory exists, if not create it
                    ensure_dir_path_exists(os.path.dirname(args.fit_report_path))
                    rep_fname = os.path.basename(args.fit_report_path)
                    rep_dname = os.path.dirname(args.fit_report_path)
                except NotADirectoryError:
                    # if fit_report_path was just a filename use inp_dir as a default
                    rep_fname = args.fit_report_path
                    rep_dname = inp_dir

            fitter.write_report(fname=rep_fname, path=rep_dname)

    if mpi.is_root:
        t_end = datetime.now()
        sim_time = (t_end - t_start).total_seconds()
        logging.info("Simulation completed in %.3f seconds", sim_time)

    logging.shutdown()


def main_mpi():
    """Entrypoint for MPI"""
    main(use_mpi=True)


if __name__ == "__main__":
    main()
