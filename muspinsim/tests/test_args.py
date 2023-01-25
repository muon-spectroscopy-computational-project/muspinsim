import sys
import os
import unittest
import tempfile
import logging

from muspinsim.__main__ import main


def set_sys_argv(vals):
    sys.argv[1:] = vals


def run_experiment(inp_string, inp_file_path, cmd_args, use_mpi=False):
    with open(inp_file_path, "w", encoding="utf-8") as f:
        f.write(inp_string)
    set_sys_argv([inp_file_path] + cmd_args)
    main(use_mpi=use_mpi)


class TestCommandLineArgs(unittest.TestCase):
    @classmethod
    def tearDownClass(self):
        # The tests in this class setup the logger to use a temporary folder but for
        # other tests that run afterwards this may no longer exist so revert to the
        # default again and print to stdout to avoid FileNotFound errors
        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, force=True)

    def _main_default_args(self, use_mpi):
        with tempfile.TemporaryDirectory() as tmp_dir:
            cmd_args = []
            inp_file_path = os.path.join(tmp_dir, "custom_args.in")
            run_experiment(
                """
name
    test_1
spins
    mu H
zeeman 1
    1 0 0
""",
                inp_file_path,
                cmd_args,
                use_mpi=use_mpi,
            )

            self.assertTrue(os.path.exists(os.path.join(tmp_dir, "custom_args.log")))
            self.assertTrue(os.path.exists(f"{tmp_dir}/test_1.dat"))

    def test_main_default_args(self):
        self._main_default_args(False)

    def test_main_default_args_mpi(self):
        self._main_default_args(True)

    def _main_custom_args(self, use_mpi):
        with tempfile.TemporaryDirectory() as tmp_dir:
            with tempfile.TemporaryDirectory() as out_tmp_dir:
                cmd_args = [
                    "-o",
                    f"{out_tmp_dir}",
                    "-l",
                    f"{out_tmp_dir}/new_test.log",
                ]
                inp_file_path = os.path.join(tmp_dir, "custom_args.in")
                run_experiment(
                    """
name
    test_2
spins
    mu H
zeeman 1
    1 0 0
""",
                    inp_file_path,
                    cmd_args,
                    use_mpi=use_mpi,
                )

                self.assertTrue(
                    os.path.exists(os.path.join(out_tmp_dir, "new_test.log"))
                )
                self.assertTrue(os.path.exists(os.path.join(out_tmp_dir, "test_2.dat")))

    def test_main_custom_args(self):
        self._main_custom_args(False)

    def test_main_custom_args_mpi(self):
        self._main_custom_args(True)

    def _main_fitting_default_args(self, use_mpi):
        with tempfile.TemporaryDirectory() as tmp_dir:
            cmd_args = []
            inp_file_path = os.path.join(tmp_dir, "fitting_default_args.in")
            run_experiment(
                """
name
    test_fitting
spins
    mu
field
    1.0/muon_gyr
fitting_data
    0 1
    1 2
fitting_variables
    g 0.1 0.0 inf
dissipation 1
    g
""",
                inp_file_path,
                cmd_args,
                use_mpi=use_mpi,
            )

            self.assertTrue(
                os.path.exists(os.path.join(tmp_dir, "fitting_default_args.log"))
            )
            self.assertTrue(os.path.exists(os.path.join(tmp_dir, "test_fitting.dat")))
            self.assertTrue(
                os.path.exists(
                    os.path.join(tmp_dir, "fitting_default_args_fit_report.txt")
                )
            )

    def test_main_fitting_default_args(self):
        self._main_fitting_default_args(False)

    def test_main_fitting_default_args_mpi(self):
        self._main_fitting_default_args(True)

    def _main_filenames_as_input(self, use_mpi):
        with tempfile.TemporaryDirectory() as tmp_dir:
            cmd_args = ["-l", "new_test.log", "-f", "new_fit_report.txt"]
            inp_file_path = os.path.join(tmp_dir, "fitting_default_args.in")
            run_experiment(
                """
name
    test_fitting
spins
    mu
field
    1.0/muon_gyr
fitting_data
    0 1
    1 2
fitting_variables
    g 0.1 0.0 inf
dissipation 1
    g
""",
                inp_file_path,
                cmd_args,
                use_mpi=use_mpi,
            )

            self.assertTrue(os.path.exists(f"{tmp_dir}/new_test.log"))
            self.assertTrue(os.path.exists(f"{tmp_dir}/test_fitting.dat"))
            self.assertTrue(os.path.exists(f"{tmp_dir}/new_fit_report.txt"))

    def test_main_filenames_as_input(self):
        self._main_filenames_as_input(False)

    def test_main_filenames_as_input_mpi(self):
        self._main_filenames_as_input(True)

    def _main_fitting_custom_args(self, use_mpi):
        with tempfile.TemporaryDirectory() as tmp_dir:
            with tempfile.TemporaryDirectory() as out_tmp_dir:
                cmd_args = [
                    "-o",
                    f"{out_tmp_dir}",
                    "-l",
                    f"{out_tmp_dir}/new_test.log",
                    "-f",
                    f"{out_tmp_dir}/new_fit_report.txt",
                ]
                inp_file_path = os.path.join(tmp_dir, "test.in")
                run_experiment(
                    """
name
    test_fitting
spins
    mu
field
    1.0/muon_gyr
fitting_data
    0 1
    1 2
fitting_variables
    g 0.1 0.0 inf
dissipation 1
    g
""",
                    inp_file_path,
                    cmd_args,
                    use_mpi=use_mpi,
                )

                self.assertTrue(
                    os.path.exists(os.path.join(out_tmp_dir, "new_test.log"))
                )
                self.assertTrue(
                    os.path.exists(os.path.join(out_tmp_dir, "test_fitting.dat"))
                )
                self.assertTrue(
                    os.path.exists(os.path.join(out_tmp_dir, "new_fit_report.txt"))
                )

    def test_main_fitting_custom_args(self):
        self._main_fitting_custom_args(False)

    def test_main_fitting_custom_args_mpi(self):
        self._main_fitting_custom_args(True)
