import sys
import os
import unittest
import tempfile
import logging

from muspinsim.__main__ import main


def set_sys_argv(vals):
    sys.argv[1:] = vals


def run_experiment(inp_string, inp_file_path, cmd_args):
    with open(inp_file_path, "w") as f:
        f.write(inp_string)
    set_sys_argv([inp_file_path] + cmd_args)
    main()


class TestCommandLineArgs(unittest.TestCase):
    @classmethod
    def tearDownClass(self):
        # The tests in this class setup the logger to use a temporary folder but for
        # other tests that run afterwards this may no longer exist so revert to the
        # default again and print to stdout to avoid FileNotFound errors
        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, force=True)

    def test_main_default_args(self):
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
            )

            self.assertTrue(os.path.exists(os.path.join(tmp_dir, "custom_args.log")))
            self.assertTrue(os.path.exists("{0}/test_1.dat".format(tmp_dir)))

    def test_main_custom_args(self):
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
                )

                self.assertTrue(
                    os.path.exists(os.path.join(out_tmp_dir, "new_test.log"))
                )
                self.assertTrue(os.path.exists(os.path.join(out_tmp_dir, "test_2.dat")))

    def test_main_fitting_default_args(self):
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

    def test_main_filenames_as_input(self):
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
            )

            self.assertTrue(os.path.exists("{0}/new_test.log".format(tmp_dir)))
            self.assertTrue(os.path.exists("{0}/test_fitting.dat".format(tmp_dir)))
            self.assertTrue(os.path.exists("{0}/new_fit_report.txt".format(tmp_dir)))

    def test_main_fitting_custom_args(self):
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
