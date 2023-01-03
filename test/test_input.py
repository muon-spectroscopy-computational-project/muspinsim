import unittest
import numpy as np
from io import StringIO
from tempfile import NamedTemporaryFile

from muspinsim.input.keyword import (
    MuSpinKeyword,
    MuSpinEvaluateKeyword,
    MuSpinExpandKeyword,
    MuSpinCouplingKeyword,
    InputKeywords,
)
from muspinsim.input.input import MuSpinInputError
from muspinsim.input import MuSpinInput


class TestInput(unittest.TestCase):
    def test_basic_keyword(self):
        # Basic keyword
        kw = MuSpinKeyword(["a b c", "d e f"])

        self.assertTrue((kw.evaluate() == [["a", "b", "c"], ["d", "e", "f"]]).all())
        self.assertEqual(len(kw), 2)

    def test_keyword_defaults(self):
        # Test that the default works
        class DefKeyword(MuSpinKeyword):
            default = "1"

        self.assertEqual(DefKeyword().evaluate()[0][0], "1")

    def test_no_defaults(self):
        with self.assertRaises(RuntimeError) as err:
            MuSpinKeyword()
        self.assertEqual(
            str(err.exception),
            "Input is empty and keyword 'keyword' doesn't have a default value",
        )

    def test_evaluate_keyword(self):
        nkw = MuSpinEvaluateKeyword(["exp(0) 1+1 2^2"])
        self.assertTrue((nkw.evaluate()[0] == [1, 2, 4]).all())

    def test_expand_keyword(self):
        exkw = MuSpinExpandKeyword(["range(0, 1)", "5.0 2.0"])
        self.assertTrue(len(exkw.evaluate()) == 101)

        def _repeat3(x):
            return [x, x, x]

        class RepeatKW(MuSpinExpandKeyword):
            _functions = {"repeat3": _repeat3}

        rkw = RepeatKW(["repeat3(1)"])
        self.assertTrue((rkw.evaluate()[0] == [1, 1, 1]).all())

    def test_invalid_number_of_args(self):
        with self.assertRaises(RuntimeError) as err:
            MuSpinKeyword([], args=["a"])  # One argument too much
        self.assertEqual(
            str(err.exception),
            "Wrong number of in-line arguments given 'a', expected 0, got 1",
        )

    def test_invalid_args(self):
        with self.assertRaises(RuntimeError) as err:
            MuSpinCouplingKeyword([], args=["a"])  # Wrong argument type
        self.assertEqual(
            str(err.exception),
            "Error parsing keyword argument(s) 'coupling_keyword': invalid literal for "
            "int() with base 10: 'a'",
        )

    def _eval_kw(self, test):
        kw = InputKeywords[test["kw"]](test["in"], args=test["args"])
        self.assertTrue((kw.evaluate() == test["out"]).all())

        self.assertEqual(
            kw.id, test["kw"] + "".join(["_{0}".format(i) for i in test["args"]])
        )

    def test_keyword_name(self):
        self._eval_kw(
            {
                "kw": "name",
                "args": [],
                "in": ["othername"],
                "out": "othername",
            }
        )

    def test_keyword_name_defaults(self):
        self._eval_kw(
            {
                "kw": "name",
                "args": [],
                "in": [],
                "out": "muspinsim",
            }
        )

    def test_keyword_spins(self):
        self._eval_kw(
            {
                "kw": "spins",
                "args": [],
                "in": ["F mu F"],
                "out": np.array(["F", "mu", "F"]),
            }
        )

    def test_keyword_spins_defaults(self):
        self._eval_kw(
            {
                "kw": "spins",
                "args": [],
                "in": [],
                "out": np.array(["mu", "e"]),
            }
        )

    def test_keyword_polarization(self):
        self._eval_kw(
            {
                "kw": "polarization",
                "args": [],
                "in": ["0 1 1"],
                "out": np.array([0, 1, 1]),
            }
        )

    def test_keyword_polarization_defaults(self):
        self._eval_kw(
            {
                "kw": "polarization",
                "args": [],
                "in": [],
                "out": np.array([1, 0, 0]),
            }
        )

    def test_keyword_polarization_longditudinal(self):
        self._eval_kw(
            {
                "kw": "polarization",
                "args": [],
                "in": ["longitudinal"],
                "out": np.array([0, 0, 1.0]),
            }
        )

    def test_keyword_polarization_transverse(self):
        self._eval_kw(
            {
                "kw": "polarization",
                "args": [],
                "in": ["transverse"],
                "out": np.array([1.0, 0, 0]),
            }
        )

    def test_keyword_field_range(self):
        self._eval_kw(
            {
                "kw": "field",
                "args": [],
                "in": ["range(0, 20, 21)"],
                "out": np.arange(21)[:, None],
            }
        )

    def test_keyword_field_mhz(self):
        kw = InputKeywords["field"](["500*MHz"])
        self.assertTrue(np.isclose(kw.evaluate()[0][0], 1.84449016))

    def test_keyword_field_defaults(self):
        self._eval_kw(
            {
                "kw": "field",
                "args": [],
                "in": [],
                "out": np.array([0]),
            }
        )

    def test_keyword_time_defaults(self):
        self._eval_kw(
            {
                "kw": "time",
                "args": [],
                "in": [],
                "out": np.linspace(0, 10, 101)[:, None],
            }
        )

    def test_keyword_time_range(self):
        self._eval_kw(
            {
                "kw": "time",
                "args": [],
                "in": ["range(0, 10, 5)"],
                "out": np.array([0, 2.5, 5, 7.5, 10])[:, None],
            }
        )

    def test_keyword_time_multi_line(self):
        self._eval_kw(
            {
                "kw": "time",
                "args": [],
                "in": ["10", "20", "30", "range(0, 10, 5)"],
                "out": np.array([10, 20, 30, 0, 2.5, 5, 7.5, 10])[:, None],
            }
        )

    def test_keyword_y_axis(self):
        self._eval_kw(
            {
                "kw": "y_axis",
                "args": [],
                "in": ["asymmetry"],
                "out": np.array(["asymmetry"]),
            }
        )

    def test_keyword_y_axis_invalid(self):
        with self.assertRaises(ValueError) as err:
            InputKeywords["y_axis"](
                ["something"], args=[]
            )  # Invalid value for argument
        self.assertEqual(
            str(err.exception),
            "Invalid value '['something']', accepts ['asymmetry', 'integral']",
        )

    def test_keyword_y_axis_defaults(self):
        self._eval_kw(
            {
                "kw": "y_axis",
                "args": [],
                "in": [],
                "out": np.array(["asymmetry"]),
            }
        )

    def test_keyword_orientation_zcw(self):
        kw = InputKeywords["orientation"](["zcw(20)"])
        self.assertTrue(len(kw.evaluate()) >= 20)

    def test_keyword_orientation_defaults(self):
        self._eval_kw(
            {
                "kw": "orientation",
                "args": [],
                "in": [],
                "out": np.array([0, 0, 0]),
            }
        )

    def test_keyword_zeeman(self):
        self._eval_kw(
            {
                "kw": "zeeman",
                "args": ["1"],
                "in": ["0 0 1"],
                "out": np.array([0, 0, 1]),
            }
        )

    def test_keyword_zeeman_defaults(self):
        self._eval_kw(
            {
                "kw": "zeeman",
                "args": ["1"],
                "in": [],
                "out": np.array([0, 0, 0]),
            }
        )

    def test_keyword_hyperfine(self):
        self._eval_kw(
            {
                "kw": "hyperfine",
                "args": ["1"],
                "in": ["1 0 0", "0 1 0", "0 0 1"],
                "out": np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
            }
        )

    def test_keyword_hyperfine_defaults(self):
        self._eval_kw(
            {
                "kw": "hyperfine",
                "args": ["1"],
                "in": [],
                "out": np.zeros((3, 3)),
            }
        )

    def test_keyword_dipolar(self):
        self._eval_kw(
            {
                "kw": "dipolar",
                "args": ["1", "2"],
                "in": ["0 0 1"],
                "out": np.array([0, 0, 1]),
            }
        )

    def test_keyword_dipolar_defaults(self):
        self._eval_kw(
            {
                "kw": "dipolar",
                "args": ["1", "2"],
                "in": [],
                "out": np.array([0, 0, 0]),
            }
        )

    def test_keyword_quadrupolar(self):
        kw = InputKeywords["quadrupolar"](
            ["1 0 0", "0 1 0", "0 0 cos(1)^2+sin(1)^2"], args=["1"]
        )
        self.assertTrue(np.allclose(kw.evaluate()[0], np.eye(3)))
        self.assertTrue(kw.id, "quadrupolar_1")

    def test_keyword_quadrupolar_defaults(self):

        self._eval_kw(
            {
                "kw": "zeeman",
                "args": ["1"],
                "in": [],
                "out": np.array([0, 0, 0]),
            }
        )

    def test_input_valid(self):
        # read valid file
        e1 = MuSpinInput(
            StringIO(
                """
name
    test_1
spins
    mu H
zeeman 1
    1 0 0
"""
            )
        ).evaluate()

        self.assertEqual(e1["name"].value[0], "test_1")
        self.assertTrue((e1["spins"].value[0] == ["mu", "H"]).all())
        self.assertTrue((e1["couplings"]["zeeman_1"].value[0] == [1, 0, 0]).all())

    def test_input_invalid_formatting(self):
        # read improperly formatted file
        with self.assertRaises(RuntimeError) as err:
            MuSpinInput(
                StringIO(
                    """
    name
        test_1
spins
    mu H
zeeman 1
    1 0 0
"""
                )
            ).evaluate()
        self.assertEqual(str(err.exception), "Badly formatted input file")

    def test_input_invalid_indent_inside_block(self):
        # indent in between keyword values does not match
        with self.assertRaises(RuntimeError) as err:
            MuSpinInput(
                StringIO(
                    """
name
    test_1
spins
    mu H
hyperfine 1
    1 0 0
     2 0 0
    3 0 0
"""
                )
            ).evaluate()
        self.assertEqual(str(err.exception), "Invalid indent in input file")

    def test_input_invalid_keyword(self):
        # incorrect keyword name given
        with self.assertRaises(MuSpinInputError) as err:
            MuSpinInput(
                StringIO(
                    """
name
    test_1
spins
    mu H
notakeyword 1
    1 0 0
"""
                )
            ).evaluate()
        self.assertEqual(
            str(err.exception),
            "Found 1 Error(s) whilst trying to parse keywords: \n\n"
            "Error occurred when parsing keyword 'notakeyword' "
            "(block starting at line 6):\n"
            "Invalid keyword notakeyword found in input file",
        )

    def test_input_fitting(self):
        # Test input focused around fitting

        i1 = MuSpinInput(
            StringIO(
                """
fitting_variables
    x 1.0 0.0 2.0
fitting_data
    0  0.0
    1  1.0
    2  4.0
    3  9.0
field
    2*x
zeeman 1
    x x 0
"""
            )
        )

        self.assertTrue(i1.fitting_info["fit"])

        data = i1.fitting_info["data"]
        self.assertTrue((data == [[0, 0], [1, 1], [2, 4], [3, 9]]).all())

        e1 = i1.evaluate(x=2.0)
        self.assertEqual(e1["field"].value[0][0], 4.0)
        self.assertTrue((e1["couplings"]["zeeman_1"].value[0] == [2, 2, 0]).all())

        variables = i1.variables

        self.assertEqual(variables["x"].value, 1.0)
        self.assertEqual(variables["x"].bounds, (0.0, 2.0))

    def _write_temp_file(self, tdata):
        tfile = NamedTemporaryFile(mode="w", delete=False)

        for d in tdata:
            tfile.write("{0} {1}\n".format(*d))
        tfile.flush()
        tfile.close()
        return tfile

    def test_load_fitting_data_from_file(self):
        tdata = np.zeros((10, 2))
        tdata[:, 0] = np.linspace(0, 1, 10)
        tdata[:, 1] = tdata[:, 0] ** 2

        tfile = self._write_temp_file(tdata)

        i2 = MuSpinInput(
            StringIO(
                """
fitting_variables
    x
fitting_data
    load("{fname}")
fitting_method
    nelder-mead
""".format(
                    fname=tfile.name
                )
            )
        )

        finfo = i2.fitting_info

        data = finfo["data"]
        self.assertTrue(finfo["fit"])
        self.assertTrue((data == tdata).all())
        self.assertEqual(finfo["method"], "nelder-mead")
        self.assertAlmostEqual(finfo["rtol"], 1e-3)

    def test_fitting_no_data(self):
        # invalid no data given
        with self.assertRaises(MuSpinInputError) as err:
            MuSpinInput(
                StringIO(
                    """
fitting_variables
    x 1.0 0.0 5.0
"""
                )
            )
        self.assertEqual(
            str(err.exception),
            "Found 1 Error(s) whilst trying to parse fitting keywords: \n\n"
            "Error occurred when parsing keyword 'fitting_variables' "
            "(block starting at line 2):\n"
            "Fitting variables defined without defining any data to fit",
        )

    def test_fitting_invalid_variable_ranges(self):
        tdata = np.zeros((10, 2))
        tdata[:, 0] = np.linspace(0, 1, 10)
        tdata[:, 1] = tdata[:, 0] ** 2
        tfile = self._write_temp_file(tdata)

        with self.assertRaises(MuSpinInputError) as err:
            MuSpinInput(
                StringIO(
                    """
fitting_variables
    x 1.0 0.0 -5.0
fitting_data
    load("{fname}")
fitting_method
    nelder-mead
field
    2*x
zeeman 1
    x x 0
""".format(
                        fname=tfile.name
                    )
                )
            )
        self.assertTrue(
            "Variable x has invalid range: "
            "(max value -5.0 cannot be less than or equal to min value 0.0)\n"
            "Variable x has invalid starting value: "
            "(starting value 1.0 cannot be greater than max value -5.0)"
            in str(err.exception)
        )

    def test_fitting_name_clash(self):
        # variable name clashes with constant
        with self.assertRaises(MuSpinInputError) as err:
            MuSpinInput(
                StringIO(
                    """
fitting_variables
    MHz 1.0 0.0 5.0
fitting_data
    0  0.0
    1  1.0
    2  4.0
    3  9.0
field
    2*x
zeeman 1
    MHz 0 0
"""
                )
            )
        self.assertEqual(
            str(err.exception),
            "Found 1 Error(s) whilst trying to parse fitting keywords: \n\n"
            "Error occurred when parsing keyword 'fitting_variables' "
            "(block starting at line 2):\n"
            "Invalid value 'MHz': variable name conflicts with a constant",
        )
