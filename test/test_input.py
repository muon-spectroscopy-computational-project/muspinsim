import unittest
import numpy as np
from io import StringIO
from tempfile import NamedTemporaryFile

from muspinsim.input.larkeval import LarkExpression, LarkExpressionError, lark_tokenize
from muspinsim.input.keyword import (
    MuSpinKeyword,
    MuSpinEvaluateKeyword,
    MuSpinExpandKeyword,
    InputKeywords,
)
from muspinsim.input.input import MuSpinInputError
from muspinsim.input import MuSpinInput


class TestInput(unittest.TestCase):
    def test_larkexpr(self):
        """Lark expression unit test"""

        def double(x):
            return 2 * x

        larktests = [
            # read expression
            {"larkexpr": "3+2*6^2/9", "funcs": {}, "invalid": False, "out": 11},
            # read function - double
            {
                "larkexpr": "double(2)",
                "funcs": {"double": double},
                "invalid": False,
                "out": 4,
            },
            # read expression within function
            {
                "larkexpr": "double(3+2*6^2/9)",
                "funcs": {"double": double},
                "invalid": False,
                "out": 22,
            },
            # read expression composed of multiple functions
            {
                "larkexpr": "double(3+2* 6^2/9) - double(2 *5)",
                "funcs": {"double": double},
                "invalid": False,
                "out": 2,
            },
            # invalid function
            {
                "larkexpr": "notafunc(123)",
                "funcs": {"double": double},
                "invalid": True,
                "out": "Invalid function: 'notafunc()', "
                "valid functions are ['double()']",
            },
            # invalid expression
            {
                "larkexpr": "3*",
                "funcs": {},
                "invalid": True,
                "out": "Invalid characters in LarkExpression",
            },
            # empty expression
            {"larkexpr": "", "funcs": {}, "invalid": True, "out": "Empty String"},
        ]

        for test in larktests:

            if not test["invalid"]:
                e1 = LarkExpression(test["larkexpr"], functions=test["funcs"])
                self.assertTrue(e1.evaluate() == test["out"])
            else:
                with self.assertRaises(LarkExpressionError) as err:
                    e1 = LarkExpression(test["larkexpr"], functions=test["funcs"])
                self.assertEqual(str(err.exception), test["out"])

    def test_larkexpr_var(self):

        # Simple expressions
        e2 = LarkExpression("x+y+1", variables="xy")

        self.assertEqual(e2.variables, {"x", "y"})
        self.assertEqual(e2.evaluate(x=2, y=5), 8)

        # Try using a function
        def double(x):
            return 2 * x

        e3 = LarkExpression("double(x)", variables="x", functions={"double": double})

        self.assertEqual(e3.variables, {"x"})
        self.assertEqual(e3.functions, {"double"})
        self.assertEqual(e3.evaluate(x=2), 4)

        # lark expression does not contain variable given
        with self.assertRaises(LarkExpressionError) as err:
            e4 = LarkExpression("x+1", variables="y")
        self.assertEqual(
            str(err.exception), "Invalid variable: 'x', valid variables are ['y']"
        )

        # value of variable not given when evaluating
        with self.assertRaises(LarkExpressionError) as err:
            e5 = LarkExpression("x+1", variables="x")
            e5.evaluate()
        self.assertEqual(
            str(err.exception),
            "Some necessary variables have not been defined when "
            "evaluating LarkExpression",
        )

        # invalid variable value given when evaluating
        with self.assertRaises(LarkExpressionError) as err:
            e5 = LarkExpression("x+1", variables="x")
            e5.evaluate(x=1, y=1)
        self.assertEqual(
            str(err.exception),
            "Some invalid variables have been defined when "
            "evaluating LarkExpression",
        )

    def test_tokenization(self):
        tokens = lark_tokenize("3+0.4 2.3 sin(x) atan2(3, 4)")
        lark_tokens = [
            LarkExpression(
                tk, variables="x", functions={"sin": np.sin, "atan2": np.arctan2}
            )
            for tk in tokens
        ]

        self.assertEqual(len(tokens), 4)
        self.assertEqual(lark_tokens[0].evaluate(), 3.4)
        self.assertEqual(lark_tokens[1].evaluate(), 2.3)
        self.assertAlmostEqual(lark_tokens[2].evaluate(x=np.pi / 2.0), 1.0)
        self.assertAlmostEqual(lark_tokens[3].evaluate(), np.arctan2(3.0, 4.0))

    def test_keyword(self):

        # Basic keyword
        kw = MuSpinKeyword(["a b c", "d e f"])

        self.assertTrue((kw.evaluate() == [["a", "b", "c"], ["d", "e", "f"]]).all())
        self.assertEqual(len(kw), 2)

        # Test that the default works

        class DefKeyword(MuSpinKeyword):
            default = "1"

        dkw = DefKeyword()

        self.assertEqual(dkw.evaluate()[0][0], "1")

        # Let's try a numerical one
        nkw = MuSpinEvaluateKeyword(["exp(0) 1+1 2^2"])

        self.assertTrue((nkw.evaluate()[0] == [1, 2, 4]).all())

        exkw = MuSpinExpandKeyword(["range(0, 1)", "5.0 2.0"])

        self.assertTrue(len(exkw.evaluate()) == 101)

        # Test expansion of line in longer line

        def _repeat3(x):
            return [x, x, x]

        class RepeatKW(MuSpinExpandKeyword):
            _functions = {"repeat3": _repeat3}

        rkw = RepeatKW(["repeat3(1)"])

        self.assertTrue((rkw.evaluate()[0] == [1, 1, 1]).all())

        # Some failure cases
        with self.assertRaises(RuntimeError) as err:
            MuSpinKeyword([], args=["a"])  # One argument too much
        self.assertEqual(
            str(err.exception), "Wrong number of arguments passed to keyword keyword"
        )

    def test_input_keywords(self):

        nkw = InputKeywords["name"]()

        self.assertEqual(nkw.evaluate()[0], "muspinsim")

        skw = InputKeywords["spins"]()

        self.assertTrue((skw.evaluate()[0] == ["mu", "e"]).all())

        pkw = InputKeywords["polarization"]()

        self.assertTrue((pkw.evaluate()[0] == [1, 0, 0]).all())

        pkw = InputKeywords["polarization"]("longitudinal")

        self.assertTrue((pkw.evaluate()[0] == [0, 0, 1.0]).all())

        pkw = InputKeywords["polarization"]("transverse")

        self.assertTrue((pkw.evaluate()[0] == [1.0, 0, 0]).all())

        fkw = InputKeywords["field"](["500*MHz"])

        self.assertTrue(np.isclose(fkw.evaluate()[0][0], 1.84449))

        # Test a range of fields
        fkw = InputKeywords["field"](["range(0, 20, 21)"])

        self.assertTrue(
            (np.array([b[0] for b in fkw.evaluate()]) == np.arange(21)).all()
        )

        fkw = InputKeywords["field"]()

        self.assertTrue(fkw.evaluate()[0] == 0.0)

        tkw = InputKeywords["time"]()

        self.assertEqual(len(tkw.evaluate()), 101)
        self.assertEqual(tkw.evaluate()[-1][0], 10.0)

        with self.assertRaises(ValueError) as err:
            ykw = InputKeywords["y_axis"](["something"])
        self.assertEqual(
            str(err.exception), "Invalid block for keyword y_axis: Invalid value"
        )

        ykw = InputKeywords["y_axis"](["asymmetry"])

        self.assertEqual(ykw.evaluate()[0][0], "asymmetry")

        okw = InputKeywords["orientation"](["zcw(20)"])

        self.assertTrue(len(okw.evaluate()) >= 20)

        zkw = InputKeywords["zeeman"](["0 0 1"], args=["1"])

        self.assertEqual(zkw.id, "zeeman_1")

        dkw = InputKeywords["dipolar"](["0 0 1"], args=["1", "2"])

        self.assertEqual(dkw.id, "dipolar_1_2")

        hkw = InputKeywords["hyperfine"]([], args=["1"])

        self.assertTrue((hkw.evaluate()[0] == np.zeros((3, 3))).all())

        qkw = InputKeywords["quadrupolar"](
            ["1 0 0", "0 1 0", "0 0 cos(1)^2+sin(1)^2"], args=["1"]
        )

        self.assertTrue(np.isclose(qkw.evaluate()[0], np.eye(3)).all())

        # Failure case (wrong argument type)
        with self.assertRaises(RuntimeError) as err:
            InputKeywords["zeeman"]([], args=["wrong"])
        self.assertEqual(
            str(err.exception), "Invalid argument type passed to keyword zeeman"
        )

    def test_read_block(self):
        """test read input file unit test"""

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
            str(err.exception), "Invalid keyword notakeyword found in input file"
        )

    def test_fitting(self):
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

        # Let's test loading from a file
        tdata = np.zeros((10, 2))
        tdata[:, 0] = np.linspace(0, 1, 10)
        tdata[:, 1] = tdata[:, 0] ** 2

        tfile = NamedTemporaryFile(mode="w", delete=False)

        for d in tdata:
            tfile.write("{0} {1}\n".format(*d))
        tfile.flush()
        tfile.close()

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

        # invalid cases

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
            "Fitting variables defined without defining a set of data to fit",
        )

        # invalid variable range
        with self.assertRaises(ValueError) as err:
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
        self.assertEqual(
            str(err.exception),
            "Variable x has invalid range: "
            "(max value -5.0 cannot be less than or equal to min value 0.0)\n"
            "Variable x has invalid starting value: "
            "(starting value 1.0 cannot be greater than max value -5.0)",
        )

        # variable name clashes with constant
        with self.assertRaises(ValueError) as err:
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
            "Variable names {'MHz'} conflict with existing constants",
        )
