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
from muspinsim.input import MuSpinInput


class TestInput(unittest.TestCase):
    def test_larkexpr(self):

        # Start by testing proper precedence order
        e1 = LarkExpression("3+2*6^2/9")

        self.assertEqual(e1.evaluate(), 11)

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

        # Check that it evaluates when possible
        e4 = LarkExpression("double(4)", functions={"double": double})
        self.assertEqual(e4._store_eval, 8)

        # Errors
        with self.assertRaises(LarkExpressionError):
            e5 = LarkExpression("print(666)")

        with self.assertRaises(LarkExpressionError):
            e6 = LarkExpression("x+1", variables="y")

        with self.assertRaises(LarkExpressionError):
            e2.evaluate()

        with self.assertRaises(LarkExpressionError):
            LarkExpression("3x%5")

        # Test tokenization
        tokens = lark_tokenize("3.4 2.3 sin(x) atan2(3, 4)")
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
        with self.assertRaises(RuntimeError):
            MuSpinKeyword([], args=["a"])  # One argument too much

    def test_input_keywords(self):

        nkw = InputKeywords["name"]()

        self.assertEqual(nkw.evaluate()[0], "muspinsim")

        skw = InputKeywords["spins"]()

        self.assertTrue((skw.evaluate()[0] == ["mu", "e"]).all())

        pkw = InputKeywords["polarization"]()

        self.assertTrue((pkw.evaluate()[0] == [1, 0, 0]).all())

        fkw = InputKeywords["field"](["500*MHz"])

        self.assertTrue(np.isclose(fkw.evaluate()[0][0], 1.84449))

        # Test a range of fields
        fkw = InputKeywords["field"](["range(0, 20, 21)"])

        self.assertTrue(
            (np.array([b[0] for b in fkw.evaluate()]) == np.arange(21)).all()
        )

        tkw = InputKeywords["time"]()

        self.assertEqual(len(tkw.evaluate()), 101)
        self.assertEqual(tkw.evaluate()[-1][0], 10.0)

        with self.assertRaises(ValueError):
            ykw = InputKeywords["y_axis"](["something"])

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
        with self.assertRaises(RuntimeError):
            InputKeywords["zeeman"]([], args=["wrong"])

    def test_input(self):

        s1 = StringIO(
            """
name
    test_1
spins
    mu H
zeeman 1
    1 0 0
"""
        )

        i1 = MuSpinInput(s1)
        e1 = i1.evaluate()

        self.assertEqual(e1["name"].value[0], "test_1")
        self.assertTrue((e1["spins"].value[0] == ["mu", "H"]).all())
        self.assertTrue((e1["couplings"]["zeeman_1"].value[0] == [1, 0, 0]).all())

    def test_fitting(self):
        # Test input focused around fitting

        s1 = StringIO(
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
        i1 = MuSpinInput(s1)

        self.assertTrue(i1.fitting_info["fit"])

        data = i1.fitting_info["data"]
        self.assertTrue((data == [[0, 0], [1, 1], [2, 4], [3, 9]]).all())

        e1 = i1.evaluate(x=2.0)
        self.assertEqual(e1["field"].value[0][0], 4.0)
        self.assertTrue((e1["couplings"]["zeeman_1"].value[0] == [2, 2, 0]).all())

        variables = i1.variables

        self.assertEqual(variables["x"].value, 1.0)
        self.assertEqual(variables["x"].bounds, (0.0, 2.0))

        # Invalid variable range
        s2 = StringIO(
            """
fitting_variables
    x 1.0 0.0 -5.0
"""
        )

        with self.assertRaises(ValueError):
            MuSpinInput(s2)

        # Let's test loading from a file
        tdata = np.zeros((10, 2))
        tdata[:, 0] = np.linspace(0, 1, 10)
        tdata[:, 1] = tdata[:, 0] ** 2

        tfile = NamedTemporaryFile(mode="w", delete=False)

        for d in tdata:
            tfile.write("{0} {1}\n".format(*d))
        tfile.flush()
        tfile.close()

        s3 = StringIO(
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

        i3 = MuSpinInput(s3)

        finfo = i3.fitting_info

        data = finfo["data"]
        self.assertTrue(finfo["fit"])
        self.assertTrue((data == tdata).all())
        self.assertEqual(finfo["method"], "nelder-mead")
        self.assertAlmostEqual(finfo["rtol"], 1e-3)
