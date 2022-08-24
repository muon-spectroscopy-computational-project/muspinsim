import unittest
from muspinsim.input.larkeval import LarkExpression, LarkExpressionError, lark_tokenize
import numpy as np


def _double(x):
    return x * 2


class TestLark(unittest.TestCase):
    def _larkepxr_parse(self, test):
        """Lark expression unit test"""
        eval_vars = {k: v for k, v in test["vars"].items() if v is not None}
        if not test["invalid"]:
            e1 = LarkExpression(
                test["expr"],
                functions=test["funcs"],
                variables=list(test["vars"].keys()),
            )
            self.assertTrue(e1.evaluate(**eval_vars) == test["out"])
        else:
            with self.assertRaises(LarkExpressionError) as err:
                e1 = LarkExpression(
                    test["expr"],
                    functions=test["funcs"],
                    variables=list(test["vars"].keys()),
                )
                e1.evaluate(**eval_vars)

            self.assertTrue(test["out"] in str(err.exception))

    def test_lark_expr(self):
        self._larkepxr_parse(
            {"expr": "3+2*6^2/9", "funcs": {}, "vars": {}, "invalid": False, "out": 11}
        )

    def test_lark_func(self):

        self._larkepxr_parse(
            {
                "expr": "double(2)",
                "funcs": {"double": _double},
                "vars": {},
                "invalid": False,
                "out": 4,
            }
        )

    def test_lark_expr_in_func(self):
        self._larkepxr_parse(
            {
                "expr": "double(3+2*6^2/9)",
                "funcs": {"double": _double},
                "vars": {},
                "invalid": False,
                "out": 22,
            }
        )

    def test_lark_multi_expr_in_func(self):
        self._larkepxr_parse(
            {
                "expr": "double(3+2* 6^2/9) - double(2 *5)",
                "funcs": {"double": _double},
                "vars": {},
                "invalid": False,
                "out": 2,
            }
        )

    def test_lark_invalid_func(self):
        self._larkepxr_parse(
            {
                "expr": "notafunc(123)",
                "funcs": {"double": _double},
                "vars": {},
                "invalid": True,
                "out": "Invalid function: 'notafunc()', "
                "valid functions are ['double()']",
            }
        )

    def test_lark_invalid_expr(self):
        self._larkepxr_parse(
            {
                "expr": "3*",
                "funcs": {},
                "vars": {},
                "invalid": True,
                "out": "Invalid characters in LarkExpression: Unexpected token",
            }
        )

    def test_lark_empty(self):
        self._larkepxr_parse(
            {
                "expr": "",
                "funcs": {},
                "vars": {},
                "invalid": True,
                "out": "Empty String",
            }
        )

    def test_lark_var(self):
        self._larkepxr_parse(
            {"expr": "x+1", "funcs": {}, "vars": {"x": 5}, "invalid": False, "out": 6}
        )

    def test_lark_func_with_var(self):
        self._larkepxr_parse(
            {
                "expr": "double(x)",
                "funcs": {"double": _double},
                "vars": {"x": 2},
                "invalid": False,
                "out": 4,
            }
        )

    def test_lark_multi_var(self):
        self._larkepxr_parse(
            {
                "expr": "x+y+1",
                "funcs": {},
                "vars": {"x": 2, "y": 5},
                "invalid": False,
                "out": 8,
            }
        )

    def test_lark_vars_value_missing(self):
        self._larkepxr_parse(
            {
                "expr": "x+1",
                "funcs": {},
                "vars": {"x": None},
                "invalid": True,
                "out": "Some necessary variable(s) {'x'} have not "
                "been defined when evaluating LarkExpression",
            }
        )

    def test_lark_invalid_var(self):
        self._larkepxr_parse(
            {
                "expr": "x+1",
                "funcs": {},
                "vars": {"y": 2},
                "invalid": True,
                "out": "Invalid variable/constant: 'x', "
                "valid variables/constants are ['y']",
            }
        )

    def test_lark_evaluate_with_invalid_var(self):
        with self.assertRaises(LarkExpressionError) as err:
            out = LarkExpression("x+1", variables="x")
            out.evaluate(x=1, y=1)
        self.assertEqual(
            str(err.exception),
            "Some invalid variable(s) {'y'} have been defined when "
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
