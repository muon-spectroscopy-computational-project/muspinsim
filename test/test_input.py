import unittest
import numpy as np

from muspinsim.input.asteval import (ASTExpression, ASTExpressionError,
                                     ast_tokenize)
from muspinsim.input.keyword import (MuSpinKeyword, MuSpinEvaluateKeyword,
                                     MuSpinExpandKeyword, MuSpinTensorKeyword)


class TestInput(unittest.TestCase):

    def test_astexpr(self):

        # Start by testing simple expressions
        def double(x):
            return 2*x

        e1 = ASTExpression('x+y+1', variables='xy')

        self.assertEqual(e1.variables, {'x', 'y'})
        self.assertEqual(e1.evaluate(x=2, y=5), 8)

        # Try using a function
        e2 = ASTExpression('double(x)', variables='x',
                           functions={'double': double})

        self.assertEqual(e2.variables, {'x'})
        self.assertEqual(e2.functions, {'double'})
        self.assertEqual(e2.evaluate(x=2), 4)

        # Check that it evaluates when possible
        e3 = ASTExpression('double(4)', functions={'double': double})
        self.assertEqual(e3._store_eval, 8)

        # Errors
        with self.assertRaises(ASTExpressionError):
            e4 = ASTExpression('print(666)')

        with self.assertRaises(ASTExpressionError):
            e5 = ASTExpression('x+1', variables='y')

        with self.assertRaises(ASTExpressionError):
            e1.evaluate()

        # Test tokenization
        ast_tokenize('3.4 2.3 sin(x) atan2(3, 4)')

    def test_keyword(self):

        # Basic keyword
        kw = MuSpinKeyword(['a b c', 'd e f'])

        self.assertTrue((kw.evaluate() == [['a', 'b', 'c'],
                                           ['d', 'e', 'f']]
                         ).all())
        self.assertEqual(len(kw), 2)

        # Let's try a numerical one
        nkw = MuSpinEvaluateKeyword(['exp(0) 1+1 2**2'])

        self.assertTrue((nkw.evaluate()[0] == [1, 2, 4]).all())

        exkw = MuSpinExpandKeyword(['range(0, 1)', '5.0 2.0'])

        self.assertTrue(len(exkw.evaluate()) == 101)

        # Now a tensor
        tkw = MuSpinTensorKeyword(['1 0 0',
                                   '0 1 0',
                                   '0 0  cos(1)**2+sin(1)**2'])

        self.assertTrue((tkw.evaluate()[0] == np.eye(3)).all())
