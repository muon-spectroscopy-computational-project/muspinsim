"""larkeval.py

Evaluation functions and classes based off the Lark grammar parser"""

from typing import Dict, List, Optional
from lark import Lark
from lark.exceptions import UnexpectedToken, UnexpectedInput

_expr_parser = Lark(
    """
    ?start: sum

    ?function: NAME "(" (sum ("," sum)*)? ")" -> fun

    ?sum: product
        | sum "+" product   -> add
        | sum "-" product   -> sub

    ?product: power
        | product "*" power  -> mul
        | product "/" power  -> div

    ?power: atom
        | power "^" atom -> pow

    ?atom: NUMBER           -> num
         | "-" atom         -> neg
         | NAME             -> var
         | STRING           -> str
         | "(" sum ")"
         | function

    %import common.CNAME -> NAME
    %import common.NUMBER
    %import common.WS_INLINE
    %import common.ESCAPED_STRING -> STRING

    %ignore WS_INLINE
""",
    parser="lalr",
)


def lark_tokenize(line):
    """Splint a line of space-separated valid expressions."""
    ls = line.split()
    tokens = []
    while len(ls) > 0:
        tk = ls.pop(0)
        valid = False
        while not valid:
            try:
                _expr_parser.parse(tk)
                valid = True
            except UnexpectedToken as exc:
                if len(ls) == 0:
                    raise RuntimeError(
                        f"Line can not be tokenized into valid expressions: {str(exc)}"
                    ) from exc
                tk = tk + ls.pop(0)
            except UnexpectedInput as exc:
                raise LarkExpressionError(f"Could not parse input: {str(exc)}") from exc
        tokens.append(tk)

    return tokens


class LarkExpressionError(Exception):
    pass


class LarkExpression:
    def __init__(
        self,
        source,
        variables: Optional[List[str]] = None,
        functions: Optional[Dict[str, callable]] = None,
    ):
        """Create an expression to parse and evaluate with the Lark grammar
        parser. Variables and functions will be accepted only if included
        in the admissible ones.

        Arguments:
            source {str} -- Source code of the expression

        Keyword Arguments
            variables {[str]} -- Names of acceptable variables
            functions {{str: callable}} -- Names and bodies of acceptable
                                           functions
        """

        if variables is None:
            variables = []
        if functions is None:
            functions = {}

        # Start by parsing the expression
        self._source = source
        # check if source empty
        if source == "":
            raise LarkExpressionError("Empty String")
        try:
            self._tree = _expr_parser.parse(source)
        except UnexpectedToken as exc:
            raise LarkExpressionError(
                f"Invalid characters in LarkExpression: {str(exc)}"
            ) from exc

        # Find the variables and the function calls
        found_vars, found_funcs = self._analyse_tree(self._tree)

        self._variables = set(found_vars)
        self._functions = set(found_funcs)
        self._all_variables = set(
            [variables] if isinstance(variables, str) else variables
        )

        # Check if they are valid
        for v in self._variables:
            if v not in self._all_variables:
                raise LarkExpressionError(
                    "Invalid variable/constant: '{0}', "
                    "valid variables/constants are ['{1}']".format(
                        v, "', '".join(self._all_variables)
                    )
                )

        for f in self._functions:
            if f not in functions.keys():
                raise LarkExpressionError(
                    "Invalid function: '{0}()', valid functions are ['{1}()']".format(
                        f, "()', '".join(functions.keys())
                    )
                )

        self._function_bodies = {fn: functions[fn] for fn in self._functions}

        self._store_eval = None
        if len(self._variables) == 0:
            # No need to wait
            self._store_eval = self.evaluate()

    def _analyse_tree(self, root):
        """Traverse the tree to look for variables and functions"""

        found_vars = []
        found_functions = []

        if hasattr(root, "data"):
            children = root.children
            if root.data == "var":
                found_vars = [root.children[0].value]
            elif root.data == "fun":
                found_functions = [root.children[0].value]
                children = root.children[1:]  # Skip the NAME
        else:
            return [], []

        for c in children:
            fv, ff = self._analyse_tree(c)
            found_vars += fv
            found_functions += ff

        return found_vars, found_functions

    def _evaluate_tree(self, root, variables=None):
        if variables is None:
            variables = []

        if hasattr(root, "data"):
            d = root.data
            evc = [self._evaluate_tree(c, variables) for c in root.children]
            if d == "add":
                return evc[0] + evc[1]
            elif d == "sub":
                return evc[0] - evc[1]
            elif d == "mul":
                return evc[0] * evc[1]
            elif d == "div":
                return evc[0] / evc[1]
            elif d == "pow":
                return evc[0] ** evc[1]
            elif d == "neg":
                return -evc[0]
            elif d == "num":
                return float(evc[0])
            elif d == "str":
                return str(evc[0][1:-1])
            elif d == "var":
                return variables[evc[0]]
            elif d == "fun":
                fname = evc[0]
                return self._function_bodies[fname](*evc[1:])
        else:
            return root.value

    @property
    def functions(self):
        return self._functions

    @property
    def variables(self):
        return self._variables

    def evaluate(self, **variables):
        """Evaluate the expression with a given set of updated variables.

        Keyword Arguments:
            All the variable names appearing in self.variables.

        Returns:
            result {any} -- Result of evaluating the expression.
        """

        vset = set(variables.keys())
        if len(self.variables - vset) > 0:
            raise LarkExpressionError(
                f"Some necessary variable(s) {self.variables - vset} have not "
                "been defined when evaluating LarkExpression"
            )
        elif len(vset - self._all_variables) > 0:
            raise LarkExpressionError(
                f"Some invalid variable(s) {vset - self._all_variables} have "
                "been defined when evaluating LarkExpression"
            )

        if self._store_eval is not None:
            return self._store_eval
        else:
            return self._evaluate_tree(self._tree, variables)
