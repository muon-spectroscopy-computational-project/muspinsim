"""asteval.py

Evaluation functions and classes based off Abstract Syntax Trees"""

import ast


def ast_tokenize(line):
    """Splint a line of space-separated valid expressions."""
    ls = line.split()
    tokens = []
    while len(ls) > 0:
        tk = ls.pop(0)
        valid = False
        while not valid:
            try:
                ast.parse(tk, mode="eval")
                valid = True
            except SyntaxError:
                if len(ls) == 0:
                    raise RuntimeError(
                        "Line can not be tokenized into valid " "expressions"
                    )
                tk = tk + ls.pop(0)
        tokens.append(tk)

    return tokens


class ASTExpressionError(Exception):
    pass


class ASTExpression(object):
    def __init__(self, source, variables=[], functions={}):
        """Create an expression to check with Abstract Syntax Trees before
        evaluating. Variables and functions will be accepted only if included
        in the admissible ones.

        Arguments:
            source {str} -- Source code of the expression

        Keyword Arguments
            variables {[str]} -- Names of acceptable variables
            functions {{str: callable}} -- Names and bodies of acceptable
                                           functions
        """

        # Start by parsing the expression
        self._source = source
        self._ast = ast.parse(source, mode="eval")

        # Find the variables and the function calls
        found_names = []
        found_functions = []
        for o in ast.walk(self._ast):
            if type(o) is ast.Call:
                # Function
                name = o.func.id
                found_functions.append(name)
            elif type(o) is ast.Name:
                name = o.id
                found_names.append(name)
            elif type(o) in (ast.Attribute,):
                # A further safety against exploits. This stuff should never appear here!
                raise ASTExpressionError("Unsafe expression used in ASTExpression")

        found_names = set(found_names)
        self._functions = set(found_functions)
        self._variables = found_names - self._functions
        self._all_variables = set(variables)

        # Check if they are valid
        if len(self.variables - self._all_variables) > 0:
            raise ASTExpressionError("Invalid variable used in ASTExpression")

        if len(self.functions - set(functions.keys())) > 0:
            raise ASTExpressionError("Invalid function used in ASTExpression")

        self._function_bodies = {fn: functions[fn] for fn in self._functions}

        self._store_eval = None
        if len(self.variables) == 0:
            # No need to wait
            self._store_eval = eval(source, self._function_bodies)

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
            raise ASTExpressionError(
                "Some necessary variables have not been "
                "defined when evaluating ASTExpression"
            )
        elif len(vset - self._all_variables) > 0:
            raise ASTExpressionError(
                "Some invalid variables have been "
                "defined when evaluating ASTExpression"
            )

        if self._store_eval is not None:
            return self._store_eval
        else:
            return eval(self._source, self._function_bodies, variables)
