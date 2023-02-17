import numpy as np


class FittingVariable:
    """Class defining a fitting variable with a starting value and bounds"""

    def __init__(self, name, value=0.0, min_value=-np.inf, max_value=np.inf):

        self._name = name
        self._value = float(value)
        self._min = float(min_value)
        self._max = float(max_value)

        invalid = ""
        if self._max <= self._min:
            invalid += (
                f"Variable {name} has invalid range: "
                f"(max value {self._max} cannot be less than or equal to min "
                f"value {self._min}"
                ")\n"
            )
        if self._value > self._max:
            invalid += (
                f"Variable {name} has invalid starting value: "
                f"(starting value {self._value} cannot be greater than max "
                f"value {self._max})"
            )
        if self._value < self._min:
            invalid += (
                f"Variable {name} has invalid starting value: "
                f"(starting value {self._value} cannot be less than min "
                f"value {self._min})"
            )

        if invalid != "":
            raise ValueError(invalid)

    @property
    def name(self):
        return self._name

    @property
    def value(self):
        return self._value

    @property
    def bounds(self):
        return (self._min, self._max)
