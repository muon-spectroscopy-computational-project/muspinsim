"""utils.py

Utility functions and classes"""

from copy import deepcopy
from collections.abc import Iterable


class Clonable(object):
    """A helper class; any object inheriting 
    from this will have a .clone method that copies it easily."""

    def clone(self):

        MyClass = self.__class__
        copy = MyClass.__new__(MyClass)
        copy.__dict__ = deepcopy(self.__dict__)

        return copy

def deepmap(func, obj):
    """Deep traverse obj, and apply func to each of its non-Iterable 
    elements"""

    if isinstance(obj, Iterable):
        return [deepmap(func, x) for x in obj]
    else:
        return func(obj)
