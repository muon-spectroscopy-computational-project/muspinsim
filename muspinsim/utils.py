"""utils.py

Utility functions and classes"""

from copy import deepcopy


class Clonable(object):
    """A helper class; any object inheriting 
    from this will have a .clone method that copies it easily."""

    def clone(self):

        MyClass = self.__class__
        copy = MyClass.__new__(MyClass)
        copy.__dict__ = deepcopy(self.__dict__)

        return copy
