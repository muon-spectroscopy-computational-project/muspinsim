"""utils.py

Utility functions and classes"""

from copy import deepcopy
from collections.abc import Iterable

from ase.quaternions import Quaternion
from soprano.calculate.powder import ZCW


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


def zcw_gen(N, mode='sphere'):
    pwd = ZCW(mode)
    return pwd.get_orient_angles(N)[0]


def quat_from_polar(theta, phi):
    """Make a Quaternion from two polar angles

    Make a Quaternion from only two polar angles.

    Arguments:
        theta {float} -- Zenithal angle
        phi {float} -- Azimuthal angle

    Returns:
        q {ase.Quaternion} -- Quaternion
    """

    return Quaternion.from_euler_angles(0.0, theta, phi, 'zyz')
