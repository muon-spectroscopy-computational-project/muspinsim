"""utils.py

Utility functions and classes"""

from copy import deepcopy
from collections.abc import Iterable

import numpy as np
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


def zcw_gen(N, mode="sphere"):
    N = int(N)
    pwd = ZCW(mode)
    return pwd.get_orient_angles(N)[0]


def eulrange_gen(N):
    # Generate a range of Euler angles and related weights
    N = int(N)  # Just making sure
    a = np.linspace(0, 2 * np.pi, N)
    b = np.linspace(0, np.pi, N + 2)[1:-1]
    c = np.linspace(0, 2 * np.pi, N)

    a, b, c = np.array(np.meshgrid(a, b, c)).reshape((3, -1))
    w = np.sin(b)

    return np.array([a, b, c, w]).T


def quat_from_polar(theta, phi):
    """Make a Quaternion from two polar angles

    Make a Quaternion from only two polar angles. This only makes the new Z'
    axis have polar angles theta and phi w.r.t. the old system, but does not
    care or guarantee anything about the other two angles.

    Arguments:
        theta {float} -- Zenithal angle
        phi {float} -- Azimuthal angle

    Returns:
        q {ase.Quaternion} -- Quaternion
    """

    return Quaternion.from_euler_angles(phi, theta, phi, "zyz")


def get_xy(z):
    """Make two axes x and y orthogonal and correctly oriented with respect
    to an axis z"""

    zn = np.linalg.norm(z)
    if zn == 0:
        raise ValueError("Can not find X and Y for null Z vector")
    z = np.array(z) / zn

    if z[0] == 0 and z[1] == 0:
        x = np.array([1.0, 0, 0])
        y = np.array([0, z[2], 0])
    else:
        x = np.array([z[1], -z[0], 0.0]) / (z[0] ** 2 + z[1] ** 2) ** 0.5
        y = np.array(
            [
                z[1] * x[2] - z[2] * x[1],
                z[2] * x[0] - z[0] * x[2],
                z[0] * x[1] - z[1] * x[0],
            ]
        )

    return x, y
