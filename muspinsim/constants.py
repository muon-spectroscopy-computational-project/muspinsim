"""constants.py

Collects methods to retrieve important physical constants
"""

import numpy as np
from scipy import constants as cnst
from soprano.nmr.utils import _get_isotope_data

# Values taken from CODATA on 18/01/2021
ELEC_GAMMA = 28024.9514242
MU_GAMMA = (ELEC_GAMMA/206.7669883)


def gyromagnetic_ratio(elem='mu', iso=None):
    """Return the gyromagnetic ratio of a given particle

    Return the gyromagnetic ratio of either an atomic nucleus,
    an electron or a muon. The value returned is in MHz/T. It 
    corresponds to a frequency, not a pulsation (so it's gamma,
    not gammabar = 2*pi*gamma).

    Keyword Arguments:
        elem {str} -- Element ('e' for electron, 'mu' for muon) (default: {'mu'})
        iso {int} -- Desired isotope. Ignored for 'e' and 'mu'. If not
                     specified, the most naturally abundant isotope is used. 
                     (default: {None})
    """

    if elem == 'e':
        return ELEC_GAMMA
    elif elem == 'mu':
        return MU_GAMMA
    else:
        try:
            val = _get_isotope_data([elem], 'gamma', isotope_list=[iso])[0]
        except RuntimeError:
            raise ValueError('Invalid isotope {0} for element {1}'.format(
                iso, elem))

        return val/(2e6*np.pi)


def quadrupole_moment(elem='mu', iso=None):
    """Return the quadrupole moment of a given particle

    Return the quadrupole moment of either an atomic nucleus,
    an electron or a muon. The value returned is in barn.

    Keyword Arguments:
        elem {str} -- Element ('e' for electron, 'mu' for muon) (default: {'mu'})
        iso {int} -- Desired isotope. Ignored for 'e' and 'mu'. If not
                     specified, the most naturally abundant isotope is used. 
                     (default: {None})
    """

    if elem in ('e', 'mu'):
        return 0
    else:
        try:
            val = _get_isotope_data([elem], 'Q', isotope_list=[iso])[0]
        except RuntimeError:
            raise ValueError('Invalid isotope {0} for element {1}'.format(
                iso, elem))

        return val


def spin(elem='mu', iso=None):
    """Return the spin of a given particle

    Return the intrinsic angular momentum of either an atomic nucleus,
    an electron or a muon. The value returned is in hbar.

    Keyword Arguments:
        elem {str} -- Element ('e' for electron, 'mu' for muon) (default: {'mu'})
        iso {int} -- Desired isotope. Ignored for 'e' and 'mu'. If not
                     specified, the most naturally abundant isotope is used. 
                     (default: {None})
    """

    if elem in ('e', 'mu'):
        return 0.5
    else:
        try:
            val = _get_isotope_data([elem], 'I', isotope_list=[iso])[0]
        except RuntimeError:
            raise ValueError('Invalid isotope {0} for element {1}'.format(
                iso, elem))

        return val