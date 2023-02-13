"""constants.py

Collects methods to retrieve important physical constants
"""

from typing import Optional
import numpy as np
from scipy import constants as cnst
from soprano.nmr.utils import _get_isotope_data
from mendeleev import Isotope, element, isotope
from fractions import Fraction

# Values taken from CODATA on 18/01/2021
# Electron gyromagnetic ratio (MHz/T)
ELEC_GAMMA = -28024.9514242
MU_GAMMA = -(ELEC_GAMMA / 206.7669883)  # Muon gyromagnetic ratio (MHz/T)
MU_TAU = 2.19703  # Muon decay rate (10^-6 s)
# EFG to MHz constant for Quadrupole couplings
# (the total quadrupole coupling in MHz is QCONST*Q*Vzz)
# The factor 1e-37 comes from 1e-6 * 1e-28 * 1e-3
# in order this converts to MHz, from barn to m^2 and the last factor
# accounts for the soprano returning quadrupole moments in millibarn
EFG_2_MHZ = (
    cnst.physical_constants["atomic unit of electric field gradient"][0]
    * cnst.e
    * 1e-37
    / cnst.h
)


def _get_isotope_data_new(elem: str, iso: Optional[int]) -> Isotope:
    """Returns isotope data from mendeleev for the requested element isotope

    Arguments:
        elem {str} -- Element
        iso {int} -- Desired isotope. If not specified, the most naturally
                     abundant isotope is used. (default: {None})

    Returns:
        {Isotope} -- Isotope data from the mendeleev package

    Raises:
        ValueError -- When either the element or isotope is invalid
    """

    # Find most abundant
    if iso is None:
        try:
            elem_data = element(elem)
        except ValueError as exc:
            raise ValueError(f"Invalid element {elem}") from exc

        elem_isotopes = sorted(
            elem_data.isotopes,
            key=lambda isotope: isotope.abundance
            # Abundance can be None => very small/unknown
            if isotope.abundance is not None else 0,
            reverse=True,
        )
        return elem_isotopes[0]

    # Select a specific isotope
    try:
        isotope_data = isotope(symbol_or_atn=elem, mass_number=iso)
    except ValueError as exc:
        raise ValueError(f"Invalid element {elem} or isotope {iso}") from exc

    return isotope_data


def gyromagnetic_ratio(elem="mu", iso=None):
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

    if elem == "e":
        return ELEC_GAMMA
    elif elem == "mu":
        return MU_GAMMA
    else:
        try:
            val = _get_isotope_data([elem], "gamma", isotope_list=[iso])[0]
        except RuntimeError as exc:
            raise ValueError(f"Invalid isotope {iso} for element {elem}") from exc

        return val / (2e6 * np.pi)


def quadrupole_moment(elem="mu", iso=None):
    """Return the quadrupole moment of a given particle

    Return the quadrupole moment of either an atomic nucleus,
    an electron or a muon. The value returned is in barn.

    Keyword Arguments:
        elem {str} -- Element ('e' for electron, 'mu' for muon) (default: {'mu'})
        iso {int} -- Desired isotope. Ignored for 'e' and 'mu'. If not
                     specified, the most naturally abundant isotope is used.
                     (default: {None})
    """

    if elem in ("e", "mu"):
        return 0
    else:
        try:
            val = _get_isotope_data([elem], "Q", isotope_list=[iso])[0]
        except RuntimeError as exc:
            raise ValueError(f"Invalid isotope {iso} for element {elem}") from exc

        return val


def spin(elem="mu", iso=None):
    """Return the spin of a given particle

    Return the intrinsic angular momentum of either an atomic nucleus,
    an electron or a muon. The value returned is in hbar.

    Keyword Arguments:
        elem {str} -- Element ('e' for electron, 'mu' for muon) (default: {'mu'})
        iso {int} -- Desired isotope. Ignored for 'mu'. If used for 'e', this
                     is interpreted as a number of strongly coupled electrons
                     acting as a single spin > 1/2. If not specified, the most
                     naturally abundant isotope is used.
                     (default: {None})
    """

    if elem == "mu":
        return 0.5
    elif elem == "e":
        iso = iso or 1
        if iso < 1 or int(iso) != iso:
            raise ValueError(f"Invalid multiplicity {iso} for electron")
        return 0.5 * int(iso)
    else:
        spin_val = _get_isotope_data_new(elem=elem, iso=iso).spin

        Q_val = _get_isotope_data([elem], "Q", isotope_list=[iso])[0]
        Q_val_new = _get_isotope_data_new(elem=elem, iso=iso).quadrupole_moment

        gamma_val = _get_isotope_data([elem], "gamma", isotope_list=[iso])[0]
        g_factor = _get_isotope_data_new(elem=elem, iso=iso).g_factor

        gamma_val_new = None
        if g_factor:
            gamma_val_new = cnst.e * g_factor / (2 * cnst.m_p)

        print(
            f"Spin elem: {elem} iso: {iso} val: {float(Fraction(spin_val))} Q_val_old: {Q_val} Q_val_new: {Q_val_new} gamma_val: {gamma_val} gamma_val_new: {gamma_val_new}"
        )

        # Spins are provided as a string in the form of a fraction e.g. 7/2 so
        # convert to float here
        return float(Fraction(spin_val))
