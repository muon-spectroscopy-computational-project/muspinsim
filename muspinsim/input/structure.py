from dataclasses import dataclass
import math
from pathlib import PurePath
from typing import IO, List, Optional, Union
import numpy as np
import ase.io
from numpy.typing import ArrayLike

from muspinsim.constants import spin


@dataclass
class CellAtom:
    """
    Dataclass storing information about an atom in a cell structure
    """

    index: int  # Start from 1, unchanged when creating a supercell
    symbol: str
    isotope: int
    position: ArrayLike
    vector_from_muon: Optional[ArrayLike] = None
    distance_from_muon: Optional[float] = None


def _get_isotope(mass: float, default_mass: float) -> int:
    """
    Helper function to determine the isotope by comparing the found mass to
    the default of the same element

    At the moment will not return anything other than 1 as there is no where
    to get the isotope masses from.
    """

    # Expect close to whole division if valid
    if not math.isclose(mass, default_mass):
        raise ValueError("Failed to identify isotope by the given masses")

    return 1


class MuonatedStructure:
    """
    A class for handling data from a muonated structure file and computing
    distances between the atoms
    """

    # Loaded data
    _unit_lengths: ArrayLike
    _unit_angles: ArrayLike
    _atoms: List[CellAtom] = []
    _muon_index: int = None

    _symbols_zero_spin: List[str] = []

    def __init__(
        self,
        file_io: Union[str, PurePath, IO],
        fmt: Union[str, None] = None,
        muon_symbol: Union[str, int] = "H",
    ):
        """Load a muonated cell structure file

        Read in a cell structure from an opened file stream

        Arguments:
            file_io {TextIOBase} -- File path, or IO stream
            fmt {str} -- Format of the file/data. See ase documentation for
                         available ones. When None ase will attempt to
                         determine it.
            muon_symbol {str} -- Symbol that should represent a muon.

        Raises:
            ValueError -- If the structure file has unit cell vector angles
                          different from [90, 90, 90].
            ValueError -- If the structure file contains more than one muon
                          with the given symbol.
            ValueError -- If the structure file has no muon with the given
                          symbol.
        """

        # Load the atomic data from the file
        # NOTE: Calculator is loaded automatically - can't see a way to
        # disable but this will cause warnings when reading CASTEP files
        # without a CASTEP installation - these can be ignored
        loaded_atoms = ase.io.read(file_io, format=fmt)

        lengths_and_angles = loaded_atoms.get_cell_lengths_and_angles()
        self._unit_lengths = lengths_and_angles[:3]
        self._unit_angles = lengths_and_angles[3:]

        # ase has a make_supercell method, unfortunately this is not suitable
        # as need indices to remain same for linking to other files
        if (self._unit_angles != 90).any():
            raise ValueError(
                f"Structure file '{file_io}' has unsupported unit cell angles"
            )

        self._atoms = [None] * len(loaded_atoms)

        # Store only needed data for calculations
        for i, loaded_atom in enumerate(loaded_atoms):

            # Determine isotope by looking at loaded mass
            default_mass = ase.data.atomic_masses[
                ase.data.atomic_numbers[loaded_atom.symbol]
            ]
            isotope = _get_isotope(loaded_atom.mass, default_mass)

            if loaded_atom.symbol == muon_symbol:
                if self._muon_index is None:
                    self._muon_index = i
                else:
                    raise ValueError(
                        f"Structure file '{file_io}' has more than one muon "
                        f"with symbol {muon_symbol}"
                    )

            self._atoms[i] = CellAtom(
                i + 1, loaded_atom.symbol, isotope, loaded_atom.position
            )

            # Keep track of any atoms with zero spin (so can exclude later)
            if (
                loaded_atom.symbol not in self._symbols_zero_spin
                and spin(elem=loaded_atom.symbol, iso=isotope if isotope > 1 else None)
                == 0
            ):
                self._symbols_zero_spin.append(loaded_atom.symbol)

        if self._muon_index is None:
            raise ValueError(
                f"Structure file '{file_io}' has no muon with symbol " f"{muon_symbol}"
            )

    def expand(self, offsets: List[ArrayLike]):
        """Expands the structure by duplicating loaded atoms (ignoring the
           muon)

        Expands the structure by cloning atoms (excluding the muon) given
        offsets to to modify their positions by.

        Arguments:
            offsets {List[ArrayLike]} -- Offset for the new atoms that will be
                                         added. For each offset will duplicate
                                         the current structure.
        """

        new_atoms = []
        for offset in offsets:
            # Copy everything but the muon
            for i, atom in enumerate(self._atoms):
                if i != self._muon_index:
                    new_atoms.append(
                        CellAtom(
                            index=atom.index,
                            symbol=atom.symbol,
                            isotope=atom.isotope,
                            position=atom.position + offset,
                        )
                    )
        self._atoms.extend(new_atoms)

    def layer_expand(self, layers: int):
        """Expands the structure outwards by duplicating it.

        Expands the structure outwards by duplicating it a specified number of
        layers.

        Arguments:
            layers {int} -- Number of layers to append e.g. if layers = 1,
                            will expand along the x, y and z axes creating a
                            structure 9 times larger than the original.
        """

        # Construct the offsets
        values = [np.arange(
            -layers * self._unit_lengths[i],
            (layers + 1) * self._unit_lengths[i],
            self._unit_lengths[i],
        ) for i in range(3)]
        offsets = np.array(np.meshgrid(*values)).T.reshape(-1, 3)
        # Remove 0, 0, 0
        offsets = np.delete(offsets, int(np.floor(len(offsets) / 2)), axis=0)

        # Now expand for each
        self.expand(offsets)

    def compute_distances(self):
        """Computes the vectors and distances between the muon and every atom.
        """
        muon = self._atoms[self._muon_index]

        for atom in self._atoms:
            atom.vector_from_muon = muon.position - atom.position
            atom.distance_from_muon = np.linalg.norm(atom.vector_from_muon)

    def get_closer_than(
        self, distance: float, ignored_symbols: Optional[List[str]] = None
    ) -> List[CellAtom]:
        """Returns atoms which are closer to the muon than a given distance

        Expects compute_distances to have been called first.

        Arguments:
            distance {float} -- Distance that the returned atoms should be
                                closer than.
            ignored_symbols {List[str]} -- List of symbols to ignore. May be
                                           None.

        Returns:
            List[CellAtom] -- List of atoms closer than the given distance
        """

        if ignored_symbols is None:
            return filter(
                lambda atom: atom.distance_from_muon < distance
                and atom.distance_from_muon != 0.0,
                self._atoms,
            )
        else:
            return filter(
                lambda atom: atom.distance_from_muon < distance
                and atom.distance_from_muon != 0.0
                and atom.symbol not in ignored_symbols,
                self._atoms,
            )

    def get_closest(
        self, number: int, ignored_symbols: Optional[List[str]] = None
    ) -> List[CellAtom]:
        """Returns atoms the closest 'number' atoms to the muon.

        Expects compute_distances to have been called first.

        Arguments:
            number {int} -- Number of closest atoms to return. If number > the
                            number of atoms will only return what was
                            available.
            ignored_symbols {List[str]} -- List of symbols to ignore. May be
                                           None.

        Returns:
            List[CellAtom] -- List of the found closest atoms to the muon
                              (ignoring any in ignored_symbols)
        """

        number = min(number, len(self._atoms))

        ordered_atoms = sorted(self._atoms, key=lambda atom: atom.distance_from_muon)

        # Remove any that should be ignored
        if ignored_symbols is not None:
            ordered_atoms = filter(
                lambda atom: atom.symbol not in ignored_symbols, ordered_atoms
            )

        # Return closest 'number' atoms (ignoring the muon itself)
        return sorted(ordered_atoms, key=lambda atom: atom.distance_from_muon)[
            1 : (number + 1)
        ]

    @property
    def symbols_zero_spin(self) -> List[str]:
        """
        List of symbols that have zero spin
        """
        return self._symbols_zero_spin
