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
    _cell_atoms: List[CellAtom]

    _muon_index: int

    _symbols_zero_spin: List[str]

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

        self._cell_atoms = []
        self._muon_index = None
        self._symbols_zero_spin = []

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

        self._cell_atoms = [None] * len(loaded_atoms)

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

            self._cell_atoms[i] = CellAtom(
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

    def expand(
        self, offsets: List[ArrayLike], ignored_symbols: Optional[List[str]] = None
    ) -> List[CellAtom]:
        """Expands the structure by duplicating unit cell atoms (ignoring the
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
            for i, atom in enumerate(self._cell_atoms):
                if i != self._muon_index and (
                    ignored_symbols is None or atom.symbol not in ignored_symbols
                ):
                    new_atoms.append(
                        CellAtom(
                            index=atom.index,
                            symbol=atom.symbol,
                            isotope=atom.isotope,
                            position=atom.position + offset,
                        )
                    )
        return new_atoms

    def _compute_layer_offsets(self, layer: int) -> List[ArrayLike]:
        """Computes offsets for expanding this structure outwards

        Arguments:
            layer {int} -- Index of layer e.g. if layer = 1, will produce 8
                           offsets along the x, y and z axes. layer = 2 will
                           then produce another 26 offsets for expanding
                           another layer outwards.
        Returns:
            offsets {List[ArrayLike]} -- List of computed offsets
        """

        # Compute all potential values for the x, y and z axes
        potential_values = [
            np.arange(
                -layer * self._unit_lengths[i],
                (layer + 1) * self._unit_lengths[i],
                self._unit_lengths[i],
            )
            for i in range(3)
        ]

        # Obtain the extremes and values in between for each axis
        extreme_values = [
            [potential_values[i][0], potential_values[i][-1]] for i in range(3)
        ]
        middle_values = [potential_values[i][1:-1:] for i in range(3)]

        # Compute combinations
        offsets = []

        # Hold z fixed at one of the extreme values and vary others
        for x_value in extreme_values[0]:
            for y_value in potential_values[1]:
                for z_value in potential_values[2]:
                    offsets.append([x_value, y_value, z_value])

        # Middle values for x
        for x_value in middle_values[0]:
            # Keep y at the extremes
            for y_value in extreme_values[1]:
                for z_value in potential_values[2]:
                    offsets.append([x_value, y_value, z_value])

            # Middle values for y, with z at extremes
            for y_value in middle_values[1]:
                for z_value in extreme_values[2]:
                    offsets.append([x_value, y_value, z_value])

        return offsets

    def compute_layer(
        self, layer: int, ignored_symbols: Optional[List[str]] = None
    ) -> List[CellAtom]:
        """Duplicates atoms from the loaded cell by expanding outwards.

        Expands the structure outwards by duplicating it for a given layer
        number.

        Arguments:
            layer {int} -- Index of layer e.g. if layer = 1, will expand
                           along the x, y and z axes so that returned atoms
                           will have been computed for the 26 surrounding
                           cells. layer = 2 will then add compute those in
                           the next layer, which will include 96 cells etc.
            ignored_symbols {List[str]} -- List of symbols to ignore. May be
                                           None.
        Returns:
            new_atoms {List[CellAtom]} -- Additional atoms with their new
                                          positions.
        """

        # Now expand for each computed offset
        return self.expand(self._compute_layer_offsets(layer), ignored_symbols)

    def _compute_distances(self, atoms: List[CellAtom]):
        """Computes the vectors and distances between the muon and every atom."""
        muon = self._cell_atoms[self._muon_index]

        for atom in atoms:
            atom.vector_from_muon = muon.position - atom.position
            atom.distance_from_muon = np.linalg.norm(atom.vector_from_muon)

    def compute_closest(
        self,
        number: int,
        ignored_symbols: Optional[List[str]] = None,
        max_layer: int = 6,
    ) -> List[CellAtom]:
        """Returns atoms the closest 'number' atoms to the muon.

        Expands structure outwards and computes distances until we are sure
        we have found the desired number of closest atoms to the muon.

        Arguments:
            number {int} -- Number of closest atoms to return. If number > the
                            number of atoms will only return what was
                            available.
            ignored_symbols {List[str]} -- List of symbols to ignore. May be
                                           None.
            max_layer {int} -- Maximum layer to allow the expansion to (for
                               avoiding any accidental long compute times)

        Returns:
            List[CellAtom] -- List of the found closest atoms to the muon
                              (ignoring any in ignored_symbols)

        Raises:
            RuntimeError -- When the expansion tries to go beyond the maximum
                            limit given by max_layer.
        """

        # Current list of atoms in expanded supercell
        atoms = self._cell_atoms
        if ignored_symbols:
            atoms = list(filter(lambda atom: atom.symbol not in ignored_symbols, atoms))

        # Compute the distances within the cell
        self._compute_distances(atoms)

        continue_expansion = True
        layer = 1

        while continue_expansion:
            # Sort by distance and obtain furthest distance of the desired
            # atom
            atoms = sorted(atoms, key=lambda atom: atom.distance_from_muon)
            furthest_atom = atoms[min(number, len(atoms) - 1)]

            # Compute another layer and find the closest
            new_atoms = self.compute_layer(layer, ignored_symbols)
            self._compute_distances(new_atoms)
            new_atoms = sorted(new_atoms, key=lambda atom: atom.distance_from_muon)

            closest_new_atom = new_atoms[0]

            # Check whether we have just found atoms closer than the furthest
            # previous one or we dont have enough atoms in the structure
            if (
                closest_new_atom.distance_from_muon < furthest_atom.distance_from_muon
                or number + 1 > len(atoms)
            ):
                # Need this expansion to be included (and run at least
                # once more to check if there are any more to include)
                atoms.extend(new_atoms)

                layer += 1

                # Avoid taking forever
                if layer > max_layer:
                    raise RuntimeError("Trying to expand past the max_layer value")
            else:
                # No more expansion needed
                continue_expansion = False

        # Return closest 'number' atoms (ignoring the muon itself)
        return atoms[1 : (number + 1)]

    @property
    def symbols_zero_spin(self) -> List[str]:
        """
        List of symbols that have zero spin
        """
        return self._symbols_zero_spin
