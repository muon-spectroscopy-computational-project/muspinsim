from dataclasses import dataclass
import re
from typing import IO, List, Optional, Union
from numpy.typing import ArrayLike
import numpy as np


@dataclass
class GIPAWAtom:
    """
    Stores data for an atom from the output
    """

    index: int  # Starts from 1
    efg_tensor: ArrayLike


class GIPAWOutput:
    """
    Parses the output of a GIPAW execution. Currently made specifically
    for parsing the results of an EFG calculation. No guarantees this will
    work.
    """

    _atoms: List[GIPAWAtom]

    def __init__(self, file_io: Union[str, IO]):
        """Load a file containing output from GIPAW

        Reads in EFG tensors from the file.

        Arguments:
            file_io {TextIOBase} -- File path, or IO stream
        """

        self._atoms = []

        if isinstance(file_io, str):
            file_io = open(file_io, encoding="utf_8")

        # Attempt to skip to relevant part of the file
        file_valid = False

        while line := file_io.readline():
            if line.strip().startswith("----- total EFG -----"):
                file_valid = True
                break

        if not file_valid:
            raise ValueError(f"Failed to load file GIPAW output file {file_io}")

        while line := file_io.readline():
            # Should be 3 lines at a time per element e.g.
            #      V    1       -0.008879        0.000000       -0.011803
            #      V    1        0.000000       -0.030597        0.000000
            #      V    1       -0.011803        0.000000        0.039476

            lines = [
                line.strip(),
                file_io.readline().strip(),
                file_io.readline().strip(),
            ]

            atom = self._parse_efg(lines)
            if atom is not None:
                self._atoms.append(atom)
            else:
                break

            # Go to the next line
            file_io.readline()

        file_io.close()

    def _parse_efg(self, lines: List[str]) -> Optional[GIPAWAtom]:
        """Parse an EFG tensor given the 3 lines that represent it e.g.

        V    1       -0.008879        0.000000       -0.011803
        V    1        0.000000       -0.030597        0.000000
        V    1       -0.011803        0.000000        0.039476

        Arguments:
            lines {List[str]} -- Lines representing the above.
        Returns:
            GIPAW_EFG | None -- Structure containing the EFG tensor or None
                                in the case parsing failed and its likely we
                                are at the end of that section.
        Raises:
            ValueError -- When the lines have different indices or symbols.
                          Modifies indices to start from 0 instead of 1.
        """

        identifier = None
        index = None
        efg_tensor = np.zeros((3, 3))

        for i in range(0, 3):
            # Split by at least one space
            split = re.split(r"\s{1,}", lines[i])

            # Return None when invalid i.e. likely finished the section describing
            # the EFG tensors
            if len(split) != 5:
                return None

            current_identifier = split[0]
            try:
                current_index = int(split[1])
            except ValueError:
                return None

            if identifier is None:
                identifier = current_identifier
                index = current_index
            elif current_identifier != identifier or current_index != index:
                raise ValueError("Error while parsing EFG")

            efg_tensor[i, 0] = float(split[2])
            efg_tensor[i, 1] = float(split[3])
            efg_tensor[i, 2] = float(split[4])

        # Start indices from 0
        return GIPAWAtom(index=index, efg_tensor=efg_tensor)

    def find_atom(self, index: int) -> Optional[GIPAWAtom]:
        """Attempt to find an EFG tensor for atom with the given index.

        First assumes the relevant element is at the given index,
        otherwise searches for it.

        Arguments:
            index {int} -- Index of the relevant atom (starting from 1)
        Returns:
            ArrayLike | None -- EFG tensor or None if not found
        """
        # Check if at index (faster)
        if index < len(self._atoms) and self._atoms[index - 1].index == index:
            return self._atoms[index - 1]

        found_atoms = list(
            filter(lambda atom_efg: atom_efg.index == index, self._atoms)
        )
        if found_atoms:
            return found_atoms[0]
        return None
