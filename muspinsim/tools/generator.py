from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

from muspinsim.input.structure import CellAtom, MuonatedStructure
from muspinsim.input.gipaw import GIPAWOutput


class InteractionGenerator(ABC):
    """
    Abstract base class for handling the generation of muspinsim interaction
    config
    """

    @abstractmethod
    def gen_config(
        self, muon_file_index: int, atom_file_index: int, atom: CellAtom
    ) -> str:
        """Should return the config required for a given atom using this
        generator

        Arguments:
            muon_file_index {int} -- Index of the muon in the file (starts
                                     from 1)
            atom_file_index {int} -- Index of the muon in the file (starts
                                     from 1)
            atom {CellAtom} -- Atom to generate config for
        """


class DipoleIntGenerator(InteractionGenerator):
    """
    Generator for the dipole interaction
    """

    def gen_config(
        self, muon_file_index: int, atom_file_index: int, atom: CellAtom
    ) -> str:
        return f"""dipolar {muon_file_index} {atom_file_index}
    {" ".join(map(str, atom.vector_from_muon))}"""


class QuadrupoleIntGenerator(InteractionGenerator):
    """
    Generator for the dipole interaction
    """

    _gipaw_file: GIPAWOutput

    def __init__(self, gipaw_file: GIPAWOutput):
        self._gipaw_file = gipaw_file

    def gen_config(
        self, muon_file_index: int, atom_file_index: int, atom: CellAtom
    ) -> str:
        # Obtain corresponding EFG tensor
        gipaw_atom = self._gipaw_file.find_atom(atom.index)

        if gipaw_atom is None:
            raise ValueError(
                f"Unable to locate atom with index {atom.index} in GIPAW output file"
            )
        efg_tensor = gipaw_atom.efg_tensor

        return f"""quadrupolar {atom_file_index}
    {' '.join(map(str, efg_tensor[0]))}
    {' '.join(map(str, efg_tensor[1]))}
    {' '.join(map(str, efg_tensor[2]))}"""


@dataclass
class GeneratorToolParams:
    """Structure for storing parameters for the interaction generator tool

    Arguments:
        structure {MuonatedStructure} -- Structure to compute terms from
        layer_expand_number {int} -- Number of layers to expand to for the
                                     supercell
        number_closest {int} -- Number of closest atoms to the muon to include
        muon_symbol {str} -- Symbol for the muon in the loaded files
    """

    structure: MuonatedStructure
    generators: List[InteractionGenerator]
    number_closest: int
    layer_expand_number: int = 3
    muon_symbol: str = "H"


def generate_input_file(params: GeneratorToolParams) -> str:
    """Utility function for generating muspinsim input config given a
    structure as input

    Will expand the muonated structure but the amount requested and
    locate

    Arguments:
        params {GeneratorToolParams} -- Parameters (see above for details)
    """

    # Locate closest elements
    params.structure.layer_expand(params.layer_expand_number)
    params.structure.compute_distances()

    selected_atoms = params.structure.get_closest(params.number_closest)

    # Generate input file
    selected_symbols = "mu"
    muon_file_index = 1
    atom_file_index = 2

    input_file = ""

    for selected_atom in selected_atoms:
        symbol = selected_atom.symbol

        if selected_atom.isotope != 1:
            symbol = f"{str(selected_atom.isotope)} {symbol}"

        selected_symbols += f" {symbol}"

        for generator in params.generators:
            input_file += f"""{
                generator.gen_config(
                    muon_file_index,
                    atom_file_index,
                    selected_atom
                )
            }\n"""

        atom_file_index += 1

    input_file = f"""spins
    {selected_symbols}
{input_file}"""

    return input_file
