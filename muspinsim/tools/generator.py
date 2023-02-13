from abc import ABC, abstractmethod
import argparse
from dataclasses import dataclass, field
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
        generators {List[InteractionGenerator]} -- List of generators for
                                                generating interaction terms
        number_closest {int} -- Number of closest atoms to the muon to include
        layer_expand_number {int} -- Number of layers to expand to for the
                                     supercell
        muon_symbol {str} -- Symbol for the muon in the loaded files
        additional_ignored_symbols {List[str]} -- List of additional symbols
                                    to ignore when counting the closest atoms.
                                    All spin 0 isotopes will be ignored by
                                    default but currently this data is not
                                    reflected by soprano for Si.
    """

    structure: MuonatedStructure
    generators: List[InteractionGenerator]
    number_closest: int
    layer_expand_number: int = 3
    muon_symbol: str = "H"
    additional_ignored_symbols: List[str] = field(default_factory=list)


def generate_input_file(params: GeneratorToolParams) -> str:
    """Utility function for generating muspinsim input config given a
    structure as input

    Will expand the muonated structure but the amount requested and
    locate

    Arguments:
        params {GeneratorToolParams} -- Parameters (see above for details)
    """

    # Locate closest elements (ignoring ones with zero spin)
    params.structure.layer_expand(params.layer_expand_number)
    params.structure.compute_distances()

    selected_atoms = params.structure.get_closest(
        params.number_closest,
        params.structure.symbols_zero_spin + params.additional_ignored_symbols,
    )

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


def main():
    """Entrypoint for command line tool"""

    parser = argparse.ArgumentParser(
        description="""Generate a MuSpinSim input file using a structure file
(.cell or .cif).""",
    )

    # Required arguments
    parser.add_argument("filepath", type=str, help="Structure file filepath")
    parser.add_argument(
        "number_closest",
        type=int,
        help="Number of nearest atoms to the muon stopping site to include",
    )
    # Optional arguments
    parser.add_argument(
        "--dipolar",
        dest="dipolar",
        action="store_true",
        help="Specify to include dipolar couplings.",
    )
    parser.add_argument(
        "--quadrupolar",
        dest="gipaw_filepath",
        type=str,
        help="""Filepath for GIPAW output data defining the EFGs to be used to
include quadrupolar couplings.""",
    )
    parser.add_argument(
        "--out",
        dest="output",
        type=str,
        help="Filepath for the output file to generate",
    )
    parser.add_argument(
        "--muon_symbol",
        dest="muon_symbol",
        type=str,
        help="Symbol used to identify the muon (default: H)",
        default="H",
    )
    parser.add_argument(
        "--expand_number",
        dest="expand_number",
        type=int,
        help="""Number of layers the structure should be expanded to before
computing distances (default: 3). Larger values may be needed when searching
for many atoms.""",
        default=3,
    )
    parser.add_argument(
        "--ignore_symbol",
        dest="ignored_symbols",
        action="append",
        help="""Allows certain elements in the cell file to be ignored by their
symbol when finding the closest atoms.""",
        default=[],
    )

    args = parser.parse_args()

    # Terms to generate
    generators = []

    if args.dipolar:
        generators.append(DipoleIntGenerator())
    if args.gipaw_filepath is not None:
        generators.append(QuadrupoleIntGenerator(GIPAWOutput(args.gipaw_filepath)))

    generate_params = GeneratorToolParams(
        structure=MuonatedStructure(args.filepath),
        generators=generators,
        number_closest=args.number_closest,
        layer_expand_number=args.expand_number,
        muon_symbol=args.muon_symbol,
        additional_ignored_symbols=args.ignored_symbols,
    )

    file_data = generate_input_file(generate_params)

    if args.output is not None:
        with open(args.output, "w", encoding="utf-8") as file:
            file.write(file_data)
    else:
        print(file_data)
