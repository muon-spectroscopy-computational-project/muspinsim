from abc import ABC, abstractmethod
import argparse
from dataclasses import dataclass, field
import logging
import sys
from typing import Callable, List

from muspinsim.input.structure import CellAtom, MuonatedStructure
from muspinsim.input.gipaw import GIPAWOutput


class InteractionGenerator(ABC):
    """
    Abstract base class for handling the generation of muspinsim interaction
    config
    """

    def __init__(self, num_inputs: int):
        """
        Number of atom indices expected as input for this generator e.g. 1 for
        quadrupole or 2 for dipole
        """
        self.num_inputs = num_inputs

    @abstractmethod
    def gen_config(self, atom_file_indices: List[int], atoms: List[CellAtom]) -> str:
        """Should return the config required for an interaction between the muon
        and given atom using this generator

        Arguments:
            atom_file_indices {List[int]} -- List of file indices for the
                                             atoms given as input (starts at
                                             1)
            atoms {List[CellAtom]} -- Atoms to generate config for
        """


class DipoleIntGenerator(InteractionGenerator):
    """
    Generator for the dipole interaction
    """

    def __init__(self):
        super().__init__(num_inputs=2)

    def gen_config(self, atom_file_indices: List[int], atoms: List[CellAtom]) -> str:
        vector_between = atoms[0].position - atoms[1].position

        return f"""dipolar {atom_file_indices[0]} {atom_file_indices[1]}
    {" ".join(map(str, vector_between))}"""


class QuadrupoleIntGeneratorGIPAWOut(InteractionGenerator):
    """
    Generator for the quadrupole interaction (From a GIPAW output file)
    """

    _gipaw_file: GIPAWOutput

    def __init__(self, gipaw_file: GIPAWOutput):
        super().__init__(num_inputs=1)

        self._gipaw_file = gipaw_file

    def gen_config(self, atom_file_indices: List[int], atoms: List[CellAtom]) -> str:
        # Obtain corresponding EFG tensor
        gipaw_atom = self._gipaw_file.find_atom(atoms[0].index)

        if gipaw_atom is None:
            raise ValueError(
                f"""Unable to locate atom with index {atoms[0].index} in GIPAW
output file"""
            )
        efg_tensor = gipaw_atom.efg_tensor

        return f"""quadrupolar {atom_file_indices[0]}
    {' '.join(map(str, efg_tensor[0]))}
    {' '.join(map(str, efg_tensor[1]))}
    {' '.join(map(str, efg_tensor[2]))}"""


class QuadrupoleIntGenerator(InteractionGenerator):
    """
    Generator for the quadrupole interaction (From data loaded with the
    structure file)
    """

    _structure: MuonatedStructure

    def __init__(self, structure: MuonatedStructure):
        super().__init__(num_inputs=1)

        self._structure = structure

    def gen_config(self, atom_file_indices: List[int], atoms: List[CellAtom]) -> str:
        # Obtain corresponding EFG tensor
        efg_tensor = self._structure.get_efg_tensor(atoms[0].index)

        return f"""quadrupolar {atom_file_indices[0]}
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
        include_interatomic {bool} -- Whether to include interactions between
                                      the atoms themselves (only applied to
                                      generators taking 2 inputs)
        additional_ignored_symbols {List[str]} -- List of additional symbols
                                    to ignore when counting the closest atoms.
                                    All spin 0 isotopes will be ignored by
                                    default but currently this data is not
                                    reflected by soprano for Si.
        max_layer {int} -- Maximum layer to allow the expansion to (for
                           avoiding any long compute times)
    """

    structure: MuonatedStructure
    generators: List[InteractionGenerator]
    number_closest: int
    include_interatomic: bool
    additional_ignored_symbols: List[str] = field(default_factory=list)
    max_layer: int = 6


def generate_input_file_from_selection(
    params: GeneratorToolParams, selected_atoms: List[CellAtom]
) -> str:
    """Utility function for generating muspinsim from a list of selected atoms
    obtained from a structure

    Arguments:
        params {GeneratorToolParams} -- Parameters (see above for details)
        selected_atoms {List[CellAtom]} -- Selected atoms to be included in
                                           the generated config
    Returns:
        input_file_text {str}: Generated muspinsim input file text
    """

    selected_symbols = "mu"
    input_file_text = ""
    muon = params.structure.muon

    # Split up generators based on how many inputs they take
    single_input_generators = list(
        filter(lambda generator: generator.num_inputs == 1, params.generators)
    )
    double_input_generators = list(
        filter(lambda generator: generator.num_inputs == 2, params.generators)
    )

    for i, selected_atom1 in enumerate(selected_atoms):
        selected_atom1_file_index = i + 2

        # Include the symbol into the spin's section
        symbol = selected_atom1.symbol

        if selected_atom1.isotope is not None:
            symbol = f"{str(selected_atom1.isotope)}{symbol}"

        selected_symbols += f" {symbol}"

        # Append config for single generators (using currently selected atom)
        for generator in single_input_generators:
            input_file_text += f"""{
                generator.gen_config(
                    atom_file_indices=[selected_atom1_file_index],
                    atoms=[selected_atom1],
                )
            }\n"""

        # Apply double input generators (with muon as first input)
        for generator in double_input_generators:
            input_file_text += f"""{
                generator.gen_config(
                    atom_file_indices=[1, selected_atom1_file_index],
                    atoms=[muon, selected_atom1],
                )
            }\n"""

        if params.include_interatomic:
            # Apply double terms between the nuclei themselves
            for j, selected_atom2 in enumerate(selected_atoms[i + 1 :], start=i + 1):
                selected_atom2_file_index = j + 2

                input_file_text += f"""{
                    generator.gen_config(
                        atom_file_indices=[
                            selected_atom1_file_index,
                            selected_atom2_file_index
                        ],
                        atoms=[selected_atom1, selected_atom2],
                    )
                }\n"""

    input_file_text = f"""spins
    {selected_symbols}
{input_file_text}"""

    return input_file_text


def _select_atoms(params: GeneratorToolParams) -> List[CellAtom]:
    """Selects atoms from a structure file to be used for generating a
    muspinsim input config

    Will expand a given muonated structure until the specified number
    of nearest neighbours are found

    Arguments:
        params {GeneratorToolParams} -- Parameters (see above for details)

    Returns:
        selected_atoms {List[CellAtom]}: Selected list of atoms
    """

    return params.structure.compute_closest(
        number=params.number_closest,
        ignored_symbols=params.structure.symbols_zero_spin
        + params.additional_ignored_symbols,
        max_layer=params.max_layer,
    )


def _run_generator_tool(
    args: List[str], selection_function: Callable[[GeneratorToolParams], List[CellAtom]]
):
    """Parsed the generator tool arguments from a given list and then
    runs it using the given selection_function to locate the nearest
    neighbours

    Arguments:
        args {List[str]} -- Arguments from the command line
        selection_function {function} -- Function that takes the generator
                                         tool params and returns the selected
                                         atoms to include in the config
                                         produced
    """

    # Setup so we can see the log output
    logging.basicConfig(
        format="[%(levelname)s] [%(asctime)s] %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="""Generate a MuSpinSim input file using a structure file
(e.g. .cell, .cif, .magres).""",
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
        nargs="?",
        const="",  # When specified gives "" unless given value, otherwise None
        action="store",
        help="""Specify to include quadrupolar couplings. If the given
structure file is not a Magres file with EFG data a file path to a GIPAW
output file will be required.""",
    )
    parser.add_argument(
        "--include_interatomic",
        dest="include_interatomic",
        action="store_true",
        help="Specify to include interactions between the atoms themselves.",
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
        "--ignore_symbol",
        dest="ignored_symbols",
        action="append",
        help="""Allows certain elements in the cell file to be ignored by
their symbol when finding the closest atoms.""",
        default=[],
    )
    parser.add_argument(
        "--max_layer",
        dest="max_layer",
        type=int,
        help="""Maximum number of layers the structure can be expanded to
before raising an error (default: 6). Larger values may be needed when
searching for many atoms.""",
        default=6,
    )

    args = parser.parse_args(args)

    # Load the structure file
    structure = MuonatedStructure(args.filepath, muon_symbol=args.muon_symbol)

    # Terms to generate
    generators = []

    if args.dipolar:
        generators.append(DipoleIntGenerator())
    if args.gipaw_filepath is not None:
        # Override if specified
        if args.gipaw_filepath != "":
            generators.append(
                QuadrupoleIntGeneratorGIPAWOut(GIPAWOutput(args.gipaw_filepath))
            )
        else:
            # Attempt to find in the structure file
            if not structure.has_efg_tensors:
                raise ValueError(
                    """No EFGs found in the structure file, and
no GIPAW output file given. Please specify either '--quadrupolar
GIPAW_OUTPUT_FILEPATH' or supply a Magres structure file with EFG's provided."""
                )

            generators.append(QuadrupoleIntGenerator(structure))

    generate_params = GeneratorToolParams(
        structure=structure,
        generators=generators,
        number_closest=args.number_closest,
        include_interatomic=args.include_interatomic,
        additional_ignored_symbols=args.ignored_symbols,
        max_layer=args.max_layer,
    )
    file_data = generate_input_file_from_selection(
        generate_params, selection_function(generate_params)
    )

    if args.output is not None:
        with open(args.output, "w", encoding="utf-8") as file:
            file.write(file_data)
    else:
        print(file_data)


def main():
    """Entrypoint for command line tool"""

    _run_generator_tool(sys.argv[1:], _select_atoms)
