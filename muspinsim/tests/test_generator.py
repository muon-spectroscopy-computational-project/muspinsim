from io import StringIO
import unittest
from unittest.mock import ANY, patch

from muspinsim.input.gipaw import GIPAWOutput
from muspinsim.input.structure import MuonatedStructure
from muspinsim.tools.generator import (
    DipoleIntGenerator,
    GeneratorToolParams,
    QuadrupoleIntGenerator,
    QuadrupoleIntGeneratorGIPAWOut,
    _run_generator_tool,
    _select_atoms,
    generate_input_file_from_selection,
)

TEST_CELL_FILE_DATA = """%BLOCK LATTICE_CART
   14.18160000000000   0.000000000000000   0.000000000000000
   0.000000000000000   14.18160000000000   0.000000000000000
   0.000000000000000   0.000000000000000   14.18160000000000
%ENDBLOCK LATTICE_CART

%BLOCK POSITIONS_FRAC
V             0.0791557133        0.0000005853        0.1708904764
V             0.9161667724        0.0000005823        0.8325662729
Si            0.4996287249        0.8331358706        0.8331663251
H             0.1666672745        0.0000018274        0.0833332099
%ENDBLOCK POSITIONS_FRAC
"""

TEST_GIPAW_FILE_DATA = """     ----- total EFG -----
     V    1       -0.008877        0.000022       -0.011811
     V    1        0.000022       -0.030585        0.000005
     V    1       -0.011811        0.000005        0.039462

     V    2        0.221597       -0.000002       -0.009013
     V    2       -0.000002       -0.109627       -0.000009
     V    2       -0.009013       -0.000009       -0.111970

     Si   3       -0.002604       -0.000000       -0.000001
     Si   3       -0.000000       -0.002551       -0.000009
     Si   3       -0.000001       -0.000009        0.005154

     H    4       -0.034266        0.000000        0.000000
     H    4        0.000000       -0.034268        0.000000
     H    4        0.000000        0.000000        0.068534
"""

TEST_MAGRES_FILE_DATA = (
    "#$magres-abinitio-v1.0\n"
    "[atoms]\n"
    "units lattice Angstrom\n"
    "lattice          14.18160000000000   0.000000000000000   0.000000000000000"
    "                 0.000000000000000   14.18160000000000   0.000000000000000"
    "                 0.000000000000000   0.000000000000000   14.18160000000000\n"
    "units atom Angstrom\n"
    "atom V       V   1"
    "     1.1225546637        0.0000083005        2.4235003801\n"
    "atom V       V   2"
    "     12.992710700        0.0000082580        11.807121856\n"
    "atom Si      Si  3"
    "     7.0855347250        11.815199663        11.815631556\n"
    "atom H:mu    H:mu   1"
    "     2.3636086201        0.0000259155        1.1817982368\n"
    "[/atoms]\n"
    "[magres]\n"
    "units efg au\n"
    "efg V  1         -0.008877        0.000022       -0.011811"
    "                  0.000022       -0.030585        0.000005"
    "                 -0.011811        0.000005        0.039462\n"
    "efg V  2          0.221597       -0.000002       -0.009013"
    "                 -0.000002       -0.109627       -0.000009"
    "                 -0.009013       -0.000009       -0.111970\n"
    "efg Si 3         -0.002604       -0.000000       -0.000001"
    "                 -0.000000       -0.002551       -0.000009"
    "                 -0.000001       -0.000009        0.005154\n"
    "efg H:mu  1      -0.034266        0.000000        0.000000"
    "                  0.000000       -0.034268        0.000000"
    "                  0.000000        0.000000        0.068534\n"
    "[/magres]\n"
)


class TestGenerator(unittest.TestCase):
    def test_generate_input_file_cell(self):
        # No ignored symbols
        generate_params = GeneratorToolParams(
            structure=MuonatedStructure(
                StringIO(TEST_CELL_FILE_DATA), fmt="castep-cell"
            ),
            generators=[
                DipoleIntGenerator(),
                QuadrupoleIntGeneratorGIPAWOut(
                    GIPAWOutput(StringIO(TEST_GIPAW_FILE_DATA))
                ),
            ],
            number_closest=4,
            include_interatomic=False,
            additional_ignored_symbols=[],
        )

        input_file_text = generate_input_file_from_selection(
            generate_params, _select_atoms(generate_params)
        )
        self.assertEqual(
            input_file_text,
            """spins
    mu V V Si Si
quadrupolar 2
    -0.008877 2.2e-05 -0.011811
    2.2e-05 -0.030585 5e-06
    -0.011811 5e-06 0.039462
dipolar 1 2
    1.2410539563139198 1.7614965359999998e-05 -1.2417021305964
quadrupolar 3
    0.221597 -2e-06 -0.009013
    -2e-06 -0.109627 -9e-06
    -0.009013 -9e-06 -0.11197
dipolar 1 3
    3.5524979205813603 1.7657510159999996e-05 3.5562763937592
quadrupolar 4
    -0.002604 -0.0 -1e-06
    -0.0 -0.002551 -9e-06
    -1e-06 -9e-06 0.005154
dipolar 1 4
    -4.72192610499264 2.366426252954879 3.54776669347968
quadrupolar 5
    -0.002604 -0.0 -1e-06
    -0.0 -0.002551 -9e-06
    -1e-06 -9e-06 0.005154
dipolar 1 5
    9.45967389500736 2.366426252954879 3.54776669347968
""",
        )

        # Ignore Si
        generate_params.additional_ignored_symbols = ["Si"]

        input_file_text = generate_input_file_from_selection(
            generate_params, _select_atoms(generate_params)
        )
        self.assertEqual(
            input_file_text,
            """spins
    mu V V V V
quadrupolar 2
    -0.008877 2.2e-05 -0.011811
    2.2e-05 -0.030585 5e-06
    -0.011811 5e-06 0.039462
dipolar 1 2
    1.2410539563139198 1.7614965359999998e-05 -1.2417021305964
quadrupolar 3
    0.221597 -2e-06 -0.009013
    -2e-06 -0.109627 -9e-06
    -0.009013 -9e-06 -0.11197
dipolar 1 3
    3.5524979205813603 1.7657510159999996e-05 3.5562763937592
quadrupolar 4
    0.221597 -2e-06 -0.009013
    -2e-06 -0.109627 -9e-06
    -0.009013 -9e-06 -0.11197
dipolar 1 4
    3.5524979205813603 1.7657510159999996e-05 -10.6253236062408
quadrupolar 5
    0.221597 -2e-06 -0.009013
    -2e-06 -0.109627 -9e-06
    -0.009013 -9e-06 -0.11197
dipolar 1 5
    -10.62910207941864 1.7657510159999996e-05 3.5562763937592
""",
        )

        # Include interatomic interactions
        generate_params.include_interatomic = True
        input_file_text = generate_input_file_from_selection(
            generate_params, _select_atoms(generate_params)
        )
        self.assertEqual(
            input_file_text,
            """spins
    mu V V V V
quadrupolar 2
    -0.008877 2.2e-05 -0.011811
    2.2e-05 -0.030585 5e-06
    -0.011811 5e-06 0.039462
dipolar 1 2
    1.2410539563139198 1.7614965359999998e-05 -1.2417021305964
dipolar 2 3
    2.3114439642674407 4.25447999999999e-08 4.7979785243555995
dipolar 2 4
    2.3114439642674407 4.25447999999999e-08 -9.3836214756444
dipolar 2 5
    -11.87015603573256 4.25447999999999e-08 4.7979785243555995
quadrupolar 3
    0.221597 -2e-06 -0.009013
    -2e-06 -0.109627 -9e-06
    -0.009013 -9e-06 -0.11197
dipolar 1 3
    3.5524979205813603 1.7657510159999996e-05 3.5562763937592
dipolar 3 4
    0.0 0.0 -14.1816
dipolar 3 5
    -14.1816 0.0 0.0
quadrupolar 4
    0.221597 -2e-06 -0.009013
    -2e-06 -0.109627 -9e-06
    -0.009013 -9e-06 -0.11197
dipolar 1 4
    3.5524979205813603 1.7657510159999996e-05 -10.6253236062408
dipolar 4 5
    -14.1816 0.0 14.1816
quadrupolar 5
    0.221597 -2e-06 -0.009013
    -2e-06 -0.109627 -9e-06
    -0.009013 -9e-06 -0.11197
dipolar 1 5
    -10.62910207941864 1.7657510159999996e-05 3.5562763937592
""",
        )

    def test_generate_input_file_magres(self):
        # No ignored symbols
        structure = MuonatedStructure(StringIO(TEST_MAGRES_FILE_DATA), fmt="magres")
        generate_params = GeneratorToolParams(
            structure=structure,
            generators=[
                DipoleIntGenerator(),
                QuadrupoleIntGenerator(structure),
            ],
            number_closest=4,
            include_interatomic=False,
            additional_ignored_symbols=[],
        )

        input_file_text = generate_input_file_from_selection(
            generate_params, _select_atoms(generate_params)
        )
        self.assertEqual(
            input_file_text,
            """spins
    mu V V Si Si
quadrupolar 2
    -0.008877 2.2e-05 -0.011811
    2.2e-05 -0.030585 5e-06
    -0.011811 5e-06 0.039462
dipolar 1 2
    1.2410539564 1.7615e-05 -1.2417021433000002
quadrupolar 3
    0.221597 -2e-06 -0.009013
    -2e-06 -0.109627 -9e-06
    -0.009013 -9e-06 -0.11197
dipolar 1 3
    3.5524979200999995 1.7657499999999998e-05 3.556276380799999
quadrupolar 4
    -0.002604 -0.0 -1e-06
    -0.0 -0.002551 -9e-06
    -1e-06 -9e-06 0.005154
dipolar 1 4
    -4.7219261049 2.3664262525 3.5477666807999997
quadrupolar 5
    -0.002604 -0.0 -1e-06
    -0.0 -0.002551 -9e-06
    -1e-06 -9e-06 0.005154
dipolar 1 5
    9.4596738951 2.3664262525 3.5477666807999997
""",
        )

        # Ignore Si
        generate_params.additional_ignored_symbols = ["Si"]

        input_file_text = generate_input_file_from_selection(
            generate_params, _select_atoms(generate_params)
        )
        self.assertEqual(
            input_file_text,
            """spins
    mu V V V V
quadrupolar 2
    -0.008877 2.2e-05 -0.011811
    2.2e-05 -0.030585 5e-06
    -0.011811 5e-06 0.039462
dipolar 1 2
    1.2410539564 1.7615e-05 -1.2417021433000002
quadrupolar 3
    0.221597 -2e-06 -0.009013
    -2e-06 -0.109627 -9e-06
    -0.009013 -9e-06 -0.11197
dipolar 1 3
    3.5524979200999995 1.7657499999999998e-05 3.556276380799999
quadrupolar 4
    0.221597 -2e-06 -0.009013
    -2e-06 -0.109627 -9e-06
    -0.009013 -9e-06 -0.11197
dipolar 1 4
    3.5524979200999995 1.7657499999999998e-05 -10.6253236192
quadrupolar 5
    0.221597 -2e-06 -0.009013
    -2e-06 -0.109627 -9e-06
    -0.009013 -9e-06 -0.11197
dipolar 1 5
    -10.6291020799 1.7657499999999998e-05 3.556276380799999
""",
        )

        # Include interatomic interactions
        generate_params.include_interatomic = True
        input_file_text = generate_input_file_from_selection(
            generate_params, _select_atoms(generate_params)
        )
        self.assertEqual(
            input_file_text,
            """spins
    mu V V V V
quadrupolar 2
    -0.008877 2.2e-05 -0.011811
    2.2e-05 -0.030585 5e-06
    -0.011811 5e-06 0.039462
dipolar 1 2
    1.2410539564 1.7615e-05 -1.2417021433000002
dipolar 2 3
    2.3114439636999995 4.2500000000000017e-08 4.7979785240999995
dipolar 2 4
    2.3114439636999995 4.2500000000000017e-08 -9.3836214759
dipolar 2 5
    -11.8701560363 4.2500000000000017e-08 4.7979785240999995
quadrupolar 3
    0.221597 -2e-06 -0.009013
    -2e-06 -0.109627 -9e-06
    -0.009013 -9e-06 -0.11197
dipolar 1 3
    3.5524979200999995 1.7657499999999998e-05 3.556276380799999
dipolar 3 4
    0.0 0.0 -14.1816
dipolar 3 5
    -14.1816 0.0 0.0
quadrupolar 4
    0.221597 -2e-06 -0.009013
    -2e-06 -0.109627 -9e-06
    -0.009013 -9e-06 -0.11197
dipolar 1 4
    3.5524979200999995 1.7657499999999998e-05 -10.6253236192
dipolar 4 5
    -14.1816 0.0 14.1816
quadrupolar 5
    0.221597 -2e-06 -0.009013
    -2e-06 -0.109627 -9e-06
    -0.009013 -9e-06 -0.11197
dipolar 1 5
    -10.6291020799 1.7657499999999998e-05 3.556276380799999
""",
        )

    def test_generate_input_error(self):
        # Should fail as indices in GIPAW file wont align
        generate_params = GeneratorToolParams(
            structure=MuonatedStructure(
                StringIO(TEST_CELL_FILE_DATA), fmt="castep-cell"
            ),
            generators=[
                DipoleIntGenerator(),
                QuadrupoleIntGeneratorGIPAWOut(
                    GIPAWOutput(
                        StringIO(
                            """     ----- total EFG -----
     V    1       -0.008877        0.000022       -0.011811
     V    1        0.000022       -0.030585        0.000005
     V    1       -0.011811        0.000005        0.039462

     V    2        0.221597       -0.000002       -0.009013
     V    2       -0.000002       -0.109627       -0.000009
     V    2       -0.009013       -0.000009       -0.111970

     Si 168       -0.002604       -0.000000       -0.000001
     Si 168       -0.000000       -0.002551       -0.000009
     Si 168       -0.000001       -0.000009        0.005154

     H  217       -0.034266        0.000000        0.000000
     H  217        0.000000       -0.034268        0.000000
     H  217        0.000000        0.000000        0.068534
"""
                        )
                    )
                ),
            ],
            number_closest=4,
            include_interatomic=False,
            additional_ignored_symbols=[],
        )

        with self.assertRaises(ValueError):
            generate_input_file_from_selection(
                generate_params, _select_atoms(generate_params)
            )

    # Mock these so dont actually do anything, just want to check parameters
    # used properly
    @patch("muspinsim.tools.generator.MuonatedStructure")
    @patch("muspinsim.tools.generator.generate_input_file_from_selection")
    def test_generate_tool_basic(
        self, generate_input_file_from_selection, mock_MuonatedStructure
    ):
        _run_generator_tool(
            [
                "V3Si_SC.cell",
                "4",
                "--dipolar",
            ],
            _select_atoms,
        )
        mock_MuonatedStructure.assert_called_with("V3Si_SC.cell", muon_symbol="H")
        generate_input_file_from_selection.assert_called_with(
            GeneratorToolParams(
                structure=mock_MuonatedStructure.return_value,
                generators=[ANY],
                number_closest=4,
                include_interatomic=False,
                additional_ignored_symbols=[],
                max_layer=6,
            ),
            mock_MuonatedStructure.return_value.compute_closest.return_value,
        )

    @patch("muspinsim.tools.generator.GIPAWOutput")
    @patch("muspinsim.tools.generator.MuonatedStructure")
    @patch("muspinsim.tools.generator.generate_input_file_from_selection")
    def test_generate_tool_gipaw(
        self,
        mock_generate_input_file_from_selection,
        mock_MuonatedStructure,
        mock_GIPAWOutput,
    ):
        _run_generator_tool(
            [
                "V3Si_SC.cell",
                "6",
                "--dipolar",
                "--quadrupolar",
                "./efg_nospin.out",
                "--ignore_symbol",
                "Si",
                "--ignore_symbol",
                "F",
                "--max_layer",
                "3",
                "--muon_symbol",
                "Cu",
            ],
            _select_atoms,
        )
        mock_MuonatedStructure.assert_called_with("V3Si_SC.cell", muon_symbol="Cu")
        mock_GIPAWOutput.assert_called_with("./efg_nospin.out")
        mock_generate_input_file_from_selection.assert_called_with(
            GeneratorToolParams(
                structure=mock_MuonatedStructure.return_value,
                generators=[ANY, ANY],
                number_closest=6,
                include_interatomic=False,
                additional_ignored_symbols=["Si", "F"],
                max_layer=3,
            ),
            mock_MuonatedStructure.return_value.compute_closest.return_value,
        )

    @patch("muspinsim.tools.generator.GIPAWOutput")
    @patch("muspinsim.tools.generator.MuonatedStructure")
    @patch("muspinsim.tools.generator.generate_input_file_from_selection")
    def test_generate_tool_invalid_quadrupolar(
        self,
        mock_generate_input_file_from_selection,
        mock_MuonatedStructure,
        mock_GIPAWOutput,
    ):
        # Don't supply a GIPAW file path while not using a magres file with
        # EFG tensors
        structure = MuonatedStructure(StringIO(TEST_CELL_FILE_DATA), fmt="castep-cell")
        mock_MuonatedStructure.return_value = structure
        with self.assertRaises(ValueError):
            _run_generator_tool(
                [
                    "V3Si_SC.cell",
                    "6",
                    "--dipolar",
                    "--quadrupolar",
                ],
                _select_atoms,
            )
        mock_MuonatedStructure.assert_called_with("V3Si_SC.cell", muon_symbol="H")

    @patch("muspinsim.tools.generator.GIPAWOutput")
    @patch("muspinsim.tools.generator.MuonatedStructure")
    @patch("muspinsim.tools.generator.generate_input_file_from_selection")
    def test_generate_tool_magres(
        self,
        mock_generate_input_file_from_selection,
        mock_MuonatedStructure,
        mock_GIPAWOutput,
    ):
        # Don't supply a GIPAW file path while using a magres file with EFG
        # tensors
        structure = MuonatedStructure(StringIO(TEST_MAGRES_FILE_DATA), fmt="magres")
        mock_MuonatedStructure.return_value = structure
        _run_generator_tool(
            [
                "V3Si_SC.cell",
                "6",
                "--dipolar",
                "--quadrupolar",
            ],
            _select_atoms,
        )
        mock_MuonatedStructure.assert_called_with("V3Si_SC.cell", muon_symbol="H")
        mock_generate_input_file_from_selection.assert_called_with(
            GeneratorToolParams(
                structure=mock_MuonatedStructure.return_value,
                generators=[ANY, ANY],
                number_closest=6,
                include_interatomic=False,
                additional_ignored_symbols=[],
                max_layer=6,
            ),
            ANY,
        )

    @patch("muspinsim.tools.generator.MuonatedStructure")
    @patch("muspinsim.tools.generator.generate_input_file_from_selection")
    def test_generate_tool_include_interatomic(
        self, mock_generate_input_file_from_selection, mock_MuonatedStructure
    ):
        _run_generator_tool(
            ["V3Si_SC.cell", "4", "--dipolar", "--include_interatomic"], _select_atoms
        )
        mock_MuonatedStructure.assert_called_with("V3Si_SC.cell", muon_symbol="H")
        mock_generate_input_file_from_selection.assert_called_with(
            GeneratorToolParams(
                structure=mock_MuonatedStructure.return_value,
                generators=[ANY],
                number_closest=4,
                include_interatomic=True,
                additional_ignored_symbols=[],
                max_layer=6,
            ),
            mock_MuonatedStructure.return_value.compute_closest.return_value,
        )
