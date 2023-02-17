from io import StringIO
import unittest
from unittest.mock import ANY, patch

from muspinsim.input.gipaw import GIPAWOutput
from muspinsim.input.structure import MuonatedStructure
from muspinsim.tools.generator import (
    DipoleIntGenerator,
    GeneratorToolParams,
    QuadrupoleIntGenerator,
    _run_generator_tool,
    generate_input_file,
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


class TestGenerator(unittest.TestCase):
    def test_generate_input_file(self):
        # No ignored symbols
        generate_params = GeneratorToolParams(
            structure=MuonatedStructure(
                StringIO(TEST_CELL_FILE_DATA), fmt="castep-cell"
            ),
            generators=[
                DipoleIntGenerator(),
                QuadrupoleIntGenerator(GIPAWOutput(StringIO(TEST_GIPAW_FILE_DATA))),
            ],
            number_closest=4,
            additional_ignored_symbols=[],
        )

        input_file_text = generate_input_file(generate_params)
        self.assertEqual(
            input_file_text,
            """spins
    mu V V Si Si
dipolar 1 2
    1.2410539563139198 1.7614965359999998e-05 -1.2417021305964
quadrupolar 2
    -0.008877 2.2e-05 -0.011811
    2.2e-05 -0.030585 5e-06
    -0.011811 5e-06 0.039462
dipolar 1 3
    3.5524979205813603 1.7657510159999996e-05 3.5562763937592
quadrupolar 3
    0.221597 -2e-06 -0.009013
    -2e-06 -0.109627 -9e-06
    -0.009013 -9e-06 -0.11197
dipolar 1 4
    -4.72192610499264 2.366426252954879 3.54776669347968
quadrupolar 4
    -0.002604 -0.0 -1e-06
    -0.0 -0.002551 -9e-06
    -1e-06 -9e-06 0.005154
dipolar 1 5
    9.45967389500736 2.366426252954879 3.54776669347968
quadrupolar 5
    -0.002604 -0.0 -1e-06
    -0.0 -0.002551 -9e-06
    -1e-06 -9e-06 0.005154
""",
        )

        # Ignore Si
        generate_params.additional_ignored_symbols = ["Si"]

        input_file_text = generate_input_file(generate_params)
        self.assertEqual(
            input_file_text,
            """spins
    mu V V V V
dipolar 1 2
    1.2410539563139198 1.7614965359999998e-05 -1.2417021305964
quadrupolar 2
    -0.008877 2.2e-05 -0.011811
    2.2e-05 -0.030585 5e-06
    -0.011811 5e-06 0.039462
dipolar 1 3
    3.5524979205813603 1.7657510159999996e-05 3.5562763937592
quadrupolar 3
    0.221597 -2e-06 -0.009013
    -2e-06 -0.109627 -9e-06
    -0.009013 -9e-06 -0.11197
dipolar 1 4
    3.5524979205813603 1.7657510159999996e-05 -10.6253236062408
quadrupolar 4
    0.221597 -2e-06 -0.009013
    -2e-06 -0.109627 -9e-06
    -0.009013 -9e-06 -0.11197
dipolar 1 5
    -10.62910207941864 1.7657510159999996e-05 3.5562763937592
quadrupolar 5
    0.221597 -2e-06 -0.009013
    -2e-06 -0.109627 -9e-06
    -0.009013 -9e-06 -0.11197
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
                QuadrupoleIntGenerator(
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
            additional_ignored_symbols=[],
        )

        with self.assertRaises(ValueError):
            generate_input_file(generate_params)

    # Mock these so dont actually do anything, just want to check parameters
    # used properly
    @patch("muspinsim.tools.generator.MuonatedStructure")
    @patch("muspinsim.tools.generator.generate_input_file")
    def test_generate_tool_basic(
        self, mock_generate_input_file, mock_MuonatedStructure
    ):
        _run_generator_tool(
            [
                "V3Si_SC.cell",
                "4",
                "--dipolar",
            ]
        )
        mock_MuonatedStructure.assert_called_with("V3Si_SC.cell", muon_symbol="H")
        mock_generate_input_file.assert_called_with(
            GeneratorToolParams(
                structure=mock_MuonatedStructure.return_value,
                generators=[ANY],
                number_closest=4,
                additional_ignored_symbols=[],
                max_layer=6,
            )
        )

    @patch("muspinsim.tools.generator.GIPAWOutput")
    @patch("muspinsim.tools.generator.MuonatedStructure")
    @patch("muspinsim.tools.generator.generate_input_file")
    def test_generate_tool(
        self, mock_generate_input_file, mock_MuonatedStructure, mock_GIPAWOutput
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
            ]
        )
        mock_MuonatedStructure.assert_called_with("V3Si_SC.cell", muon_symbol="Cu")
        mock_GIPAWOutput.assert_called_with("./efg_nospin.out")
        mock_generate_input_file.assert_called_with(
            GeneratorToolParams(
                structure=mock_MuonatedStructure.return_value,
                generators=[ANY, ANY],
                number_closest=6,
                additional_ignored_symbols=["Si", "F"],
                max_layer=3,
            )
        )
