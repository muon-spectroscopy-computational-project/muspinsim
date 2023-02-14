from io import StringIO
import unittest
import numpy as np

from muspinsim.input.structure import CellAtom, MuonatedStructure

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


def _optional_array_close(value1, value2):
    # Compares two values that may be an array or none
    if value1 is None or value2 is None:
        return value1 == value2
    else:
        return np.allclose(value1, value2)


class CellAtomMatcher:
    expected: CellAtom

    def __init__(self, expected):
        self.expected = expected

    def __repr__(self):
        return repr(self.expected)

    def __eq__(self, other):
        return (
            self.expected.index == other.index
            and self.expected.symbol == other.symbol
            and _optional_array_close(self.expected.position, other.position)
            and _optional_array_close(
                self.expected.vector_from_muon, other.vector_from_muon
            )
            and np.isclose(self.expected.distance_from_muon, other.distance_from_muon)
        )


class TestStructure(unittest.TestCase):
    def test_parsing(self):
        structure = MuonatedStructure(StringIO(TEST_CELL_FILE_DATA), fmt="castep-cell")

        self.assertEqual(len(structure._atoms), 4)
        self.assertEqual(structure._muon_index, 3)
        self.assertEqual(
            CellAtomMatcher(structure._atoms[2]),
            CellAtom(
                index=3,
                symbol="Si",
                isotope=1,
                position=np.array([7.08553473, 11.81519966, 11.81563156]),
                vector_from_muon=None,
                distance_from_muon=None,
            ),
        )

    def test_invalid_vector_angels(self):
        with self.assertRaises(ValueError):
            MuonatedStructure(
                StringIO(
                    """%BLOCK LATTICE_CART
   14.18160000000000   14.18160000000000   0.000000000000000
   0.000000000000000   14.18160000000000   0.000000000000000
   0.000000000000000   0.000000000000000   14.18160000000000
%ENDBLOCK LATTICE_CART

%BLOCK POSITIONS_FRAC
H             0.1666672745        0.0000018274        0.0833332099
%ENDBLOCK POSITIONS_FRAC"""
                ),
                fmt="castep-cell",
            )

    def test_no_muon_found(self):
        with self.assertRaises(ValueError):
            MuonatedStructure(
                StringIO(TEST_CELL_FILE_DATA),
                muon_symbol="Cu",
                fmt="castep-cell",
            )

    def test_multiple_muons_found(self):
        with self.assertRaises(ValueError):
            MuonatedStructure(
                StringIO(TEST_CELL_FILE_DATA),
                muon_symbol="V",
                fmt="castep-cell",
            )

    def test_layer_expand(self):
        structure = MuonatedStructure(StringIO(TEST_CELL_FILE_DATA), fmt="castep-cell")
        structure.layer_expand(1)

        self.assertEqual(len(structure._atoms), 1 + 3 * 27)  # Copy everything but muon
        self.assertEqual(structure._muon_index, 3)
        self.assertEqual(
            CellAtomMatcher(structure._atoms[11]),
            CellAtom(
                index=2,
                symbol="V",
                isotope=1,
                position=np.array([-1.1888893, 14.18160826, -2.37447814]),
                vector_from_muon=None,
                distance_from_muon=None,
            ),
        )

    def test_compute_distances(self):
        structure = MuonatedStructure(StringIO(TEST_CELL_FILE_DATA), fmt="castep-cell")
        structure.compute_distances()

        self.assertEqual(
            CellAtomMatcher(structure._atoms[2]),
            CellAtom(
                index=3,
                symbol="Si",
                isotope=1,
                position=np.array([7.08553473, 11.81519966, 11.81563156]),
                vector_from_muon=np.array([-4.7219261, -11.81517375, -10.63383331]),
                distance_from_muon=16.58231972930012,
            ),
        )
