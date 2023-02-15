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


def _optional_float_close(value1, value2):
    # Compares two values that may be a float or none
    if value1 is None or value2 is None:
        return value1 == value2
    else:
        return np.isclose(value1, value2)


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
            and _optional_float_close(
                self.expected.distance_from_muon, other.distance_from_muon
            )
        )


class TestStructure(unittest.TestCase):
    def test_parsing(self):
        structure = MuonatedStructure(StringIO(TEST_CELL_FILE_DATA), fmt="castep-cell")

        self.assertEqual(len(structure._cell_atoms), 4)
        self.assertEqual(structure._muon_index, 3)
        self.assertEqual(
            CellAtomMatcher(structure._cell_atoms[2]),
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

    def test_compute_layer_offsets(self):
        structure = MuonatedStructure(StringIO(TEST_CELL_FILE_DATA), fmt="castep-cell")

        offsets = structure._compute_layer_offsets(1)
        self.assertEqual(len(offsets), 26)
        self.assertTrue(
            np.allclose(
                offsets,
                [
                    [-14.1816, -14.1816, -14.1816],
                    [-14.1816, -14.1816, 0.0],
                    [-14.1816, -14.1816, 14.1816],
                    [-14.1816, 0.0, -14.1816],
                    [-14.1816, 0.0, 0.0],
                    [-14.1816, 0.0, 14.1816],
                    [-14.1816, 14.1816, -14.1816],
                    [-14.1816, 14.1816, 0.0],
                    [-14.1816, 14.1816, 14.1816],
                    [14.1816, -14.1816, -14.1816],
                    [14.1816, -14.1816, 0.0],
                    [14.1816, -14.1816, 14.1816],
                    [14.1816, 0.0, -14.1816],
                    [14.1816, 0.0, 0.0],
                    [14.1816, 0.0, 14.1816],
                    [14.1816, 14.1816, -14.1816],
                    [14.1816, 14.1816, 0.0],
                    [14.1816, 14.1816, 14.1816],
                    [0.0, -14.1816, -14.1816],
                    [0.0, -14.1816, 0.0],
                    [0.0, -14.1816, 14.1816],
                    [0.0, 14.1816, -14.1816],
                    [0.0, 14.1816, 0.0],
                    [0.0, 14.1816, 14.1816],
                    [0.0, 0.0, -14.1816],
                    [0.0, 0.0, 14.1816],
                ],
            )
        )

        # Check correct number for larger system
        offsets = structure._compute_layer_offsets(2)
        self.assertEqual(len(offsets), 98)

        # Check offsets are correct for a system with different cell lengths
        structure = MuonatedStructure(
            StringIO(
                """%BLOCK LATTICE_CART
   14.18160000000000   0.000000000000000   0.000000000000000
   0.000000000000000   20.00000000000000   0.000000000000000
   0.000000000000000   0.000000000000000   8.000000000000000
%ENDBLOCK LATTICE_CART

%BLOCK POSITIONS_FRAC
H             0.1666672745        0.0000018274        0.0833332099
%ENDBLOCK POSITIONS_FRAC"""
            ),
            fmt="castep-cell",
        )
        offsets = structure._compute_layer_offsets(1)
        self.assertTrue(
            np.allclose(
                offsets,
                np.array(
                    [
                        [-14.1816, -20.0, -8.0],
                        [-14.1816, -20.0, 0.0],
                        [-14.1816, -20.0, 8.0],
                        [-14.1816, 0.0, -8.0],
                        [-14.1816, 0.0, 0.0],
                        [-14.1816, 0.0, 8.0],
                        [-14.1816, 20.0, -8.0],
                        [-14.1816, 20.0, 0.0],
                        [-14.1816, 20.0, 8.0],
                        [14.1816, -20.0, -8.0],
                        [14.1816, -20.0, 0.0],
                        [14.1816, -20.0, 8.0],
                        [14.1816, 0.0, -8.0],
                        [14.1816, 0.0, 0.0],
                        [14.1816, 0.0, 8.0],
                        [14.1816, 20.0, -8.0],
                        [14.1816, 20.0, 0.0],
                        [14.1816, 20.0, 8.0],
                        [0.0, -20.0, -8.0],
                        [0.0, -20.0, 0.0],
                        [0.0, -20.0, 8.0],
                        [0.0, 20.0, -8.0],
                        [0.0, 20.0, 0.0],
                        [0.0, 20.0, 8.0],
                        [0.0, 0.0, -8.0],
                        [0.0, 0.0, 8.0],
                    ]
                ),
            )
        )

    def test_compute_layer(self):
        structure = MuonatedStructure(StringIO(TEST_CELL_FILE_DATA), fmt="castep-cell")
        new_atoms = structure.compute_layer(1)

        self.assertEqual(len(new_atoms), 3 * 26)  # Copy everything but muon
        self.assertEqual(
            CellAtomMatcher(new_atoms[11]),
            CellAtom(
                index=3,
                symbol="Si",
                isotope=1,
                position=np.array([-7.09606527, 11.81519966, -2.36596844]),
                vector_from_muon=None,
                distance_from_muon=None,
            ),
        )

    def test_compute_layer_ignore(self):
        structure = MuonatedStructure(StringIO(TEST_CELL_FILE_DATA), fmt="castep-cell")
        new_atoms = structure.compute_layer(1, ignored_symbols=["Si"])

        print(new_atoms[11])

        self.assertEqual(len(new_atoms), 2 * 26)  # Copy everything but muon and silicon
        self.assertEqual(
            CellAtomMatcher(new_atoms[11]),
            CellAtom(
                index=2,
                symbol="V",
                isotope=1,
                position=np.array([-1.18888930e00, 8.25794568e-06, 2.59887219e01]),
                vector_from_muon=None,
                distance_from_muon=None,
            ),
        )

    def test_compute_closest(self):
        structure = MuonatedStructure(StringIO(TEST_CELL_FILE_DATA), fmt="castep-cell")
        closest_atoms = structure.compute_closest(1)

        self.assertEqual(len(closest_atoms), 1)
        self.assertEqual(
            CellAtomMatcher(closest_atoms[0]),
            CellAtom(
                index=1,
                symbol="V",
                isotope=1,
                position=np.array([1.12255466e00, 8.30049048e-06, 2.42350038e00]),
                vector_from_muon=np.array(
                    [1.24105396e00, 1.76149654e-05, -1.24170213e00]
                ),
                distance_from_muon=1.7555737250028431,
            ),
        )

        more_closest_atoms = structure.compute_closest(6)

        self.assertEqual(len(more_closest_atoms), 6)
        self.assertEqual(CellAtomMatcher(closest_atoms[0]), more_closest_atoms[0])
        self.assertEqual(
            CellAtomMatcher(more_closest_atoms[5]),
            CellAtom(
                index=2,
                symbol="V",
                isotope=1,
                position=np.array([1.29927107e01, 8.25794568e-06, -2.37447814e00]),
                vector_from_muon=np.array(
                    [-1.06291021e01, 1.76575102e-05, 3.55627639e00]
                ),
                distance_from_muon=11.208251995910084,
            ),
        )
