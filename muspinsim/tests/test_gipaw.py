from io import StringIO
import unittest
import numpy as np

from muspinsim.input.gipaw import GIPAWAtom, GIPAWOutput

TEST_GIPAW_FILE_DATA = """     H  217        0.000008       -0.000024        0.068534

     ----- total EFG -----
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


     NQR/NMR SPECTROSCOPIC PARAMETERS:
"""


class GIPAWAtomMatcher:
    expected: GIPAWAtom

    def __init__(self, expected):
        self.expected = expected

    def __repr__(self):
        return repr(self.expected)

    def __eq__(self, other):
        return self.expected.index == other.index and np.allclose(
            self.expected.efg_tensor, other.efg_tensor
        )


class TestGIPAW(unittest.TestCase):
    def test_parsing(self):
        gipaw_out_file = GIPAWOutput(StringIO(TEST_GIPAW_FILE_DATA))

        self.assertEqual(len(gipaw_out_file._atoms), 4)
        self.assertEqual(
            GIPAWAtomMatcher(gipaw_out_file._atoms[2]),
            GIPAWAtom(
                index=168,
                efg_tensor=np.array(
                    [
                        [-2.604e-03, -0.000e00, -1.000e-06],
                        [-0.000e00, -2.551e-03, -9.000e-06],
                        [-1.000e-06, -9.000e-06, 5.154e-03],
                    ]
                ),
            ),
        )

    def test_load_error(self):
        with self.assertRaises(ValueError):
            GIPAWOutput(
                StringIO(
                    """some random text
                       with newlines"""
                )
            )

    def test_parsing_errors(self):
        with self.assertRaises(ValueError):
            GIPAWOutput(
                StringIO(
                    """     ----- total EFG -----
     V    1       -0.008877        0.000022       -0.011811
     V    2        0.000022       -0.030585        0.000005
     V    1       -0.011811        0.000005        0.039462"""
                )
            )

        with self.assertRaises(ValueError):
            GIPAWOutput(
                StringIO(
                    """     ----- total EFG -----
     V    1       -0.008877        0.000022       -0.011811
     Si   1        0.000022       -0.030585        0.000005
     V    1       -0.011811        0.000005        0.039462"""
                )
            )

    def test_find_atom(self):
        gipaw_out_file = GIPAWOutput(StringIO(TEST_GIPAW_FILE_DATA))

        # Atom that will be at the corresponding index
        self.assertEqual(
            GIPAWAtomMatcher(gipaw_out_file.find_atom(2)),
            GIPAWAtom(
                index=2,
                efg_tensor=np.array(
                    [
                        [2.21597e-01, -2.00000e-06, -9.01300e-03],
                        [-2.00000e-06, -1.09627e-01, -9.00000e-06],
                        [-9.01300e-03, -9.00000e-06, -1.11970e-01],
                    ]
                ),
            ),
        )

        # This will not be at the corresponding index
        self.assertEqual(
            GIPAWAtomMatcher(gipaw_out_file.find_atom(168)),
            GIPAWAtom(
                index=168,
                efg_tensor=np.array(
                    [
                        [-2.604e-03, -0.000e00, -1.000e-06],
                        [-0.000e00, -2.551e-03, -9.000e-06],
                        [-1.000e-06, -9.000e-06, 5.154e-03],
                    ]
                ),
            ),
        )
