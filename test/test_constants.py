import unittest

from muspinsim import constants


class TestConstants(unittest.TestCase):
    def test_values(self):

        self.assertAlmostEqual(constants.ELEC_GAMMA, -28024.9514242)
        self.assertAlmostEqual(constants.MU_GAMMA, 135.53880943285955)
        self.assertAlmostEqual(constants.MU_TAU, 2.19703)
        self.assertAlmostEqual(constants.EFG_2_MHZ, 0.23496477815245767)

    def test_gyromagnetic(self):

        self.assertEqual(constants.gyromagnetic_ratio("e"), constants.ELEC_GAMMA)
        self.assertEqual(constants.gyromagnetic_ratio("mu"), constants.MU_GAMMA)
        self.assertAlmostEqual(constants.gyromagnetic_ratio("H"), 42.577478518, 3)

        with self.assertRaises(ValueError) as err:
            constants.gyromagnetic_ratio("H", 4)
        self.assertEqual(str(err.exception), "Invalid isotope 4 for element H")

    def test_quadrupole(self):

        self.assertEqual(constants.quadrupole_moment("e"), 0)
        self.assertEqual(constants.quadrupole_moment("H"), 0)
        self.assertAlmostEqual(constants.quadrupole_moment("H", 2), 2.859, 2)

        with self.assertRaises(ValueError) as err:
            constants.gyromagnetic_ratio("H", 4)
        self.assertEqual(str(err.exception), "Invalid isotope 4 for element H")

    def test_spin(self):

        self.assertEqual(constants.spin("mu"), 0.5)
        self.assertEqual(constants.spin("e"), 0.5)
        self.assertEqual(constants.spin("H"), 0.5)
        self.assertEqual(constants.spin("H", 2), 1.0)

        with self.assertRaises(ValueError) as err:
            constants.spin("H", 4)
        self.assertEqual(str(err.exception), "Invalid isotope 4 for element H")

        with self.assertRaises(ValueError) as err:
            constants.spin("e", 0.5)
        self.assertEqual(str(err.exception), "Invalid multiplicity 0.5 for electron")
