import unittest

from muspinsim import constants


class TestConstants(unittest.TestCase):
    def test_values(self):

        self.assertAlmostEqual(constants.ELEC_GAMMA, -28024.9514242)
        self.assertAlmostEqual(constants.MU_GAMMA, 135.53880943285955)

    def test_gyromagnetic(self):

        self.assertEqual(constants.gyromagnetic_ratio("e"), constants.ELEC_GAMMA)
        self.assertAlmostEqual(constants.gyromagnetic_ratio("H"), 42.577478518, 3)

        with self.assertRaises(ValueError):
            constants.gyromagnetic_ratio("H", 4)

    def test_quadrupole(self):

        self.assertEqual(constants.quadrupole_moment("e"), 0)
        self.assertEqual(constants.quadrupole_moment("H"), 0)
        self.assertAlmostEqual(constants.quadrupole_moment("H", 2), 2.859, 2)

    def test_spin(self):

        self.assertEqual(constants.spin("e"), 0.5)
        self.assertEqual(constants.spin("H"), 0.5)
        self.assertEqual(constants.spin("H", 2), 1.0)
