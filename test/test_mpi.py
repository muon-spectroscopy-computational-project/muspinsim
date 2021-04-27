import unittest
import numpy as np

from muspinsim.mpi import split_efficiently_1D, split_efficiently_2D


class TestMuSpinMPI(unittest.TestCase):

    def test_split(self):
        # Test efficient splitting of processes
        a = list(range(20))
        b = list(range(17))

        split7 = split_efficiently_1D(a, 7)

        self.assertTrue(np.all(split7[0] == [0, 1, 2]))
        self.assertTrue(np.all([len(s) for s in split7] ==
                               np.array([3, 3, 3, 3, 3, 3, 2])))

        split8 = split_efficiently_1D(a, 8)

        self.assertTrue(np.all(split8[0] == [0, 1, 2]))
        self.assertTrue(np.all([len(s) for s in split8] ==
                               np.array([3, 3, 3, 3, 2, 2, 2, 2])))

        split30 = split_efficiently_2D(a, b, 30)

        self.assertTrue(np.all(split30[0][0] == [0, 1]))
        self.assertTrue(np.all(split30[0][1] == [0, 1, 2, 3, 4, 5]))
        self.assertTrue(np.all(split30[2][0] == [0, 1]))
        self.assertTrue(np.all(split30[2][1] == [12, 13, 14, 15, 16]))
