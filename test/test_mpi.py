import unittest
import numpy as np

from muspinsim.mpi import mpi_controller


class TestMuSpinMPI(unittest.TestCase):
    def test_split(self):
        # Test efficient splitting of processes
        a = list(range(20))
        b = list(range(17))

        split7 = mpi_controller.split_1D(a, 7)

        self.assertTrue(np.all(split7[0] == [0, 1, 2]))
        self.assertTrue(
            np.all([len(s) for s in split7] == np.array([3, 3, 3, 3, 3, 3, 2]))
        )

        split8 = mpi_controller.split_1D(a, 8)

        self.assertTrue(np.all(split8[0] == [0, 1, 2]))
        self.assertTrue(
            np.all([len(s) for s in split8] == np.array([3, 3, 3, 3, 2, 2, 2, 2]))
        )

        split30 = mpi_controller.split_2D(a, b, 30)

        self.assertTrue(np.all(split30[0][0] == [0, 1]))
        self.assertTrue(np.all(split30[0][1] == [0, 1, 2, 3, 4, 5]))
        self.assertTrue(np.all(split30[2][0] == [0, 1]))
        self.assertTrue(np.all(split30[2][1] == [12, 13, 14, 15, 16]))

    def test_broadcast(self):
        class A(object):
            def __init__(self, x):
                self.x = x

        class B(object):
            def __init__(self, x):
                self.a = A(x)

        b = B(mpi_controller.rank)
        mpi_controller.broadcast_object(b)

        self.assertEqual(b.a.x, 0)

        d = np.array([b.a.x])
        d = mpi_controller.sum_data(d)

        if mpi_controller.is_root:
            self.assertEqual(d[0], 0)
