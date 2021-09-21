import unittest

from muspinsim.utils import Clonable, deepmap


class TestUtils(unittest.TestCase):

    def test_clonable(self):

        class TestClass(Clonable):
            def __init__(self):
                self.data = {'x': [1, 2, 3]}

        tc = TestClass()
        # Clone it
        tc2 = tc.clone()
        # Change values
        tc2.data['x'][0] = 2
        # Check it didn't change the original
        self.assertEqual(tc.data['x'][0], 1)

    def test_deepmap(self):

        data = [[1, 2, 3], [4, 5], [6, [7, 8]]]

        def square(x):
            return x**2

        data2 = deepmap(square, data)

        self.assertEqual(data2[0], [1, 4, 9])
        self.assertEqual(data2[1], [16, 25])
        self.assertEqual(data2[2][0], 36)
        self.assertEqual(data2[2][1], [49, 64])
