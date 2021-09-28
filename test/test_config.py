import unittest
import numpy as np
from io import StringIO

from muspinsim.input import MuSpinInput
from muspinsim.simconfig import MuSpinConfig, MuSpinConfigRange


class TestConfig(unittest.TestCase):

    def test_config(self):

        stest = StringIO("""
field
    1.0
""")

        itest = MuSpinInput(stest)

        cfg = MuSpinConfig(itest.evaluate())

        print(cfg._parameters)
