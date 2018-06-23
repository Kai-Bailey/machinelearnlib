import unittest
import sys
import os
import numpy as np

# Insert the path to the machinelearnlib folder so python will search those modules on import statements
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import machinelearnlib.activationFunc as af


class TestTesting(unittest.TestCase):

    def test_sigmoid(self):
        result  = af.sigmoid(np.array([-1, 0, 1]))
        np.testing.assert_allclose(result, np.array([0.26894142136999512, 0.5, 0.73105857863]), atol=0.00001)

if __name__ == '__main__':
    unittest.main()







