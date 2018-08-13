import unittest
import sys
import os
import numpy as np

# Insert the path to the machinelearnlib folder so python will search those modules on import statements
topLevelFolder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
machinelearnlibFolder = os.path.join(topLevelFolder, 'machinelearnlib')
sys.path.insert(0, machinelearnlibFolder)

import activationFunc as af


class TestActivationFunc(unittest.TestCase):

    def test_sigmoid(self):
        result  = af.sigmoid(np.array([-1, 0, 1]))
        np.testing.assert_allclose(result, np.array([0.26894142136999512, 0.5, 0.73105857863]), atol=0.00001)

    def test_sigmoid_gradient(self):
        result  = af.sigmoid_gradient(np.array([-1, -0.5, 0, 0.5, 1]))
        np.testing.assert_allclose(result, np.array([0.196612, 0.235004, 0.25000000, 0.235004, 0.196612]), atol=0.00001)

if __name__ == '__main__':
    unittest.main()







