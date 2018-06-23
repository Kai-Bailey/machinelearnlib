import unittest
import os
import sys
import numpy as np


# Insert the path to the machinelearnlib folder so python will search those modules on import statements
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import machinelearnlib.models.linearRegression as linReg


class TestLinearRegression(unittest.TestCase):

    def test_predict(self):

        features = np.array([[2, 5, 4], [3, 6, 8], [9, 3, 2]])
        weights = np.array([4, 1, 2, 3])
        prediction = linReg.predict(features, weights)

        np.testing.assert_array_equal(prediction, np.array([28, 43, 25]))


        # Each should be an example with one feature
        features = np.array([[8], [3], [6], [5]])
        weights = np.array([2, 3])
        prediction = linReg.predict(features, weights)

        np.testing.assert_array_equal(prediction, np.array([26, 11, 20, 17]))


