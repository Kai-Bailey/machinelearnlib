import unittest
import os
import sys
import numpy as np


# Insert the path to the machinelearnlib folder so python will search those modules on import statements
topLevelFolder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
machinelearnlibFolder = os.path.join(topLevelFolder, 'machinelearnlib')
sys.path.insert(0, machinelearnlibFolder)

import models.linearRegression as linReg


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


    def test_cost(self):

        # Test without Regularization
        features = np.array([[2, 5, 4], [3, 6, 8], [9, 3, 2]])
        weights = np.array([4, 1, 2, 3])
        labels = np.array([30, 40, 15])
        cost = linReg.cost(features, labels, weights)

        np.testing.assert_almost_equal(cost, 18.83333333333, decimal=5)

        #Test with Reularization = 0.1
        cost = linReg.cost(features, labels, weights, 0.1)
        np.testing.assert_almost_equal(cost, 19.0666666666, decimal=5)

