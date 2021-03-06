import unittest
import os
import sys
import numpy as np

# Insert the path to the top level folder so python will search those modules on import statements
topLevelFolder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, topLevelFolder)

import machinelearnlib.models.logisticRegression as logReg


class TestLogisticRegression(unittest.TestCase):

    def test_predict(self):

        features = np.array([[2, 5, 4], [3, 6, 8], [9, 3, 2]])
        weights = np.array([0.04, 0.01, 0.02, 0.03])
        prediction = logReg.predict(features, weights)

        np.testing.assert_allclose(prediction, np.array([0.5695462, 0.6058737, 0.5621765]), atol=0.0001)


        # Each should be an example with one feature
        features = np.array([[8], [3], [-6], [-1]])
        weights = np.array([0.02, 0.03])
        prediction = logReg.predict(features, weights)

        np.testing.assert_allclose(prediction, np.array([0.5646363, 0.5274723, 0.4600851, 0.49750002]), atol=0.0001)


    def test_cost(self):

        features = np.array([[2, 5, 4], [3, 6, 8], [9, 3, 2]])
        weights = np.array([0.04, 0.01, 0.02, 0.03])
        labels = np.array([1, 0, 1])

        logRegCost  = logReg.cost(features, labels, weights)

        np.testing.assert_array_almost_equal(logRegCost, np.array([0.689979553]), decimal=5)


    def test_gradient(self):

        features = np.array([[2, 5, 4], [3, 6, 8], [9, 3, 2]])
        weights = np.array([0.04, 0.01, 0.02, 0.03])
        labels = np.array([1, 0, 1])

        grad = logReg.gradient(features, labels, weights, 0.1)

        np.testing.assert_allclose(grad, np.array([-0.08746787, -0.99423268, 0.05716754, 0.75084241]), atol=0.0001)

