import unittest
import os
import sys
import numpy as np

# Insert the path to the machinelearnlib folder so python will search those modules on import statements
topLevelFolder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
machinelearnlibFolder = os.path.join(topLevelFolder, 'machinelearnlib')
sys.path.insert(0, machinelearnlibFolder)

import featureScaling


class TestFeaturesScaling(unittest.TestCase):
        
    def test_meanNormalization(self):

        class Model():

            def __init__(self):
                self.features = np.array([[2, 5, 4], [3, 6, 8], [9, 3, 9]], dtype='float32')

        mlModel = Model()

        featureScaling.meanNormalization(mlModel)
        np.testing.assert_allclose(mlModel.mean, np.array([4.666666, 4.6666666, 7]), atol=0.001)
        np.testing.assert_allclose(mlModel.features, np.array([[-2.666666, 0.333333, -3.0], [-1.666666, 1.333333, 1.0], [4.3333, -1.666666, 2]]), atol=0.001)
        
    
    def test_standardDeviationNormalization(self):

        class Model():

            def __init__(self):
                self.features = np.array([[2, 5, 4], [3, 6, 8], [9, 3, 9]], dtype='float32')

        mlModel = Model()

        featureScaling.stdNormalization(mlModel)
        np.testing.assert_allclose(mlModel.std, np.array([3.0912, 1.247219, 2.160247]), atol=0.001)
        np.testing.assert_allclose(mlModel.features, np.array([[0.646997, 4.00819, 1.85164], [0.970495, 4.810702, 3.70328], [2.911485, 2.405351, 4.166191]]), atol=0.001)
