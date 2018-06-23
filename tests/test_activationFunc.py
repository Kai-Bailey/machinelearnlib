import unittest
import sys
import os
import numpy as np

# Insert the path to the machinelearnlib folder so python will search those modules on import statements
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import machinelearnlib.activationFunc as af


class TestTesting(unittest.TestCase):

    def test_multiply(self):
        result = af.multiply(4, 5)
        self.assertEqual(result, 20)



if __name__ == '__main__':
    unittest.main()







