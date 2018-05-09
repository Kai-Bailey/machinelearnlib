import unittest
import os
import sys

# Insert the path to the machinelearnlib folder so python will search those modules on import statements
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import machinelearnlib.testing as testing


class TestTesting(unittest.TestCase):

    def test_add(self):
        result = testing.add(10, 5)
        self.assertEqual(result, 15)


if __name__ == '__main__':
    unittest.main()