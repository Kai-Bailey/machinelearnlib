import unittest
from .config import machinelearnlib

class TestTesting(unittest.TestCase):

    def test_add(self):
        result = testing.add(10, 5)
        self.assertEqual(result, 15)


if __name__ == '__main__':
    unittest.main()