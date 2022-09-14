import unittest

from smartgrid.util.bounded import (increase_bounded, decrease_bounded)


class TestBounded(unittest.TestCase):

    def test_increase(self):
        # (0 + 100) < 1000
        # => 100, added=100, overhead=0
        new, added, overhead = increase_bounded(0, 100, 1000)
        self.assertEqual(new, 100)
        self.assertEqual(added, 100)
        self.assertEqual(overhead, 0)

        # (0 + 1500) < 1000
        # => 1000, added=1000, overhead=500
        new, added, overhead = increase_bounded(0, 1500, 1000)
        self.assertEqual(new, 1000)
        self.assertEqual(added, 1000)
        self.assertEqual(overhead, 500)

        # (1000 + 500) < 1000
        # => 1000, overhead=500
        new, added, overhead = increase_bounded(1000, 500, 1000)
        self.assertEqual(new, 1000)
        self.assertEqual(added, 0)
        self.assertEqual(overhead, 500)

    def test_decrease(self):
        # (500 - 100) > 0
        # => 400, subtracted=100, missing = 0
        new, subtracted, missing = decrease_bounded(500, 100, 0)
        self.assertEqual(new, 400)
        self.assertEqual(subtracted, 100)
        self.assertEqual(missing, 0)

        # (250 - 1000) > 0
        # => 0, subtracted=250, missing=750
        new, subtracted, missing = decrease_bounded(250, 1000, 0)
        self.assertEqual(new, 0)
        self.assertEqual(subtracted, 250)
        self.assertEqual(missing, 750)

        # (0 - 500) > 0
        # => 0, subtracted=0, missing=500
        new, subtracted, missing = decrease_bounded(0, 500, 0)
        self.assertEqual(new, 0)
        self.assertEqual(subtracted, 0)
        self.assertEqual(missing, 500)


if __name__ == '__main__':
    unittest.main()
