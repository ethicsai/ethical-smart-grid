import unittest

import numpy as np

from smartgrid.util import interpolate


class TestInterpolate(unittest.TestCase):

    def test_interpolate_scalar(self):
        """Interpolate single values."""
        # Interpolate from [0, 100] to [0, 1].
        value = 47
        old_bounds = [0, 100]
        new_bounds = [0, 1]
        interpolated = interpolate(value, old_bounds, new_bounds)
        self.assertIsInstance(interpolated, float)
        self.assertGreaterEqual(interpolated, new_bounds[0])
        self.assertLessEqual(interpolated, new_bounds[1])
        self.assertAlmostEqual(interpolated, 0.47)

        # Interpolate from [0, 1] to [-100, 100].
        value = 0.2
        old_bounds = [0, 1]
        new_bounds = [-100, 100]
        interpolated = interpolate(value, old_bounds, new_bounds)
        self.assertIsInstance(interpolated, float)
        self.assertGreaterEqual(interpolated, new_bounds[0])
        self.assertLessEqual(interpolated, new_bounds[1])
        self.assertAlmostEqual(interpolated, -60)

    def test_interpolate_array(self):
        """Interpolate arrays of values."""
        # Interpolate from [ [0, 100]^3 ] to [ [0,1]^3 ].
        value = [80, 50, 66]
        old_bounds = [ [0, 100], [0, 100], [0, 100] ]
        new_bounds = [ [0, 1], [0, 1], [0, 1] ]
        interpolated = interpolate(value, old_bounds, new_bounds)
        self.assertIsInstance(interpolated, np.ndarray)
        self.assertEqual(len(interpolated), 3)
        expected = [0.8, 0.5, 0.66]
        for i, k in enumerate(interpolated):
            self.assertGreaterEqual(k, new_bounds[i][0])
            self.assertLessEqual(k, old_bounds[i][1])
            self.assertAlmostEqual(k, expected[i])

        # Interpolate from [ [0,1]^3 ] to [ [-100, 100]^3 ].
        value = [0.1, 0.4, 0.9]
        old_bounds = [ [0, 1], [0, 1], [0, 1] ]
        new_bounds = [ [-100, 100], [-100, 100], [-100, 100] ]
        interpolated = interpolate(value, old_bounds, new_bounds)
        self.assertIsInstance(interpolated, np.ndarray)
        self.assertEqual(len(interpolated), 3)
        expected = [-80, -20, 80]
        for i, k in enumerate(interpolated):
            self.assertGreaterEqual(k, new_bounds[i][0])
            self.assertLessEqual(k, new_bounds[i][1])
            self.assertAlmostEqual(k, expected[i])

        # Interpolate with different bounds for each dimension.
        value = [0.1, 50, -15]
        old_bounds = [ [0, 1], [0, 100] , [-20, 0] ]
        new_bounds = [ [-100, 100], [0, 1], [0, 20] ]
        interpolated = interpolate(value, old_bounds, new_bounds)
        self.assertIsInstance(interpolated, np.ndarray)
        self.assertEqual(len(interpolated), 3)
        expected = [-80, 0.5, 5]
        for i, k in enumerate(interpolated):
            self.assertGreaterEqual(k, new_bounds[i][0])
            self.assertLessEqual(k, new_bounds[i][1])
            self.assertAlmostEqual(k, expected[i])


if __name__ == '__main__':
    unittest.main()
