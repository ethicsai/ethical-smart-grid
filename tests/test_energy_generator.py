import random
import unittest
import numpy as np

# Note that the Scarcity and Generous generators are not tested, as they
# are based on RandomEnergyGenerator, and simply change its lower/upper bounds.
from smartgrid.util import (RandomEnergyGenerator,
                            RealisticEnergyGenerator)


class TestEnergyGenerator(unittest.TestCase):

    def test_random_generator(self):
        """Test the RandomEnergyGenerator."""
        lower = 0.8
        upper = 1.2
        generator = RandomEnergyGenerator(lower_proportion=lower,
                                          upper_proportion=upper)

        # The (total) min and max needs for all agents, at all time steps
        min_need = 10_000
        max_need = 100_000

        # Test that bounds are correct
        # The current need and current step should not be used here, so
        # let's use random values to ensure they do not change the result!
        current_need = random.randint(-100_000, 100_000)
        current_step = random.randint(0, 10_000)
        min_bound, max_bound = generator.available_energy_bounds(
            current_need, current_step, min_need, max_need
        )
        self.assertEqual(min_bound, int(min_need * lower))
        self.assertEqual(max_bound, int(max_need * upper))

        # Test that random amounts are generated within the bounds
        # (many draws to increase reliability)
        for _ in range(100_000):
            # Now it's important that current_need stays in [min_need, max_need]
            current_need = random.randint(min_need, max_need)
            # The loop index could also be used, but the current step should
            # not change anything, so let's randomize it again.
            current_step = random.randint(0, 10_000)

            energy = generator.generate_available_energy(
                current_need, current_step, min_need, max_need
            )
            # The generated energy is within the possible bounds (min and max)
            self.assertGreaterEqual(energy, min_bound)
            self.assertLessEqual(energy, max_bound)
            # We can further restrict it to
            # [lower*current_need, upper*current_need]
            self.assertGreaterEqual(energy, int(lower*current_need))
            self.assertLessEqual(energy, int(upper*current_need))

    def test_random_generator_with_seed(self):
        """Test the RandomEnergyGenerator with a fixed seed."""
        random.seed(42)
        generator = RandomEnergyGenerator(lower_proportion=0.8,
                                          upper_proportion=1.2)
        energy = generator.generate_available_energy(
            current_need=10_000, current_step=0, min_need=0, max_need=100_000
        )
        self.assertEqual(energy, 10_619)

    def test_realistic_generator(self):
        """Test the RealisticEnergyGenerator."""
        data = [0.3, 0.8, 0.7]
        generator = RealisticEnergyGenerator(data)

        # The (total) min and max needs for all agents, at all time steps
        min_need = 10_000
        max_need = 100_000

        # Test that bounds are correct
        # The current need and current step should not be used here, so
        # let's use random values to ensure they do not change the result!
        current_need = random.randint(-100_000, 100_000)
        current_step = random.randint(0, 10_000)
        min_bound, max_bound = generator.available_energy_bounds(
            current_need, current_step, min_need, max_need
        )
        self.assertEqual(min_bound, int(max_need * min(data)))
        self.assertEqual(max_bound, int(max_need * max(data)))

        # Test that generated amounts remain within bounds and use the data.
        # We do not need many draws here since data consist of only 3 values.
        for current_step in range(20):
            # current_need should still not impact the result
            current_need = random.randint(-100_000, 100_000)

            energy = generator.generate_available_energy(
                current_need, current_step, min_need, max_need
            )
            # The generated energy is within the possible bounds (min and max)
            self.assertGreaterEqual(energy, min_bound)
            self.assertLessEqual(energy, max_bound)
            # We can further restrict the generated energy to
            # data[current_step] * max_need
            ratio = data[current_step % len(data)]
            self.assertAlmostEqual(energy, int(ratio * max_need))

    def test_realistic_generator_with_random_data(self):
        """Test the RealisticEnergyGenerator using an unknown array of data."""
        data_size = 1_000
        data = np.random.random(data_size)
        generator = RealisticEnergyGenerator(data)

        # The (total) min and max needs for all agents, at all time steps
        min_need = 10_000
        max_need = 100_000

        # Test that bounds are correct
        # The current need and current step should not be used here, so
        # let's use random values to ensure they do not change the result!
        current_need = random.randint(-100_000, 100_000)
        current_step = random.randint(0, 10_000)
        min_bound, max_bound = generator.available_energy_bounds(
            current_need, current_step, min_need, max_need
        )
        self.assertEqual(min_bound, int(max_need * min(data)))
        self.assertEqual(max_bound, int(max_need * max(data)))

        # Test that generated amounts remain within bounds and use the data.
        # We want more draws than the data size, but not too much (it should
        # cycle anyway).
        for current_step in range(2 * data_size):
            # current_need should still not impact the result
            current_need = random.randint(-100_000, 100_000)

            energy = generator.generate_available_energy(
                current_need, current_step, min_need, max_need
            )
            # The generated energy is within the possible bounds (min and max)
            self.assertGreaterEqual(energy, min_bound)
            self.assertLessEqual(energy, max_bound)
            # We can further restrict the generated energy to
            # data[current_step] * max_need
            ratio = data[current_step % len(data)]
            self.assertAlmostEqual(energy, int(ratio * max_need))


if __name__ == '__main__':
    unittest.main()
