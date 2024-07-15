import random
import unittest

from smartgrid import make_basic_smartgrid

import os

# Set the current working dir to the project root, so we can import data
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))


class TestSeed(unittest.TestCase):

    def _helper_single_episode_single_step(self, seed):
        """
        Helper to test a single seed, in a single episode, for a single step.
        """
        env = make_basic_smartgrid()

        obs1 = env.reset(seed=seed)
        obs2 = env.reset(seed=seed)

        self.assertEqual(obs1, obs2)

    def _helper_single_episode_several_steps(self, seed, nb_steps):
        """
        Helper to test a single seed, in a single episode, for several steps.
        """
        env = make_basic_smartgrid()

        # 1st run, capture metrics (observations)
        obs_list = []
        obs = env.reset(seed=seed)
        for step in range(nb_steps):
            obs_list.append(obs)
            actions = {
                agent_name: (0.5, 0.5, 0.5, 0.5, 0.5, 0.5)
                for agent_name in env.agents
            }
            obs = env.step(actions)

        # 2nd run, check that the metrics correspond
        obs = env.reset(seed=seed)
        for step in range(nb_steps):
            self.assertEqual(obs, obs_list[step])
            actions = {
                agent_name: (0.5, 0.5, 0.5, 0.5, 0.5, 0.5)
                for agent_name in env.agents
            }
            obs = env.step(actions)

    def _helper_several_episodes_single_step(self, get_seed, nb_episodes):
        env = make_basic_smartgrid()

        for episode in range(nb_episodes):
            seed = get_seed()
            obs1 = env.reset(seed=seed)
            obs2 = env.reset(seed=seed)
            self.assertEqual(obs1, obs2)

    def _helper_several_episodes_several_steps(self, get_seed, nb_episodes, nb_steps):
        env = make_basic_smartgrid()

        for episode in range(nb_episodes):
            seed = get_seed()
            # 1st run: capture and memorize several metrics.
            obs_list = []
            obs = env.reset(seed=seed)
            for step in range(nb_steps):
                obs_list.append(obs)
                actions = {
                    agent_name: (0.5, 0.5, 0.5, 0.5, 0.5, 0.5)
                    for agent_name in env.agents
                }
                obs = env.step(actions)
            # 2nd run with same seed: check that all metrics are the same as
            # the 1st run.
            obs = env.reset(seed=seed)
            for step in range(nb_steps):
                self.assertEqual(obs, obs_list[step])
                actions = {
                    agent_name: (0.5, 0.5, 0.5, 0.5, 0.5, 0.5)
                    for agent_name in env.agents
                }
                obs = env.step(actions)

    def test_available_energy(self):
        """
        Simple test, we check only for the initial available energy.
        """
        env = make_basic_smartgrid()

        env.reset(seed=1234)
        available_energy_step1 = env.world.available_energy

        env.reset(seed=1234)
        available_energy_step2 = env.world.available_energy

        self.assertEqual(available_energy_step1, available_energy_step2)

    def test_simple_hardcoded_seed(self):
        """
        Simple test again, but check for all observations instead of only energy.
        """
        self._helper_single_episode_single_step(seed=1234)

    def test_complex_hardcoded_seed(self):
        """
        Still a hardcoded seed, but we test for several steps.
        """
        self._helper_single_episode_several_steps(seed=1234, nb_steps=100)

    def test_simple_random_seed(self):
        """
        Test a random seed for a single step.
        """
        seed = random.randint(0, 100_000)
        self._helper_single_episode_single_step(seed=seed)

    def test_random_seed_several_steps(self):
        """
        Test a single random seed for several steps.
        """
        seed = random.randint(0, 100_000)
        self._helper_single_episode_several_steps(seed=seed, nb_steps=100)

    def test_random_seed_several_episodes(self):
        """
        Test several random seeds in several episodes (but a single step each).
        """
        self._helper_several_episodes_single_step(
            get_seed=lambda: random.randint(0, 100_000),
            nb_episodes=10
        )

    def test_complex_random_seeds(self):
        """
        Test several random seeds in several episodes, for several steps each.
        """
        self._helper_several_episodes_several_steps(
            get_seed=lambda: random.randint(0, 100_000),
            nb_episodes=10,
            nb_steps=100
        )


if __name__ == '__main__':
    unittest.main()
