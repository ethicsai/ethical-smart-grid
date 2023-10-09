import os
import unittest
from pathlib import Path

import numpy as np

from smartgrid import make_basic_smartgrid
from smartgrid.wrappers import NormsWrapper


class TestNorms(unittest.TestCase):

    def setUp(self):
        # Needed to be able to correctly import the `data/` files, as if
        # we were in the root (parent) folder.
        os.chdir(Path(__file__).parent.parent)

    def test_simple(self):
        env = make_basic_smartgrid()
        wrapper = NormsWrapper(
            env,
            [
                # Simple norm checking that all parameters of an action are > 0
                lambda a: np.all(np.asarray(a) > 0.0)
            ],
            True
        )

        # Test that a correct action does not violate norms
        action_ok = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        self.assertTrue(wrapper._check_action(action_ok))

        # Test that an incorrect action (at least one `0.0` value) violates norms.
        action_not_ok = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6]
        self.assertFalse(wrapper._check_action(action_not_ok))

        # Test that actions are effectively replaced if they violate norms, and
        # violations are correctly logged.
        violates, new_actions = wrapper._filter_actions([action_ok, action_not_ok])
        self.assertListEqual([False, True], violates)
        self.assertEqual(action_ok, new_actions[0])
        self.assertEqual(NormsWrapper.default_action, new_actions[1])

    def test_env(self):
        env = make_basic_smartgrid()
        wrapper = NormsWrapper(env, [lambda a: np.all(np.asarray(a) > 0.0)], True)

        # Generate a correct action, an incorrect one, and random actions for
        # the remaining agents (env contains N agents, we only care about the
        # first 2).
        action_ok = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        action_not_ok = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6]
        actions = [
            action_ok,
            action_not_ok
        ] + [
            env.action_space[i].sample()
            for i in range(2, env.n_agent)
        ]

        # Execute a step in the environment, while checking for violations.
        wrapper.reset()
        _, rewards, _, _, info = wrapper.step(actions)

        # Test that rewards are removed, and violations are correctly logged.
        self.assertIsNotNone(rewards[0])
        self.assertIsNone(rewards[1])
        self.assertFalse(info['violations'][0])
        self.assertTrue(info['violations'][1])


if __name__ == '__main__':
    unittest.main()
