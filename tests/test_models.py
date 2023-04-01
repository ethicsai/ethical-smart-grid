import unittest

from smartgrid import make_basic_smartgrid
from algorithms.naive import RandomModel
from algorithms.qsom import QSOM


class TestModels(unittest.TestCase):
    """
    Test models (learning algorithms) and the interaction loop.
    """

    def test_random_model(self):
        env = make_basic_smartgrid()
        model = RandomModel(env.unwrapped, {})
        obs = env.reset(seed=123)

        max_step = 10
        for step in range(max_step):
            actions = model.forward(obs)
            assert len(actions) == env.unwrapped.n_agent, "There should be exactly as much actions as agents"
            obs, rewards, _, _, infos = env.step(actions)
        env.close()

    def test_qsom_model(self):
        env = make_basic_smartgrid()

        hyperparams = {
            'initial_tau': 0.5,
            'tau_decay': False,
            'tau_decay_coeff': 1.0,
            'noise': 0.08,
            'sigma_state': 1.0,
            'lr_state': 0.8,
            'sigma_action': 1.0,
            'lr_action': 0.7,
            'q_learning_rate': 0.7,
            'q_discount_factor': 0.9,
            'update_all': True,
            'use_neighborhood': True,
        }
        model = QSOM(env.unwrapped, hyperparams)
        obs = env.reset(seed=123)

        max_step = 10
        for step in range(max_step):
            actions = model.forward(obs)
            obs, rewards, _, _, infos = env.step(actions)
            model.backward(obs, rewards)
        env.close()


if __name__ == '__main__':
    unittest.main()
