"""
Model that returns purely random actions.
"""

from algorithms.model import Model


class RandomModel(Model):
    """
    Model that returns purely random actions.

    The actions are based on the :py:attr:`~smartgrid.environment.SmartGrid.action_space`
    for each agent, using the :py:meth:`Space.sample() <gymnasium.spaces.space.Space.sample>`
    method.
    """

    def __init__(self, env, hyper_parameters: dict):
        super().__init__(env, hyper_parameters)

    def forward(self, observations_per_agent):
        actions = []
        for i in range(self.env.n_agent):
            action = self.env.action_space[i].sample()
            actions.append(action)

        return actions

    def backward(self, observations_per_agent, reward_per_agent):
        pass
