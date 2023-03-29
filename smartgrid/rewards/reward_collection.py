"""
The RewardCollection is responsible for computing rewards from reward functions.
"""

from typing import List, Dict

from smartgrid.agents import Agent
from .reward import Reward


class RewardCollection:
    """
    The RewardCollection holds all desired reward functions, and computes the rewards.

    This class allows for multi-objective reinforcement learning, by holding
    several reward functions, and returning dicts of rewards (names -> values),
    instead of using a single reward function.

    The multiple reward functions can be aggregated (scalarized) to adapt to
    single-objective learning algorithms, by using a
    :py:class:`~smartgrid.wrappers.reward_aggregator.RewardAggregator` wrapper
    over the environment.
    """

    def __init__(self, rewards: List[Reward]):
        """
        Create a RewardCollection based on a list of "reward functions".

        :param rewards: The list of "reward functions" (actually instances of
            the :py:class:`~smartgrid.rewards.reward.Reward` class). This
            list must contain at least 1 element.
        """
        assert len(rewards) > 0, "You need to specify at least one Reward."
        self.rewards = rewards

    def compute(self, world: 'World', agent: Agent) -> Dict[str, float]:
        """
        Compute the list of :py:class:`.Reward` for the Agent.

        :param world: reference on the world for global information.
        :param agent: reference on the agent for local information.

        :return: A dictionary mapping the rewards' name to their value, for
            each reward function in this Reward Collection.
        """
        to_return = {}
        for reward in self.rewards:
            to_return[reward.name] = reward.calculate(world, agent)

        return to_return

    def reset(self):
        """
        Reset the reward functions.
        """
        for reward in self.rewards:
            reward.reset()

    def __repr__(self):
        rewards = ' ; '.join(map(str, self.rewards))
        return 'RewardCollection{' + rewards + '}'
