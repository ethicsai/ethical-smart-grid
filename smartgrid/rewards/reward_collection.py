from typing import List, Dict

from smartgrid.agents.agent import Agent
from smartgrid.rewards.reward import Reward


class RewardCollection:
    """
    Reward Collection is responsible for keeping reward used in memory and compute it.

    Multiple reward used an :py:class:`.AggregateFunction` that can be extended.
    Mono reward used :py:class:`.BasicAggregateFunction`.
    """

    def __init__(self, rewards: List[Reward]):
        assert len(rewards) > 0, "You need to specify at least one Reward."
        self.rewards = rewards

    def compute(self, world: 'World', agent: Agent) -> Dict[str, float]:
        """
        Compute the list of :py:class:`.Reward` for the Agent.

        :param world: reference on the world for global information.
        :param agent: reference on the agent for local information.

        :rtype: dict
        :return: The name of the reward with his value.
        """
        to_return = {}
        for reward in self.rewards:
            to_return[reward.name] = reward.calculate(world, agent)

        return to_return

    def __str__(self):
        to_return = []
        for reward in self.rewards:
            to_return.append(str(reward))
        return '_'.join(to_return)
