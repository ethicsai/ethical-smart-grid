from typing import List, Dict

from smartgrid.agents.agent import Agent
from smartgrid.rewards.reward import Reward
from smartgrid.world import World


class RewardCollection:
    """
    Reward Collection is a set of Reward for one Agent.
    If you have multiple reward, you need to specify an adequate AggregateFunction for passing to one value.
    """

    def __init__(self, rewards: List[Reward]):
        assert len(rewards) > 0, "You need to specify at least one Reward."
        self.rewards = rewards

    def compute(self, world: World, agent: Agent) -> Dict[str, float]:
        """
        Compute the set of Reward for the Agent.
        :param world: need to be specified for certain Reward
        :param agent: the agent
        """
        to_return = {}
        for reward in self.rewards:
            to_return[reward.name] = reward.calculate(world, agent)

        return to_return
