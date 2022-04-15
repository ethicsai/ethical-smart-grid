from typing import List

from smartgrid.agents.agent import Agent
from smartgrid.rewards.reward import Reward
from smartgrid.world import World


class RewardCollection:

    def __init__(self, rewards: List[Reward]):
        assert len(rewards) > 0, "You need to specify at least one Reward."
        self.rewards = rewards

    def compute(self, world: World, agent: Agent):
        to_return = {}
        for reward in self.rewards:
            to_return[reward.name] = reward.calculate(world, agent)

        return to_return
