from abc import ABC

import numpy as np

from smartgrid.agents.agent import Agent
from smartgrid.rewards.reward import Reward
from smartgrid.util.equity import hoover
from smartgrid.world import World


class EquityRewardPerAgent(Reward):
    """
    Reward based on the equity of comforts measure.

    It follows the principle of Difference Rewards: we compare the measure
    in the actual environment (in which the agent acted) and in a
    hypothetical environment (in which the agent would not have acted).
    If the actual environment is better than the hypothetical one, the
    agent's action improved it and should be rewarded.
    Otherwise, the agent degraded it and should be punished.
    """

    def __init__(self):
        super().__init__("EquityRewardPerAgent")

    def calculate(self, world: World, agent: Agent):
        # Comforts of all agents
        comforts = [a.state.comfort for a in world.agents]

        # Compute the equity in the actual environment (others + agent)
        # we use 1-x since hoover returns 0=equity and 1=inequity
        actual_equity = 1.0 - hoover(comforts)

        return actual_equity
