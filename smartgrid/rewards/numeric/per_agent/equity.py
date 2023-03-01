from smartgrid.agents.agent import Agent
from smartgrid.rewards.reward import Reward
from smartgrid.util.equity import hoover
from smartgrid.world import World


class EquityRewardPerAgent(Reward):
    """
    Reward based on the equity of comforts measure.

    It's a measure of statical dispersion of the Comfort metrics of all agents.
    The Comfort metric compute by the function field attach to :py:class:`.AgentProfile`, after that the comfort is
    stored in :py:class:`.AgentState`.
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
