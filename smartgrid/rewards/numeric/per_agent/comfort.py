from smartgrid.agents.agent import Agent
from smartgrid.rewards.reward import Reward
from smartgrid.world import World


class Comfort(Reward):
    """
    Comfort is the metric compute by the function field attach to :py:class:`.AgentProfile`, after that the comfort is
    stored in :py:class:`.AgentState`.
    """

    def __init__(self):
        super().__init__("Comfort")

    def calculate(self, world: World, agent: Agent):
        return agent.state.comfort
