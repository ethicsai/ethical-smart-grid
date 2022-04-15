from smartgrid.agents.agent import Agent
from smartgrid.rewards.reward import Reward
from smartgrid.world import World


class Comfort(Reward):
    """
    Comfort is already compute in state
    """
    def __init__(self):
        super().__init__("Comfort")

    def calculate(self, world: World, agent: Agent):
        return agent.state.comfort
