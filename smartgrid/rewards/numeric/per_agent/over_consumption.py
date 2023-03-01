from smartgrid.agents.agent import Agent
from smartgrid.rewards.reward import Reward
from smartgrid.world import World


class OverConsumptionPerAgent(Reward):
    """
    Reward representing the overConsumption percentage of an Agent.
    """

    def __init__(self):
        super().__init__("OverConsumptionPerAgent")

    def calculate(self, world: World, agent: Agent):
        # The energy taken from the grid by each agent
        sum_taken = 0.0
        for a in world.agents:
            sum_taken += a.enacted_action.grid_consumption
        # Global reward
        take_by_agent = agent.enacted_action.grid_consumption + agent.enacted_action.store_energy

        local_oc = 1 - take_by_agent / (sum_taken + 10E-300)

        return local_oc
