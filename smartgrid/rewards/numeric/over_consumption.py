from smartgrid.agents.agent import Agent
from smartgrid.rewards.reward import Reward
from smartgrid.world import World


class OverConsumption(Reward):
    """
    Reward representing the overConsumption percentage of an Agent.
    """

    def __init__(self):
        super().__init__("OverConsumption")

    def calculate(self, world: World, agent: Agent):
        # The total quantity of energy over-consumed
        global_oc = world.get_observation_global().over_consumption
        # The energy taken from the grid by each agent
        sum_taken = 0.0
        for a in world.agents:
            sum_taken += a.enacted_action.grid_consumption
        # Global reward
        global_reward = 1.0 - global_oc / (sum_taken + 10E-300)

        take_by_agent = agent.enacted_action.grid_consumption \
                        + agent.enacted_action.store_energy

        local_oc = global_oc - take_by_agent / (sum_taken + 10E-300)

        return global_reward - local_oc
