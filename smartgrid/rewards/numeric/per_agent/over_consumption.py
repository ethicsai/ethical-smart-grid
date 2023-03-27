from smartgrid.rewards.reward import Reward


class OverConsumptionPerAgent(Reward):
    """
    Reward representing the overConsumption percentage of an Agent.
    """

    name: str

    def __init__(self):
        super().__init__()

    def calculate(self, world, agent):
        # The energy taken from the grid by each agent
        sum_taken = 0.0
        for a in world.agents:
            sum_taken += a.enacted_action.grid_consumption
            sum_taken += a.enacted_action.store_energy

        # Energy taken by the current agent only
        take_by_agent = agent.enacted_action.grid_consumption + agent.enacted_action.store_energy
        # Proportion of energy taken by current agent compared to the sum
        local_oc = 1 - take_by_agent / (sum_taken + 10E-300)

        return local_oc
