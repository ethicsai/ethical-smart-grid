from agents.agent import Agent
from rewards.reward import Reward
from world import World


class OverConsumption(Reward):
    """
    Reward representing the *over-consumption* of an Agent.

    The *over-consumption* is the quantity of energy that was consumed by
    the society of agents, but which was not available in the grid.
    (We assume that the grid automatically buys from the national grid to
    compensate, which has a negative impact. Over-consumption should thus be
    avoided).

    We compare the quantity of energy *taken* (i.e., consumed + stored from
    the grid) by all agents to the quantity of energy over-consumed by all
    agents. This gives us a *global* component (the current environment).

    Then, we compare the quantity of energy over-consumed, minus the agent's
    taken energy, and we compare to the sum of energy taken by all agents.
    This gives us a *local* component (the hypothetical environment, had the
    agent not acted).

    The reward follows the Difference Reward principle, and thus is the global
    component minus the local component.
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

        take_by_agent = agent.enacted_action.grid_consumption + agent.enacted_action.store_energy

        local_oc = global_oc - take_by_agent / (sum_taken + 10E-300)

        return global_reward - local_oc
