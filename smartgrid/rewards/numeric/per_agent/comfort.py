from smartgrid.agents.agent import Agent
from smartgrid.rewards.reward import Reward
from smartgrid.world import World


class Comfort(Reward):
    """
    Uses the agent's comfort directly as a reward.

    This reward function simply encourages the agent to increase its comfort.
    It is best used in addition with other functions that encourage other
    moral values, such as equity or preventing over-consumption, to avoid the
    agent optimizing its comfort by consuming as much as allowed.
    """

    name: str

    def __init__(self):
        super().__init__()

    def calculate(self, world: World, agent: Agent):
        return agent.state.comfort
