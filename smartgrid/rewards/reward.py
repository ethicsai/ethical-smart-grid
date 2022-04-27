from abc import ABC, abstractmethod

from smartgrid.agents.agent import Agent
from smartgrid.world import World


class Reward(ABC):
    """
    The class for representing a Reward.
    A reward is always a single value (See RewardCollection for multiple Reward)
    """
    name: str

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def calculate(self, world: World, agent: Agent) -> float:
        """
        Methods for having a reward depending on:
        :param world: representation of the physical environment
        :param agent: indicate information for calculating his reward
        """
        # todo add params
        pass
