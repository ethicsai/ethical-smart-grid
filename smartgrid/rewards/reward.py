from abc import ABC, abstractmethod

from smartgrid.agents.agent import Agent
from smartgrid.world import World


class Reward(ABC):
    name: str

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def calculate(self, world: World, agent: Agent):
        # todo add params
        pass
