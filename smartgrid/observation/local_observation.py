from abc import ABC, abstractmethod

from smartgrid.agents.agent import Agent
from smartgrid.world import World


#TODO herit from nameTuple
class LocalObservation(ABC):
    """
    Observation for a single Agent, it contains:
        - personal_storage: the amount of energy in the battery.
        - comfort: represent the fulfilled of his need with a Richard_curve
        - payoff: money earn in selling energy.
    """
    personal_storage: float
    comfort: float
    payoff: float

    @abstractmethod
    def compute(self, world: World, agent: Agent):
        """
        Compute all metric for the local observation.
        """
        pass


class BaseLocal(LocalObservation):

    def compute(self, world: World, agent: Agent):
        # Individual data
        self.personal_storage = agent.storage_ratio
        self.comfort = agent.state.comfort
        self.payoff = agent.payoff_ratio
