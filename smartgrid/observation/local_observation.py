from abc import abstractmethod
from collections import namedtuple

from smartgrid.agents.agent import Agent

local_field = [
    'personal_storage',
    'comfort',
    'payoff',
]


class LocalObservation(namedtuple('LocalObservation', local_field)):
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
    def compute(self, world: 'World', agent: Agent):
        """
        Compute all metric for the local observation.
        """
        # Individual data
        self.personal_storage = agent.storage_ratio
        self.comfort = agent.state.comfort
        self.payoff = agent.payoff_ratio
