from abc import abstractmethod
from collections import namedtuple

from smartgrid.agents.agent import Agent



class LocalObservation(namedtuple('LocalObservation', local_field)):
    """
    Observation for a single Agent, it contains:
        - personal_storage: the amount of energy in the battery.
        - comfort: represent the fulfilled of his need with a Richard_curve
        - payoff: money earn in selling energy.
    """

    @classmethod
    def compute(cls, world: World, agent: Agent):
        pass


class BaseLocal(LocalObservation):

    def compute(self, world: World, agent: Agent):
        # Individual data
        personal_storage = agent.storage_ratio
        comfort = agent.comfort
        payoff = agent.payoff_ratio

        return cls(
            personal_storage=personal_storage,
            comfort=comfort,
            payoff=payoff,
        )
