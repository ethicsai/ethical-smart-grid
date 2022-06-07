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

    @classmethod
    def compute(cls, agent: Agent):
        """
        Compute all metric for the local observation.
        """
        # Individual data
        personal_storage = agent.storage_ratio
        comfort = agent.comfort
        payoff = agent.payoff_ratio

        return cls(
            personal_storage=personal_storage,
            comfort=comfort,
            payoff=payoff,
        )
