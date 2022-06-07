from abc import abstractmethod
from typing import Dict, Type

from smartgrid.agents.agent import Agent
from smartgrid.observation.global_observation import GlobalObservation
from smartgrid.observation.local_observation import LocalObservation


class ObservationManager:
    """
    ObservationManager is a handler of the computation of the observation for one Agent.
    It's fusion global and local observation.
    """
    global_observation: Type[GlobalObservation]
    local_observation: Type[LocalObservation]

    def __init__(self, local_observation: Type[LocalObservation], global_observation: Type[GlobalObservation]):
        self.global_observation = global_observation
        self.local_observation = local_observation

    @abstractmethod
    def compute_agent(self, world: 'World', agent: Agent) -> LocalObservation:
        """
        Create the observation for an Agent.
        :param world: use for global and local observation
        :param agent: use for local observation
        """
        return self.local_observation.compute(agent)

    def compute_global(self, world) -> GlobalObservation:
        return self.global_observation.compute(world)

    @property
    def shape(self) -> Dict[str, int]:
        return {"agent_state": len(self.local_observation._fields) + len(self.global_observation._fields),
                "local_state": len(self.local_observation._fields),
                "global_state": len(self.global_observation._fields)}

    def reset(self):
        self.global_observation.reset()
