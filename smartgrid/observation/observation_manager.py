from abc import abstractmethod

from smartgrid.agents.agent import Agent
from smartgrid.observation.global_observation import GlobalObservation
from smartgrid.observation.local_observation import LocalObservation
from smartgrid.observation.observations import Observation


class ObservationManager:
    """
    ObservationManager is a handler of the computation of the observation for one Agent.
    It's fusion global and local observation.
    """
    global_observation: GlobalObservation
    local_observation: LocalObservation

    def __init__(self, local_observation: LocalObservation, global_observation: GlobalObservation):
        self.global_observation = global_observation
        self.local_observation = local_observation

    @abstractmethod
    def compute(self, world: 'World', agent: Agent) -> Observation:
        """
        Create the observation for an Agent.
        :param world: use for global and local observation
        :param agent: use for local observation
        """
        self.global_observation.compute(world)
        self.local_observation.compute(world, agent)

        return Observation.create(self.local_observation, self.global_observation)

    def reset(self):
        self.global_observation.reset()
