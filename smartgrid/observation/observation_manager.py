from abc import abstractmethod

from smartgrid.agents.agent import Agent
from smartgrid.observation.global_observation import GlobalObservation
from smartgrid.observation.local_observation import LocalObservation
from smartgrid.observation.observations import Observation


# TODO herit from nameTuple
class ObservationManager(ABC):
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
    def compute(self, world: World, agent: Agent) -> Observation:
        """
        Create the observation for an Agent.
        :param world: use for global and local observation
        :param agent: use for local observation
        """
        pass

    def reset(self):
        self.global_observation.reset()


class BaseObservationManager(ObservationManager):

    def __init__(self):
        super().__init__(BaseLocal(), BaseGlobal())

    def compute(self, world: World, agent: Agent):
        self.global_observation.compute(world)
        self.local_observation.compute(world, agent)

    def compute_global(self, world) -> GlobalObservation:
        return self.global_observation.compute(world)

    @property
    def shape(self) -> Dict[str, int]:
        return {"agent_state": len(self.local_observation._fields)+len(self.global_observation._fields),
                "local_state": len(self.local_observation._fields),
                "global_state": len(self.global_observation._fields)}

    def reset(self):
        self.global_observation.reset()
