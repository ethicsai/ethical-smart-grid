"""
The ObservationManager is responsible for computing observations.
"""

from typing import Dict, Type

from smartgrid.agents import Agent
from smartgrid.world import World
from .observations import Observation
from .global_observation import GlobalObservation
from .local_observation import LocalObservation


class ObservationManager:
    """
    The ObservationManager is responsible for computing observations.

    Its primary purpose is to allow extensibility: the attributes
    :py:attr:`.global_observation` and :py:attr:`.local_observation`, which
    are set through the constructor, control which Observation classes will
    be used in the simulator. It is thus possible to subclass
    :py:class:`.GlobalObservation` and/or :py:class:`.LocalObservation` to
    use different observations.

    The computing calls (:py:meth:`.compute_agent` and :py:meth:`.compute_global`)
    are delegated to the corresponding calls through these attributes.
    """

    global_observation: Type[GlobalObservation]
    """
    The class that will be used to compute global observations.
    It should be a subclass of :py:class:`.GlobalObservation` to ensure that
    necessary methods are present.
    Please note that this field should be set to a *class* itself, not an
    instance, e.g., ``GlobalObservation`` (instead of ``GlobalObservation()``).
    """

    local_observation: Type[LocalObservation]
    """
    The class that will be used to compute local observations.
    It should be a subclass of :py:class:`.LocalObservation` to ensure that
    necessary methods are present.
    Please note that this field should be set to a *class* itself, not an
    instance, e.g., ``LocalObservation`` (instead of ``LocalObservation()``).
    """

    observation: Type[Observation]
    """
    The class that will be used as the "complete" observation.
    """

    def __init__(self,
                 local_observation: Type[LocalObservation] = LocalObservation,
                 global_observation: Type[GlobalObservation] = GlobalObservation,
                 observation: Type[Observation] = Observation):
        self.global_observation = global_observation
        self.local_observation = local_observation
        self.observation = observation

    def compute_agent(self, world: World, agent: Agent) -> LocalObservation:
        """
        Create the local observation for an Agent.
        """
        return self.local_observation.compute(world, agent)

    def compute_global(self, world) -> GlobalObservation:
        """
        Create the global observation for the World.
        """
        return self.global_observation.compute(world)

    @property
    def shape(self) -> Dict[str, int]:
        """
        Describe the shapes of the various Observations (local, global, merged).

        :rtype: dict
        :return: A dict comprised of: ``agent_state``, ``local_state``, and
            ``global_state``. Each of these fields describe the shape (i.e.,
            number of dimensions) of the corresponding observation. Note that
            ``agent_state`` refers to the merged (both local and global) case.
        """
        return {"agent_state": len(self.local_observation._fields) + len(self.global_observation._fields),
                "local_state": len(self.local_observation._fields),
                "global_state": len(self.global_observation._fields)}

    def reset(self):
        """
        Reset the ObservationManager.

        It is particularly important to reset the memoization process of
        :py:class:`.GlobalObservation`.
        """
        self.global_observation.reset()
        self.local_observation.reset()
