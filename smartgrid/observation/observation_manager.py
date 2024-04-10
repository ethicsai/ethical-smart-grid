"""
The ObservationManager is responsible for computing observations.
"""
import dataclasses
from typing import Dict, Type

from smartgrid.agents import Agent
from smartgrid.world import World
from .base_observation import BaseObservation, Observation
from .global_observation import GlobalObservation
from .local_observation import LocalObservation


def _create_observation_type(
        global_observation_type: Type[GlobalObservation],
        local_observation_type: Type[LocalObservation]
) -> Type[Observation]:
    """
    Create a new class that represents an Observation.

    An Observation merges data from both Global and Local observations.
    """
    @dataclasses.dataclass(frozen=True)
    class _Observation(
        global_observation_type,
        local_observation_type,
        Observation
    ):
        # Add specific "fields" to get the original "global_obs" and "local_obs"
        # objects. They should not be used in object representation, nor in
        # comparisons, nor in `fields` and `asdict`, ... Basically, not anywhere.
        _global_obs: global_observation_type = dataclasses.field(
            repr=False,
            compare=False,
            metadata={'include': False}
        )
        _local_obs: local_observation_type = dataclasses.field(
            repr=False,
            compare=False,
            metadata={'include': False}
        )

        # Override the qualname so that the `str` method returns a non-garbage
        # (easily understandable) class name. By default, it would return
        # `_create_observation_type.<locals>._Observation(personal_storage=...)`
        # which is ugly and hard to understand. `'Observation'` is much better,
        # even though it is not exactly correct (the class is indeed defined as
        # a local variable of a function call, but that is not important to
        # the third-party users).
        __qualname__ = 'Observation'

        @classmethod
        def create(cls,
                   global_observation: global_observation_type,
                   local_observation: local_observation_type):
            obj = cls(
                **global_observation.asdict(),
                **local_observation.asdict(),
                _global_obs=global_observation,
                _local_obs=local_observation,
            )
            return obj

        def get_global_observation(self):
            return self._global_obs

        def get_local_observation(self):
            return self._local_obs

    return _Observation


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
    The class that represents the "whole" observation (local and global).

    It combines fields from the :py:attr:`.global_observation` and
    :py:attr:`.local_observation` dataclasses. Because these two attributes
    are set at runtime, this class is dynamically created. To simplify usage,
    it supports the methods defined in :py:class:`.BaseObservation` (``fields``,
    ``asdict``, and transformation to NumPy array with ``np.asarray``).
    """

    def __init__(
            self,
            local_observation: Type[LocalObservation] = LocalObservation,
            global_observation: Type[GlobalObservation] = GlobalObservation,
    ):
        self.global_observation = global_observation
        self.local_observation = local_observation
        self.observation = _create_observation_type(global_observation,
                                                    local_observation)

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

    def compute(self, world: World, agent: Agent) -> Observation:
        global_obs = self.compute_global(world)
        local_obs = self.compute_agent(world, agent)
        return self.observation.create(global_obs, local_obs)

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
        nb_local = len(self.local_observation.fields())
        nb_global = len(self.global_observation.fields())
        return {
            "agent_state": nb_local + nb_global,
            "local_state": nb_local,
            "global_state": nb_global
        }

    def reset(self):
        """
        Reset the ObservationManager.

        It is particularly important to reset the memoization process of
        :py:class:`.GlobalObservation`.
        """
        self.global_observation.reset()
        self.local_observation.reset()
