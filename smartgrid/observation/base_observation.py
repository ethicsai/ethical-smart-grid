"""
Base classes to simplify definition of Observations.
"""
import abc
import dataclasses
from typing import Tuple, Dict, Any

import numpy as np
from gymnasium.spaces import Space, Box


@dataclasses.dataclass(frozen=True)
class BaseObservation:
    """
    Base class for defining any kind of Observation (global, local, "total").

    Observations are information that agents receive about the environment.
    They describe the current state of the environment through various metrics,
    and are used to take decisions (actions).

    This class, although being a dataclass, does not define any attribute.
    Instead, it defines several helper methods, to avoid duplicating them
    in the various Observations classes:

    - :py:meth:`~smartgrid.observation.base_observation.BaseObservation.fields`,
      which lists the fields that an Observation class possesses.
    - :py:meth:`~smartgrid.observation.base_observation.BaseObservation.asdict`,
      which returns a dictionary representation of an Observation.
    - :py:meth:`~smartgrid.observation.base_observation.BaseObservation.space`,
      which returns the :py:class:`~gymnasium.spaces.Space` in which
      Observations live (i.e., which values they can take).
    - a magic method that allows to easily create NumPy arrays by using the
      standard :py:func:`numpy.asarray` method, such as: ``np.asarray(obs)``.

    Classes that extend this BaseObservation can thus be used either:

    - in a user-friendly manner, as a dataclass, by accessing the fields by
      their names, getting the fields names;
    - or in a programmatic manner, by converting them to NumPy arrays, which
      are commonly used in learning algorithms, as inputs of neural networks
      (tensors).
    """

    @classmethod
    def fields(cls) -> Tuple[str]:
        """
        Returns the names of fields that compose an Observation.

        Fields can be excluded by setting the metadata ``include`` custom
        property to ``False``, such as:
        ``my_field: Any = field(metadata={'include': False})``.

        :param cls: Either the class itself, or an instance of the class; this
            method supports both. In other words, it can be used as
            ``Observation.fields()``, or ``obs = Observation(...); obs.fields()``.

        :return: The fields' names as a tuple, in their order of definition.
        """
        fields = dataclasses.fields(cls)
        # `fields` is a tuple of `Field` objects, we only want their names.
        fields = tuple(
            field.name
            for field in fields
            if field.metadata.get('include', True)
        )
        return fields

    def asdict(self) -> Dict[str, Any]:
        """
        Return the Observation as a dictionary.

        Fields can be excluded by setting the metadata ``include`` custom
        property to ``False``, such as:
        ``my_field: Any = field(metadata={'include': False})``.

        :param self: An instance of observation.

        :return: The observation represented as a dictionary, with the fields'
            names as keys and the fields' values as values, in the order of
            definition.
        """
        # Although `asdict` has a `dict_factory` parameter that we could use,
        # it would not be a good idea, because it would be used recursively
        # for anything contained inside this dataclass. Instead, we get the
        # "whole" dictionary.
        d = dataclasses.asdict(self)
        # Now, we must remove the (potential) elements that should be excluded.
        included_fields = self.fields()
        return {
            key: d[key]
            for key in included_fields
            if key in d
        }

    @classmethod
    def space(cls, world: 'World', agent: 'Agent') -> Space:
        """
        Describe the space in which Observations take their values.

        This method is useful if an algorithm has assumptions or requirements
        on the observation space. For example, values can be interpolated,
        by knowing their original domain.

        We currently use ratios in ``[0, 1]`` for each metric of observations.
        This makes it easier for learning algorithms (avoids perceiving a
        given dimension as more important than another because of an extended
        range). It also means that the ``world`` and ``agent`` parameters do
        not influe on the space (they could be ``None``).

        In the future, we could use the true ranges from the agent's
        :py:class:`~smartgrid.agents.profile.AgentProfile` and let users
        convert these observations to ``[0, 1]`` when necessary. This would
        provide more useful information, e.g., the actual battery storage
        in ``[0, capacity]``, rather than a ratio, or the actual hour in
        ``[0, 23]`` rather than a value ``(h % 24) / 24``, which is hard to
        interpret for human users.

        :param world: The :py:class:`~smartgrid.world.World` instance in which
            observations will be computed. Currently unused.
        :param agent: The :py:class:`~smartgrid.agents.agent.Agent` instance
            of the agent for which we compute observations. Currently unused.

        :rtype: gymnasium.spaces.Box
        :return: A gym Box, whose ``low`` field indicates the minimum value
            of each element of the observation vector. Similarly, the
            ``high`` field indicates the maximum value of each element, such
            that each element *i* of the vector is contained between ``low[i]``
            and ``high[i]``. The Box's shape is the number of fields.
        """
        nb_fields = len(cls.fields())
        return Box(
            low=np.asarray([0.0] * nb_fields),
            high=np.asarray([1.0] * nb_fields),
            # We use float64, as the (default) float32 raises a warning
            # about the bounds' precision.
            dtype=np.float64
        )

    def __array__(self) -> np.ndarray:
        """
        Magic method that simplifies the translation into NumPy arrays.

        This method should usually not be used directly; instead, it allows
        using the well-known :py:func:`numpy.asarray` function to transform
        an instance of Observation into a NumPy :py:class:`np.ndarray`.

        Fields can be excluded by setting the metadata ``include`` custom
        property to ``False``, such as:
        ``my_field: Any = field(metadata={'include': False})``.

        The resulting array's values are guaranteed to be in the same order
        as the Observation's fields, see :py:meth:`~.fields`.
        """
        # Using `[*values()]` seems more efficient than other methods
        # e.g., `list(values())` or `values()` directly.
        return np.array([*self.asdict().values()])


class Observation(BaseObservation, abc.ABC):
    """
    Represents a merged observation of both global and local metrics.

    This class is only used for typing purposes; please see the
    :py:class:`~smartgrid.observation.observation_manager.ObservationManager`
    for the actual, dynamically-created, class.
    """

    @classmethod
    @abc.abstractmethod
    def create(
            cls,
            global_obs: BaseObservation,
            local_obs: BaseObservation
    ) -> 'Self':
        pass

    @abc.abstractmethod
    def get_global_observation(self):
        pass

    @abc.abstractmethod
    def get_local_observation(self):
        pass
