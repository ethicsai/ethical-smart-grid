"""
Observations that are local (individual) to a single Agent.
"""

import dataclasses
from typing import Tuple, Dict

import numpy as np
from gymnasium.spaces import Space, Box

from smartgrid.agents import Agent


@dataclasses.dataclass(frozen=True)
class LocalObservation:
    """
    Observations that are local (individual) to a single Agent.

    Observations cannot be modified once created, to limit potential bugs.
    Local observations are not shared with other agents, and contain the
    following measures:

    personal_storage
        The amount of energy currently available in the :py:class:`.Agent`'s
        personal battery.
        This amount is represented as a ratio between 0 (empty) and 1 (full),
        w.r.t. the Agent's battery capacity. See :py:attr:`.Agent.storage_ratio`
        for details.

    comfort
        This represents to which degree the agent satisfied its need by
        consuming energy. Intuitively, the more an agent's consumption is
        close to its need, the closer the comfort will be to 1. Conversely,
        if an agent does not consume, its comfort will tend towards 0.
        Comfort is computed through the Agent's comfort function; we describe
        several examples in the :py:mod:`~smartgrid.agents.profile.comfort`
        module, which rely on *generalized logistic curves* (similar to a
        sigmoid).

    payoff
        The agent's current amount of money. Money can be won by selling
        energy from the personal battery to the national grid, or lost by
        buying money from the national grid to the personal battery.
        The payoff observation is interpolated from the agent's real payoff
        and the payoff range to obtain a value between 0 (a loss) and 1 (a win),
        with 0.5 being the neutral value (neither win nor loss).
    """

    personal_storage: float
    """
    The ratio of energy available in the agent's personal storage, over capacity.
    """

    comfort: float
    """
    The agent's comfort, a value in ``[0,1]`` based on its consumption and need.
    """

    payoff: float
    """
    The agent's current payoff, expressed as a ratio in ``[0,1]`` based on
    maximal and minimal allowed values.
    """

    @classmethod
    def compute(cls, world: 'World', agent: Agent) -> 'Self':
        """
        Return local observations for a single agent.

        This function extracts the relevant measures from an :py:class:`.Agent`.
        Most of the computing has already been done in the
        :py:meth:`.Agent.update` and :py:meth:`.Agent.handle_action` methods.

        :param world: The World in which the Agent is contained, for eventual
            data stored outside the agent.

        :param agent: The Agent for which we want to compute the local
            observations.

        :rtype: LocalObservation
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

    @classmethod
    def reset(cls):
        """
        Reset the LocalObservation class.

        This method currently does nothing but is implemented to mirror the
        behaviour of :py:class:`.GlobalObservation`, and to allow extensions
        to use complex mechanisms that require a ``reset``.
        """
        pass

    @classmethod
    def fields(cls) -> Tuple[str]:
        """
        Returns the names fields that compose a LocalObservation, as a tuple.

        :param cls: Either the class itself, or an instance of the class; this
            method supports both. In other words, it can be used as
            ``LocalObservation.fields()``, or
            ``obs = LocalObservation(...); obs.fields()``.

        :return: The fields' names as a tuple, in their order of definition.
            For the basic LocalObservation, this corresponds to
            ``('personal_storage', 'comfort', 'payoff')``.
        """
        fields = dataclasses.fields(cls)
        # `fields` is a tuple of `Field` objects, we only want their names.
        fields = tuple(field.name for field in fields)
        return fields

    @classmethod
    def space(cls, world: 'World', agent: Agent) -> Space:
        """
        Returns the Space in which LocalObservations live.
        """
        # We currently use ratios (values in `[0,1]`) for each observation.
        # In the future, maybe we could return the true value from the agent's
        # state? The Space would then depend on the agent.
        return Box(
            low=np.asarray([0.0, 0.0, 0.0]),
            high=np.asarray([1.0, 1.0, 1.0]),
            # We use float64, as the (default) float32 raises a warning
            # about the bounds' precision.
            dtype=np.float64
        )

    def asdict(self) -> Dict[str, float]:
        """
        Return the LocalObservation as a dictionary.
        """
        return dataclasses.asdict(self)

    def __array__(self) -> np.ndarray:
        """
        Magic method that simplifies the translation into NumPy arrays.

        This method should usually not be used directly; instead, it allows
        using the well-known :py:func:`numpy.asarray` function to transform
        an instance of :py:class:`.LocalObservation` into a NumPy
        :py:class:`np.ndarray`.

        The resulting array's values are guaranteed to be in the same order
        as the LocalObservation's fields, see :py:meth:`.fields`.
        """
        # Using `[*values()]` seems more efficient than other methods
        # e.g., `list(values())` or `values()` directly.
        return np.array([*self.asdict().values()])
