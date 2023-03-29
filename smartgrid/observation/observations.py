"""
Observations are information that agents receive about the environment.
"""

from __future__ import annotations

from collections import namedtuple

import numpy as np
from gymnasium.spaces import Box

from .global_observation import GlobalObservation
from .local_observation import LocalObservation


class Observation(namedtuple('Observation', LocalObservation._fields + GlobalObservation._fields)):
    """
    Observations are information that agents receive about the environment.

    They describe the current state of the environment through various metrics,
    and are used to take decisions (actions).
    We represent Observations as named tuples, so that these metrics can be
    easily accessed by name, e.g., ``obs.hour``.

    Most algorithms use NumPy vectors, or even TensorFlow / PyTorch tensors,
    to represent observations, instead of named tuples. To simplify their use,
    instances of this class can be transformed into NumPy vectors through the
    usual :py:func:`numpy.asarray`, e.g., ``np.asarray(obs)`` returns the
    NumPy vector corresponding to the named tuple ``obs``.

    The *observation space*, i.e., the domain in which observations take their
    values, can be programmatically accessed at runtime through the
    :py:meth:`Observation.get_observation_space` class method.
    """

    @classmethod
    def get_observation_space(cls):
        """
        Describes the space in which Observations take their values.

        This method is useful if an algorithm has assumptions or requirements
        on the observation space. For example, values can be interpolated,
        by knowing their original domain.

        :rtype: gym.spaces.Box
        :return: A gym Box, whose ``low`` field indicates the minimum value
            of each element of the observation vector. Similarly, the
            ``high`` field indicates the maximum value of each element, such
            that each element *i* of the vector is contained between ``low[i]``
            and ``high[i]``. The Box's shape is the number of fields.
        """
        return Box(low=0.0, high=1.0, shape=(len(cls._fields),))

    def check_between_0_and_1(self):
        """
        Verifies that values are in the correct interval.

        This method is mainly used internally, when a new Observation is
        created, to ensure that the vector's values are correct, w.r.t. the
        space.

        If some values are not in [0,1], they are signalled (name and value)
        through a warning.
        """
        errors = {k: v for k, v in self._asdict().items() if v < 0.0 or v > 1.0}
        if len(errors) > 0:
            import warnings
            warnings.warn('Incorrect observations, not in [0,1]: {}'.format(errors),
                          stacklevel=2)

    def __array__(self) -> np.ndarray:
        """
        Magic method that simplifies the translation into NumPy vectors.

        This method should usually not be used directly; instead, it allows
        using the well-known NumPy's ``asarray`` function to transform an
        instance of :py:class:`.Observation` into a NumPy :py:class:`np.ndarray`.

        The resulting vector's values are guaranteed to be in the same order
        as the Observation's fields, see :py:attr:`.Observation._fields`.
        """
        # Using `[*values()]` seems more efficient than other methods
        # e.g., `list(values())` or `values()` directly.
        return np.array([*self._asdict().values()])

    @classmethod
    def create(cls, local_observation: LocalObservation, global_observation: GlobalObservation):
        """
        Create a new Observation from existing local and global ones.

        The fields of both :py:class:`.LocalObservation` and
        :py:class:`.GlobalObservation` are merged.

        :rtype: Observation
        """
        obs = cls(hour=global_observation.hour,
                  available_energy=global_observation.available_energy,
                  personal_storage=local_observation.personal_storage,
                  comfort=local_observation.comfort,
                  payoff=local_observation.payoff,
                  equity=global_observation.equity,
                  energy_loss=global_observation.energy_loss,
                  autonomy=global_observation.autonomy,
                  exclusion=global_observation.exclusion,
                  well_being=global_observation.well_being,
                  over_consumption=global_observation.over_consumption)
        obs.check_between_0_and_1()

        return obs
