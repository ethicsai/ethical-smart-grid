"""
This module defines classes to generate an amount of available energy.

Each step, the :py:class:`World` "produces" a certain amount of energy, which
is made available to the :py:class:`Agent`\\ s. To adapt to various sizes of
environments(i.e., number of Agents), generators are given access to the
*total need* of all Agents in the World. This allows adapting to any number
of agents, and any :py:class:`AgentProfile`\\ s.
Note that this *total need* can be ignored, leading to a generator which
ignores the number of Agents.

Several generators are implemented in this module, using various methods:

- a random percent based on the agents' needs, for example an amount between
  80% and 120% of their total need.
- a scarcity variation, similar to the 1st one but with a random between
  60% and 80%.
- a generous variation, similar to the 1st one but with a random between
  100% and 120%.
- a realistic variation, using real data.

All these methods lead to different bounds for the amount of available energy.

Knowing these bounds, and especially the upper one (we can assume `0` for
the lower bound), allows us to scale the amount of available energy to `[0,1]`
when computing :py:class:`Observation`\\ s.

Therefore, instead of using a simple function to generate this amount,
we use a class that defines 2 functions, one for generating the amount,
and the other to return the bounds.
"""

from __future__ import annotations

import random
import warnings
from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np


class EnergyGenerator(ABC):
    """
    An *EnergyGenerator* is responsible for the production of energy each step.
    """

    def __str__(self):
        return type(self).__name__

    @abstractmethod
    def generate_available_energy(self,
                                  current_need: int,
                                  current_step: int,
                                  min_need: int,
                                  max_need: int) -> int:
        """
        Generate an amount of available energy during a single time step.

        EnergyGenerators can use any method to compute this amount, e.g.,
        returning a fixed value, drawing from a distribution, using an array
        of realistic data for each time step, etc.

        :param current_need: The total energy needed by all agents at the
            current step. This value can be used to effectively scale the
            generator to the current agent population. This value can also
            be ignored by the generator. The ``current_need`` should be
            comprised between ``min_need`` and ``max_need``.

        :param current_step: The current time step. Mostly used by "realistic"
            or data-based generators that need to know the current date/hour.

        :param min_need: The minimum energy needed by all agents, for all time
            steps. It does not need to be exact, and only serve as a lower
            bound, e.g., ``0`` is a perfectly sane value. However, the more
            accurate this value is, the more accurate the scaling of the
            :py:class:`.Observation` space will be.

        :param max_need: The maximum energy needed by all agents, for all time
            steps. It does not need to be exact, and only serves as an upper
            bound, e.g., any sufficiently high value can be used. However,
            the more accurate this value is, the more accurate the scaling of
            the :py:class:`.Observation` space will be.

        :return: The amount of available energy, a value in :math:`\\mathbb{R}`.
            For example, returning ``40_000`` means that 40,000Wh are available
            for the current time step.
        """
        pass

    @abstractmethod
    def available_energy_bounds(self,
                                current_need: int,
                                current_step: int,
                                min_need: int,
                                max_need: int) -> Tuple[int, int]:
        """
        Determine the possible min and max bounds for the energy generation.

        This method is used to provide a range (a domain), which is important
        for specifying the :py:class:`.Observation` space of :py:class:`.Agent`\\ s.
        This also allows scaling the generated amount to ``[0,1]``. For example,
        assuming the bounds are ``[0, 10_000]``, a generated amount of ``8_000``
        can be scaled to ``0.8``, which is easier to use by learning algorithms.

        :param current_need: The total energy needed by all agents at the
            current step.

        :param current_step: The current time step.

        :param min_need: The minimum energy needed by all agents, for all time
            steps. This value is used for the same objective as ``max_need``,
            but it does not need to be accurate, e.g., ``0`` can be used as a
            safe default.

        :param max_need: The maximum energy needed by all agents, for all time
            steps. This value can be used to accurately determine the bounds
            for all time steps, instead of a single time step. The ``need_at_step``
            should always be lower or equal to the ``max_need``.

        :return: The min and max bounds of the energy generator, i.e., the
            minimum and maximum possible values that
            :py:meth:`.generate_available_energy` may return. It is important
            that these bounds are coherent with the method, otherwise scaling
            may not work properly, and Agents may receive incorrect observations.

        .. note: To avoid changing the Observation space, the available energy
            bounds *should not* shift from a time step to another. In other
            words, this method *should* return the same bounds for any value
            of ``current_need`` and ``current_step``. However, the code structure
            intentionally allows not respecting this, to avoid restricting
            potential experiments. It can be considered as a "not recommended"
            setup.
        """
        pass


class RandomEnergyGenerator(EnergyGenerator):
    """
    Generate a random amount, with respect to the agents' current energy needed.

    Assuming that the total maximum energy needed is ``M``, that we want at least
    a lower bound of L=80% (i.e., L=0.8), and an upper bound of U=120% (i.e.,
    U=1.2), this class returns amounts in the interval ``[L*M, U*M]``.

    Knowing the minimum sum of energy needed by all agents ``minM``, we derive
    that the lowest amount of energy that can be produced by this generator
    is ``L*minM``, for any time step.
    Similarly, assuming the maximum sum is ``maxM``, the highest amount that
    can be produced is ``U*maxM``.
    Thus, this generator's possible bounds are ``[L*minM, U*maxM]``.

    Lower and upper bounds are configurable.
    """

    lower: float
    """Lower bound for generating energy, in proportion of the total need."""

    upper: float
    """Upper bound for generating energy, in proportion of the total need."""

    def __init__(self,
                 lower_proportion=0.8,
                 upper_proportion=1.2,
                 ):
        self.lower = lower_proportion
        self.upper = upper_proportion

    def generate_available_energy(self,
                                  current_need: int,
                                  current_step: int,
                                  min_need: int,
                                  max_need: int):
        if not min_need <= current_need <= max_need:
            warnings.warn('Incoherent current need and min/max needs; '
                          f'found min={min_need}, current_need={current_need}, '
                          f'max_need={max_need}. Continuing, but the result '
                          'may be incoherent with the possible bounds.')
        lower_bound = int(self.lower * current_need)
        upper_bound = int(self.upper * current_need)
        return random.randint(lower_bound, upper_bound)

    def available_energy_bounds(self,
                                current_need: int,
                                current_step: int,
                                min_need: int,
                                max_need: int):
        lower_bound = int(self.lower * min_need)
        upper_bound = int(self.upper * max_need)
        return lower_bound, upper_bound


class ScarceEnergyGenerator(RandomEnergyGenerator):
    """
    Similar to the :py:class:`.RandomEnergyGenerator`, but simulating scarcity.

    In practice, the bounds are set to [60%, 80%].
    Note that, as the upper bound is set to less 100% of the max, we
    force conflicts between agents by not giving them enough.
    """

    lower: float
    upper: float

    def __init__(self):
        super(ScarceEnergyGenerator, self).__init__(
            lower_proportion=0.6,
            upper_proportion=0.8,
        )


class GenerousEnergyGenerator(RandomEnergyGenerator):
    """
    Similar to the :py:class:`.RandomEnergyGenerator`, but simulating a generous env.

    In practice, the bounds are set to [100%, 120%].
    Note that, as the lower bound is set to 100% of the max, we always
    have enough energy available for all agents.
    """

    lower: float
    upper: float

    def __init__(self):
        super(GenerousEnergyGenerator, self).__init__(
              lower_proportion=1.0,
              upper_proportion=1.2,
        )


class RealisticEnergyGenerator(EnergyGenerator):
    """
    A realistic generator that generates energy based on real-world data.

    The ``data`` parameter should be a NumPy ndarray giving the ratio of
    energy for each step, with respect to the maximum amount of energy
    needed by the agents.

    For example, ``[0.3, 0.8, 0.7]`` means that at the 1st step, we should make
    30% of the agents' maximum need available ; 80% at the 2nd step, and 70%
    at the 3rd step.
    """

    data: np.ndarray
    """
    Data representing how much of the maximum need should be available each step.
    """

    def __init__(self, data):
        data = np.asarray(data)
        assert len(data.shape) == 1
        self._data = data

    def generate_available_energy(self,
                                  current_need: int,
                                  current_step: int,
                                  min_need: int,
                                  max_need: int):
        step = current_step % len(self._data)
        ratio = self._data[step]
        return int(ratio * max_need)

    def available_energy_bounds(self,
                                current_need: int,
                                current_step: int,
                                min_need: int,
                                max_need: int):
        min_ratio = min(self._data)
        max_ratio = max(self._data)
        lower_bound = int(min_ratio * max_need)
        upper_bound = int(max_ratio * max_need)
        return lower_bound, upper_bound
