"""
This module defines classes to generate an amount of available energy.

Each step, the World "produces" a certain amount of energy, which is made
available to the agents. Various methods can be used to determine the amount:

- a random percent based on the agents' needs, for example an amount between
  80% and 120% of their total need.
- a scarcity variation, similar to the 1st one but with a random between
  60% and 80%.
- a generous variation, similar to the 1st one but with a random between
  100% and 120%.
- a realistic variation, using real data.

All these methods lead to different bounds for the amount of available energy.

Knowing these bounds, and especially the upper one (we can also assume `0` for
the lower bound), allows us to scale the amount of available energy to `[0,1]`
when computing the `Observation`s.

Therefore, instead of using a simple function to generate this amount,
we use a class that defines 2 functions, one for generating the amount,
and the other to return the bounds.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import random
from typing import Tuple

import numpy as np


class EnergyGenerator(ABC):

    @abstractmethod
    def generate_available_energy(self, world: 'World') -> int:
        pass

    @abstractmethod
    def available_energy_bounds(self, world: 'World') -> Tuple[int, int]:
        pass


class RandomEnergyGenerator(EnergyGenerator):
    """
    Generate a random amount, with respect to the agents' max energy needed.

    Assuming that the total maximum energy needed is `M`, that we want at least
    a lower bound of L=80% (i.e., L=0.8), and an upper bound of U=120% (i.e.,
    U=1.2), this class returns amounts in the interval `[L*M, U*M]`.

    Lower and upper bounds are configurable.
    """

    def __init__(self,
                 lower_proportion=0.8,
                 upper_proportion=1.2,
                 ):
        self._lower = lower_proportion
        self._upper = upper_proportion

    def generate_available_energy(self, world: 'World'):
        lower_bound, upper_bound = self.available_energy_bounds(world)
        return random.randint(lower_bound, upper_bound)

    def available_energy_bounds(self, world: 'World'):
        max_needed = world.max_needed_energy
        lower_bound = int(self._lower * max_needed)
        upper_bound = int(self._upper * max_needed)
        return lower_bound, upper_bound


class ScarceEnergyGenerator(RandomEnergyGenerator):
    """
    Similar to the `RandomEnergyGenerator`, but simulating scarcity.

    In practice, the bounds are set to [60%, 80%].
    Note that, as the upper bound is set to less 100% of the max, we
    force conflicts between agents by not giving them enough.
    """

    def __init__(self):
        super(ScarceEnergyGenerator, self).__init__(lower_proportion=0.6,
                                                    upper_proportion=0.8)


class GenerousEnergyGenerator(RandomEnergyGenerator):
    """
    Similar to the `RandomEnergyGenerator`, but simulating a generous env.

    In practice, the bounds are set to [100%, 120%].
    Note that, as the lower bound is set to 100% of the max, we always
    have enough energy available for all agents.
    """

    def __init__(self):
        super(GenerousEnergyGenerator, self).__init__(lower_proportion=1.0,
                                                      upper_proportion=1.2)


class RealisticEnergyGenerator(EnergyGenerator):
    """
    A realistic generator that generates energy based on real-world data.

    The `data` parameter should be a NumPy ndarray giving the ratio of
    energy for each step, with respect to the maximum amount of energy
    needed by the Agents.

    For example, `[0.3, 0.8, 0.7]` means that at the 1st step, we should make
    30% of the agents' total need available ; 80% at the 2nd step, and 70%
    at the 3rd step.
    """

    def __init__(self, data):
        data = np.asarray(data)
        assert len(data.shape) == 1
        self._data = data

    def generate_available_energy(self, world: World):
        step = world.current_step % len(self._data)
        ratio = self._data[step]
        max_needed = world.max_needed_energy
        return int(ratio * max_needed)

    def available_energy_bounds(self, world: World):
        max_needed = world.max_needed_energy
        min_ratio = min(self._data)
        max_ratio = max(self._data)
        lower_bound = int(min_ratio * max_needed)
        upper_bound = int(max_ratio * max_needed)
        return lower_bound, upper_bound
