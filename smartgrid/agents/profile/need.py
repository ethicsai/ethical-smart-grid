"""
Determine the energy needed by an Agent for each step of the simulation.
"""

from typing import List


class NeedProfile:
    """
    Determines the energy needed by an Agent for each step of the simulation.

    We recall that, at each time step, all Agents consume energy to satisfy
    their need. In other words, the *need* acts as some sort of target, and
    Agents aim to get as close as possible to this target, while taking
    into account their other ethical considerations (e.g., equity).

    The need may differ between Agents and time steps; to offer variety,
    and yet a common structure, we introduce the *NeedProfile*.
    Intuitively, it represents some sort of distribution from which needs
    are drawn.

    This class uses a data array as source of needs (see :py:attr:`.need_per_hour`
    for details), but can be extended, e.g., for introducing stochasticity.
    However, it is important to specify the :py:attr:`.max_energy_needed`
    attribute, as it is used in other components of the simulator, e.g.,
    to determine the quantity of energy that the grid should generate at
    each time step (see :py:mod:`smartgrid.util.available_energy`).

    The *NeedProfile* is usually constructed in the :py:class:`.DataConversion`
    class, and is part of the :py:class:`.AgentProfile`.
    """

    need_per_hour: List[float]
    """
    Array of needs (one for each time step).
    
    We assume that a time step represents an hour; this array should contain
    the expected need of an Agent for each hour.
    For example, if the array is ``[800, 600, 1200]``, it means that the
    Agent's need will be, during the first hour (t=0), ``800``, then ``600``
    during the second hour (t=1), and ``1200`` during the third hour (t=2).
    
    This class automatically handles out-of-bounds time steps, simply by
    cycling over the array. Thus, in the previous example, the need at time
    step t=3 will be the same as t=(3 % 3)=0, i.e., ``800``.
    This allows for any number of time steps in the simulation, whereas the
    *NeedProfile* itself has a fixed length.
    
    Typically, this array should have a coherent number of elements, such as
    ``24`` (a *daily* profile, with a need for each hour of a single day),
    or ``365 * 24 = 8760`` (an *annual* profile, with a need for each hour of
    each day). However, this is not a hard requirement, any length would work.
    """

    max_energy_needed: float
    """
    Maximum amount of energy that can be needed through this *NeedProfile*.
    
    This value is used by other components of the simulator, such as the
    :py:class:`.EnergyGenerator`, which is based on the total need from all
    agents.
    """

    def __init__(self, need_per_hour):
        """
        Create a *NeedProfile*.

        :param need_per_hour: The list (array) of needs, i.e., floats, for
            each hour, such that the 1st element represents the 1st hour, and
            so on.
        """
        self.need_per_hour = need_per_hour
        self.max_energy_needed = max(need_per_hour)

    def compute(self, step=0) -> float:
        """
        Determine the need at a given time step.

        :param step: The desired time step.
        :return: The need for this time step.
        """
        step %= len(self.need_per_hour)
        return self.need_per_hour[step]
