"""
Determines the energy produced by an Agent for each step of the simulation.
"""

from typing import List


class ProductionProfile:
    """
    Determines the energy produced by an Agent for each step of the simulation.

    At each time step, all Agents produce a (small) quantity of energy that
    is put directly into their personal battery, and available only to them.
    This quantity is separated from the grid's production itself. Intuitively,
    this local production can be, e.g., from a small photovoltaic (PV) panel,
    whereas the grid has more important means of production, e.g., a
    hydraulic power plant.

    This production may differ:

    - between Agents, some of them may have more surface, and thus a bigger
      PV panel, which produces more energy;
    - between time steps, as the production can fluctuate. A realistic PV panel
      should produce less at night, for example.

    The *ProductionProfile* is introduced to offer a common structure that
    Agents can rely upon while offering variety between different types of
    productions.
    Intuitively, it represents some sort of distribution from which productions
    are drawn.

    This class uses a data array as source of productions (see
    :py:attr:`.production_per_hour` for details), but can be extended, e.g.,
    for introducing stochasticity.

    The *ProductionProfile* is usually constructed in the
    :py:class:`.DataConversion` class, and is part of the
    :py:class:`.AgentProfile`.
    """

    production_per_hour: List[float]
    """
    Array of productions (one for each time step).
    
    We assume that a time step represents an hour; this array should contain
    the expected production of an Agent for each hour.
    For example, if the array is ``[40, 70, 30]``, it means that the
    Agent's production will be, during the first hour (t=0), ``40``, then 
    ``70`` during the second hour (t=1), and ``30`` during the third hour (t=2).
    
    This class automatically handles out-of-bounds time steps, simply by
    cycling over the array. Thus, in the previous example, the production at 
    time step t=3 will be the same as t=(3 % 3)=0, i.e., ``40``.
    This allows for any number of time steps in the simulation, whereas the
    *ProductionProfile* itself has a fixed length.
    
    Typically, this array should have a coherent number of elements, such as
    ``24`` (a *daily* profile, with a production for each hour of a single day),
    or ``365 * 24 = 8760`` (an *annual* profile, with a production for each hour
    of each day). However, this is not a hard requirement, any length would work.
    """

    def __init__(self, production_per_hour):
        """
        Create a *ProductionProfile*.

        :param production_per_hour: The list (array) of production, i.e., floats,
            for each hour, such that the 1st element represents the 1st hour,
            and so on.
        """
        self.production_per_hour = production_per_hour

    def compute(self, step=0) -> float:
        """
        Determine the production at a given time step.

        :param step: The desired time step.
        :return: The production for this time step.
        """
        step %= len(self.production_per_hour)
        return self.production_per_hour[step]
