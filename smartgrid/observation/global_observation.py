"""
Global observations of the World, shared by all Agents in the smart grid.
"""

from collections import namedtuple

import numpy as np

from smartgrid.util import hoover

global_fields = [
    'hour',
    'available_energy',
    'equity',
    'energy_loss',
    'autonomy',
    'exclusion',
    'well_being',
    'over_consumption',
]


class GlobalObservation(namedtuple('GlobalObservation', global_fields)):
    """
    Global observations of the World, shared by all Agents in the smart grid.

    These observations are not directly linked to a particular agent, but
    rather to the whole society of agents in the :py:class:`.World`, i.e.,
    in this smart grid. Thus, the measures are the same for all agents.

    To optimize computations, we thus create global observations only once
    each step.

    A global observation is a vector containing the following measures:

    hour
        The current hour in the simulated world. It is computed as a ratio
        between 0 and 1, and days are ignored by using a modulo.
        Specifically, assuming that the current time step is ``t``, the *hour*
        measure is computed as ``(t % 24) / 24``.

    available_energy
        The quantity of energy available in the grid, which is accessible to
        all agents. This is a large pool of energy, however they should avoid
        over-consuming it, and take an appropriate quantity so as to let
        other agents profit as well.
        This measure is normalized as a value between 0 and 1, from the *real*
        available quantity, w.r.t. the bounds of energy that could have been
        generated at this step. See the :py:mod:`.energy_generator` module
        for more details on energy generators, and their bounds.

    equity
        The equity of comforts between all agents in the grid, i.e., to which
        degree do they have a similar comfort. It is computed as a statistical
        indicator of dispersion named the
        `Hoover index <https://en.wikipedia.org/wiki/Hoover_index>`_, which
        is a well-known tool in economy, originally made to describe income
        inequality.
        ``equity`` is computed as ``1 - hoover(comforts)``, such that 0
        represents a perfect inequality (one person has everything, the others
        nothing), and 1 a perfect equality (everybody has the same comfort).

    energy_loss
        The quantity of energy that was available to agents, but not used
        (i.e., neither consumed nor stored) at this time step.

    autonomy
        This measure represents the autonomy, or self-sustainability, of the
        smart grid. It is measured based on the transactions (i.e., selling or
        buying energy from and to the national grid), w.r.t. the total amount
        of energy exchanged within the grid (given, stored, consumed).

    exclusion
        The proportion of agents that have a comfort lower than half the median
        of agents' comforts. Such agents are said to be *excluded*.

    well_being
        The median of all agents' comfort. Using a median rather than an average
        reduces the impact of outliers.

    over_consumption
        The quantity of energy that agents have consumed, but was not
        originally available in the grid. We assume that the grid automatically
        bought this missing energy from the national grid.
        It is computed as the sum of energy consumed from the grid and stored
        from the grid, by all agents, minus the sum of energy given by all
        agents, and the energy initially available, divided by the sum of
        energy taken by all agents, to obtain a ratio between 0 and 1.
        If the measure is less than 0, we set it to 0.
    """

    last_step_compute = -1
    """
    Last time step at which global observations were computed.
    
    This is used to optimize the computations and avoid re-computing already
    known observations, since these are the same for all agents at a given
    time step.
    """

    computed = None
    """
    Memoized global observations, computed at the time step indicated by
    :py:attr:`.last_step_compute`.
    """

    @classmethod
    def _is_compute(cls, world: 'World') -> bool:
        """
        Private method to know whether the current step has already been computed.
        """
        return world.current_step == cls.last_step_compute

    @classmethod
    def compute(cls, world: 'World'):
        """
        Return the global observations computed from the World state.

        This method uses memoization through :py:attr:`.computed`,
        :py:meth:`._is_compute` and :py:attr:`.last_step_compute` to avoid
        re-computing already known observations. In such cases, the cached
        instance is returned. Otherwise, measures are computed, and a new
        instance is created, memoized, and returned.

        :type world: smartgrid.world.World
        :param world: The World for which we want to compute the global
            observations.

        :rtype: GlobalObservation
        """
        # return directly if the step have been computed
        if cls._is_compute(world):
            return cls.computed

        # Pre-compute some intermediate data
        comforts = []
        sum_taken, sum_given, sum_transactions, sum_consumed, sum_stored = 0, 0, 0, 0, 0
        for a in world.agents:
            comforts.append(a.state.comfort)
            sum_taken += a.enacted_action.grid_consumption + a.enacted_action.store_energy
            sum_given += a.enacted_action.give_energy
            sum_transactions += a.enacted_action.buy_energy + a.enacted_action.sell_energy
            sum_consumed += a.enacted_action.grid_consumption + a.enacted_action.storage_consumption
            sum_stored += a.enacted_action.store_energy

        # Compute some common measures about env
        hour = (world.current_step % 24) / 24
        available_energy = np.interp(
            world.available_energy,
            world.energy_generator.available_energy_bounds(
                world.current_need,
                world.current_step,
                world.min_needed_energy,
                world.max_needed_energy
            ),
            (0, 1)
        )
        equity = 1.0 - hoover(comforts)

        over_consumption = max(0.0, sum_taken - sum_given - world.available_energy)
        over_consumption /= (sum_taken + 10E-300)

        energy_loss = max(0.0, -over_consumption)

        autonomy = 1.0 - sum_transactions / (sum_consumed + sum_stored
                                             + sum_given + sum_transactions + 10E-300)

        well_being = np.median(comforts)
        if np.isnan(well_being):
            well_being = 0.0

        threshold = well_being / 2
        exclusion = len([c for c in comforts if c < threshold]) / len(comforts)

        cls.last_step_compute = world.current_step
        cls.computed = cls(hour=hour,
                           available_energy=available_energy,
                           equity=equity,
                           energy_loss=energy_loss,
                           autonomy=autonomy,
                           exclusion=exclusion,
                           well_being=well_being,
                           over_consumption=over_consumption,
                           )
        return cls.computed

    @classmethod
    def reset(cls):
        """
        Reset the counter of steps computed, i.e., the memoization.
        """
        cls.last_step_compute = -1
