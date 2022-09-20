"""
Adaptability rewards change their definition as time goes by.

As their name indicates, they allow testing the agents' capability to adapt
to such changes: can their behaviour evolve with the new expectations?

These changes can be incremental, i.e., adding new objectives after some steps,
or more brutal, i.e., completely replacing the targeted objectives by others.
"""

from smartgrid.rewards.numeric.per_agent.comfort import Comfort
from smartgrid.rewards.numeric.differentiated.equity import EquityRewardOne
from smartgrid.rewards.numeric.differentiated.multi_objective_sum import MultiObjectiveSum
from smartgrid.rewards.numeric.differentiated.over_consumption import OverConsumption
from smartgrid.rewards.reward import Reward


class AdaptabilityOne(Reward):
    """
    AdaptabilityOne relies on *equity*, *comfort* and *over-consumption*.

    - When the step is inferior to ``3000``, this function is the same as
      :py:class:`.EquityRewardOne` (which relies on *equity*).

    - Otherwise, this function is the same as :py:class:`.MultiObjectiveSum`
      (which relies on *comfort* and *over-consumption*).
    """

    def __init__(self):
        super().__init__("AdaptabilityOne")
        self.equity = EquityRewardOne()
        self.mos = MultiObjectiveSum()

    def calculate(self, world: 'World', agent: 'Agent') -> float:
        if world.current_step < 3000:
            return self.equity.calculate(world, agent)
        else:
            return self.mos.calculate(world, agent)


class AdaptabilityTwo(Reward):
    """
    AdaptabilityTwo relies on *equity* and *over-consumption*.

    - When the current step is inferior to ``2000``, this function is the same
      as :py:class:`.EquityRewardOne`.

    - Otherwise, it computes the mean of :py:class:`.EquityRewardOne` and
      :py:class:`.OverConsumption`.
    """

    def __init__(self):
        super().__init__("AdaptabilityTwo")
        self.equity = EquityRewardOne()
        self.over_consumption = OverConsumption()

    def calculate(self, world: 'World', agent: 'Agent') -> float:
        if world.current_step < 2000:
            return self.equity.calculate(world, agent)
        else:
            equity = self.equity.calculate(world, agent)
            oc = self.over_consumption.calculate(world, agent)
            return (equity + oc) / 2


class AdaptabilityThree(Reward):
    """
    AdaptabilityThree relies on *equity*, *over-consumption*, and *comfort*.

    - When the step is inferior to ``2000``, this function is the same as
      :py:class:`.EquityRewardOne`.

    - When the step is inferior to ``6000``, it computes the mean of
      :py:class:`.EquityRewardOne` and :py:class:`.OverConsumption`.

    - Otherwise, it computes the mean of :py:class:`.EquityRewardOne`,
      :py:class:`.OverConsumption`, and :py:class:`.Comfort`.
    """

    def __init__(self):
        super().__init__("AdaptabilityThree")
        self.adaptability = AdaptabilityTwo()
        self.comfort = Comfort()

    def calculate(self, world: 'World', agent: 'Agent') -> float:
        adaptability = self.adaptability.calculate(world, agent)
        if world.current_step < 6000:
            return adaptability
        else:
            equity_and_oc = adaptability * 2
            comfort = self.comfort.calculate(world, agent)
            return (equity_and_oc + comfort) / 3
