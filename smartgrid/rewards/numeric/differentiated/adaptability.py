"""
Adaptability rewards change their definition as time goes by.

As their name indicates, they allow testing the agents' capability to adapt
to such changes: can their behaviour evolve with the new expectations?

These changes can be incremental, i.e., adding new objectives after some steps,
or more brutal, i.e., completely replacing the targeted objectives by others.
"""

from smartgrid.rewards.numeric.differentiated.equity import EquityRewardOne
from smartgrid.rewards.numeric.differentiated.multi_objective_sum import MultiObjectiveSum
from smartgrid.rewards.numeric.differentiated.over_consumption import OverConsumption
from smartgrid.rewards.numeric.per_agent.comfort import Comfort
from smartgrid.rewards.reward import Reward


class AdaptabilityOne(Reward):
    """
    Adaptability One depends on step for calculating. You have two cases:
        - step is inferior to 3000, you look at the Equity metrics.
        - otherwise, it's a weighted sum.
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
    Adaptability Two depends on step for calculating. You have two cases:
        - step is inferior to 2000, you look at the Equity metrics.
        - otherwise, it's the mean of Equity and OverConsumption
    """

    def __init__(self):
        super().__init__("AdaptabilityTwo")
        self.equity = EquityRewardOne()
        self.over_consumption = OverConsumption()

    def calculate(self, world: 'World', agent: 'Agent') -> float:
        if world.current_step < 2000:
            return self.equity.calculate(world, agent)
        else:
            return (self.equity.calculate(world, agent) + self.over_consumption.calculate(world, agent)) / 2


class AdaptabilityThree(Reward):
    """
    Adaptability Three depends on step for calculating. You have two cases:
        - step is inferior to 2000, you look at the Equity metrics.
        - otherwise, it's the mean of Equity, OverConsumption and Comfort
    """

    def __init__(self):
        super().__init__("AdaptabilityTwo")
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
