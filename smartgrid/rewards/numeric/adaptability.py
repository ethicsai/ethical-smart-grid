from smartgrid.rewards.numeric.comfort import Comfort
from smartgrid.rewards.numeric.equity import EquityRewardOne
from smartgrid.rewards.numeric.multi_objective_sum import MultiObjectiveSum
from smartgrid.rewards.numeric.over_consumption import OverConsumption
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
            return self.equity.calculate(world,agent)
        else:
            return self.mos.calculate(world,agent)


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
            return self.equity.calculate(world,agent)
        else:
            sum = self.equity.calculate(world,agent)
            sum += self.over_consumption.calculate(world,agent)
            return sum/2


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
        adaptability = self.adaptability.calculate(world,agent)
        if world.current_step < 6000:
            return adaptability
        else:
            sum = adaptability * 2
            sum += self.comfort.calculate(world,agent)
            return sum/3
