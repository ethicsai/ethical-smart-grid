from smartgrid.rewards.numeric.per_agent.comfort import Comfort
from smartgrid.rewards.numeric.differentiated.equity import EquityRewardOne
from smartgrid.rewards.numeric.differentiated.multi_objective_sum import MultiObjectiveSum
from smartgrid.rewards.numeric.differentiated.over_consumption import OverConsumption
from smartgrid.rewards.numeric.per_agent.equity import EquityRewardPerAgent
from smartgrid.rewards.numeric.per_agent.multi_objective_sum import MultiObjectiveSumPerAgent
from smartgrid.rewards.numeric.per_agent.over_consumption import OverConsumptionPerAgent
from smartgrid.rewards.reward import Reward


class AdaptabilityOnePerAgent(Reward):
    """
    Adaptability One depends on step for calculating. You have two cases:
        - step is inferior to 3000, you look at the Equity metrics.
        - otherwise, it's a weighted sum.
    """

    def __init__(self):
        super().__init__("AdaptabilityOnePerAgent")
        self.equity = EquityRewardPerAgent()
        self.mos = MultiObjectiveSumPerAgent()

    def calculate(self, world: 'World', agent: 'Agent') -> float:
        if world.current_step < 3000:
            return self.equity.calculate(world, agent)
        else:
            return self.mos.calculate(world, agent)


class AdaptabilityTwoPerAgent(Reward):
    """
    Adaptability Two depends on step for calculating. You have two cases:
        - step is inferior to 2000, you look at the Equity metrics.
        - otherwise, it's the mean of Equity and OverConsumption
    """

    def __init__(self):
        super().__init__("AdaptabilityTwoPerAgent")
        self.equity = EquityRewardPerAgent()
        self.over_consumption = OverConsumptionPerAgent()

    def calculate(self, world: 'World', agent: 'Agent') -> float:
        if world.current_step < 2000:
            return self.equity.calculate(world, agent)
        else:
            return (self.equity.calculate(world, agent) + self.over_consumption.calculate(world, agent)) / 2


class AdaptabilityThreePerAgent(Reward):
    """
    Adaptability Three depends on step for calculating. You have two cases:
        - step is inferior to 2000, you look at the Equity metrics.
        - otherwise, it's the mean of Equity, OverConsumption and Comfort
    """

    def __init__(self):
        super().__init__("AdaptabilityThreePerAgent")
        self.adaptability = AdaptabilityTwoPerAgent()
        self.comfort = Comfort()

    def calculate(self, world: 'World', agent: 'Agent') -> float:
        adaptability = self.adaptability.calculate(world, agent)
        if world.current_step < 6000:
            return adaptability
        else:
            sum = adaptability * 2
            sum += self.comfort.calculate(world, agent)
            return sum / 3
