from rewards.numeric.per_agent.comfort import Comfort
from rewards.numeric.per_agent.equity import EquityRewardPerAgent
from rewards.numeric.per_agent.multi_objective_sum import MultiObjectiveSumPerAgent
from rewards.numeric.per_agent.over_consumption import OverConsumptionPerAgent
from rewards.reward import Reward


class AdaptabilityOnePerAgent(Reward):
    """
    AdaptabilityOnePerAgent depends on step for calculating. You have two cases:
        - step is inferior to 3000, you look at :py:class:`.EquityRewardPerAgent`.
        - otherwise, it's :py:class:`.MultiObjectiveSumPerAgent` that calculate the reward.
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
    AdaptabilityTwoPerAgent depends on step for calculating. You have two cases:
        - step is inferior to 2000, you look at :py:class:`.EquityRewardPerAgent`.
        - otherwise, it's mean of :py:class:`.OverConsumptionPerAgent` and :py:class:`.EquityRewardPerAgent`.
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
    AdaptabilityThreePerAgent depends on step for calculating. You have two cases:
        - step is inferior to 2000, you look at :py:class:`.EquityRewardPerAgent`.
        - step is inferior to 6000, it's mean of :py:class:`.OverConsumptionPerAgent` and
         :py:class:`.EquityRewardPerAgent`.
        - otherwise, it's mean of :py:class:`.OverConsumptionPerAgent`, :py:class:`.EquityRewardPerAgent`
         and :py:class:`.Comfort`.
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
            total = adaptability * 2
            total += self.comfort.calculate(world, agent)
            return total / 3
