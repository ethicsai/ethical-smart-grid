from smartgrid.rewards.numeric.comfort import Comfort
from smartgrid.rewards.numeric.over_consumption import OverConsumption
from smartgrid.rewards.reward import Reward


class MultiObjectiveSum(Reward):
    """
    Weighted sum calculation.
    """

    def __init__(self):
        super().__init__("MultiObjectiveSum")
        self.coefficient = {
            "Comfort": 0.2,
            "OverConsumption": 0.8
        }
        self.comfort = Comfort()
        self.over_consumption = OverConsumption()

    def calculate(self, world: 'World', agent: 'Agent') -> float:
        sum = self.coefficient["Comfort"] * self.comfort.calculate(world, agent)
        sum += self.coefficient["OverConsumption"] * self.over_consumption.calculate(world, agent)
        return sum
