from smartgrid.rewards.numeric.per_agent.comfort import Comfort
from smartgrid.rewards.numeric.differentiated.over_consumption import OverConsumption
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
        to_return = self.coefficient["Comfort"] * self.comfort.calculate(world, agent)
        to_return += self.coefficient["OverConsumption"] * self.over_consumption.calculate(world, agent)
        return to_return