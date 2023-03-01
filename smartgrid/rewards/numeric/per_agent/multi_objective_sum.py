from smartgrid.rewards.numeric.per_agent.comfort import Comfort
from smartgrid.rewards.numeric.per_agent.over_consumption import OverConsumptionPerAgent
from smartgrid.rewards.reward import Reward


class MultiObjectiveSumPerAgent(Reward):
    """
    Weighted sum calculation.
    Depends on :py:class:`.Comfort` and :py:class:`.OverConsumptionPerAgent`.
    :py:attr:`.coefficient` can be changed by the extension of this class.
    """

    def __init__(self):
        super().__init__("MultiObjectiveSumPerAgent")
        self.coefficient = {
            "Comfort": 0.2,
            "OverConsumptionPerAgent": 0.8
        }
        self.comfort = Comfort()
        self.over_consumption = OverConsumptionPerAgent()

    def calculate(self, world: 'World', agent: 'Agent') -> float:
        to_return = self.coefficient["Comfort"] * self.comfort.calculate(world, agent)
        to_return += self.coefficient["OverConsumptionPerAgent"] * self.over_consumption.calculate(world, agent)
        return to_return
