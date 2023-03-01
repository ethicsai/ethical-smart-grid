from smartgrid.rewards.numeric.differentiated.over_consumption import OverConsumption
from smartgrid.rewards.numeric.per_agent.comfort import Comfort
from smartgrid.rewards.reward import Reward


class MultiObjectiveSum(Reward):
    """
    Weighted sum of multiple objectives: *comfort*, and *over-consumption*.

    The reward is equal to ``0.2 * comfort + 0.8 * overconsumption``, where
    ``comfort`` refers to the reward of :py:class:`.Comfort`, and
    ``overconsumption`` refers to the reward of :py:class:`.OverConsumption`.
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
        comfort = self.coefficient["Comfort"] * self.comfort.calculate(world, agent)
        oc = self.coefficient["OverConsumption"] * self.over_consumption.calculate(world, agent)
        return comfort + oc
