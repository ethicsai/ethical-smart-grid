from smartgrid.rewards.numeric.per_agent.comfort import Comfort
from smartgrid.rewards.numeric.per_agent.over_consumption import OverConsumptionPerAgent
from smartgrid.rewards.reward import Reward


class MultiObjectiveSumPerAgent(Reward):
    """
    Weighted sum of OverConsumptionPerAgent and Comfort.

    The reward is equal to ``0.2 * comfort + 0.8 * overconsumption``, where
    ``comfort`` refers to the reward of :py:class:`.Comfort`, and
    ``overconsumption`` refers to the reward of
    :py:class:`.OverConsumptionPerAgent`.

    The coefficients (``0.2`` and ``0.8``) can be configured in the constructor.
    Note that, in this case, the sum of coefficients *should* be equal to ``1``,
    in order to have a weighted average, but this is not strictly mandatory.
    """

    name: str

    def __init__(self, coefficients=None):
        super().__init__()
        if coefficients is None:
            coefficients = {
                'Comfort': 0.2,
                'OverConsumptionPerAgent': 0.8
            }
        self.coefficients = coefficients
        self.comfort = Comfort()
        self.over_consumption = OverConsumptionPerAgent()

    def calculate(self, world, agent) -> float:
        comfort = self.coefficients['Comfort'] * self.comfort.calculate(world, agent)
        oc = self.coefficients['OverConsumptionPerAgent'] * self.over_consumption.calculate(world, agent)
        return comfort + oc
