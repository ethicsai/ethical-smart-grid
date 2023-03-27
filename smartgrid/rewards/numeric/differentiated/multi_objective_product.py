from smartgrid.util import interpolate
from smartgrid.rewards.numeric.differentiated.over_consumption import OverConsumption
from smartgrid.rewards.numeric.per_agent.comfort import Comfort
from smartgrid.rewards.reward import Reward


class MultiObjectiveProduct(Reward):
    """
    Product of multiple objectives: *comfort*, and *over-consumption*.

    The reward is equal to ``comfort * overconsumption``, where
    ``comfort`` refers to the reward of :py:class:`.Comfort`, and
    ``overconsumption`` refers to the reward of :py:class:`.OverConsumption`.

    .. note::
        The overconsumption is interpolated from ``[-1, 1]`` to ``[0, 1]``
        to use the same range as the comfort, and avoid "semantic" problems,
        e.g., ``-0.9 * 0.1 = -0.09``, where ``-0.09`` is actually better than
        ``-0.9``, although both rewards were very low.
    """

    name: str

    def __init__(self):
        super().__init__()
        self.comfort = Comfort()
        self.over_consumption = OverConsumption()

    def calculate(self, world, agent):
        comfort = self.comfort.calculate(world, agent)
        oc = self.over_consumption.calculate(world, agent)
        # `oc` is in `[-1, 1]`, needs to be interpolated to `[0,1]`.
        oc = interpolate(oc, (-1, 1), (0, 1))
        return comfort * oc
