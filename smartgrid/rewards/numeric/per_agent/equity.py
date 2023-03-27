from smartgrid.rewards.reward import Reward
from smartgrid.util.equity import hoover


class EquityPerAgent(Reward):
    """
    Reward based on the equity of comforts measure.

    It's a measure of statical dispersion of the Comfort metrics of all agents.
    Instead of comparing the *actual* and *hypothetical* environments (as in
    :py:class:`smartgrid.rewards.numeric.differentiated.equity.Equity`),
    it simply computes the equity (Hoover index) of all agents' comfort.
    """

    name: str

    def __init__(self):
        super().__init__()

    def calculate(self, world, agent):
        # Comforts of all agents
        comforts = [a.state.comfort for a in world.agents]

        # Compute the equity in the actual environment (others + agent)
        # we use 1-x since hoover returns 0=equity and 1=inequity
        actual_equity = 1.0 - hoover(comforts)

        return actual_equity
