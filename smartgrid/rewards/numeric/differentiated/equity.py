from smartgrid.rewards.reward import Reward
from smartgrid.util.equity import hoover


class Equity(Reward):
    """
    Reward based on the equity of comforts.

    We first get the comfort of all agents in the society, and we compute the
    equity of these comforts (see the :py:mod:`smartgrid.util.equity` and
    especially :py:func:`smartgrid.util.equity.hoover` for details). This gives
    us a *global* component (the current environment).

    Then, we compute the equity only for the *others*' comforts, i.e., all
    agents except the one being currently judged. This gives us a *local*
    component (the hypothetical environment, had the agent not acted).

    The reward follows the Difference Reward principle, and thus is the global
    component minus the local component.
    """

    name: str

    def __init__(self):
        super().__init__()

    def calculate(self, world, agent):
        # Comforts of all other agents (excluding the current `agent`)
        other_comforts = [a.state.comfort for a in world.agents if a != agent]
        # Comfort of the current agent
        agent_comfort = agent.state.comfort

        # Compute the equity in the actual environment (others + agent)
        # we use 1-x since hoover returns 0=equity and 1=inequity
        actual_equity = 1.0 - hoover(other_comforts + [agent_comfort])

        # Compute the equity in the hypothetical environment
        hypothetical_equity = 1.0 - hoover(other_comforts)

        # Return the difference between the 2 environments
        return actual_equity - hypothetical_equity
