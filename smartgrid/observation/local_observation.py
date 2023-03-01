from collections import namedtuple

from smartgrid.agents import Agent

local_field = [
    'personal_storage',
    'comfort',
    'payoff',
]


class LocalObservation(namedtuple('LocalObservation', local_field)):
    """
    Observations that are local (individual) to a single Agent.

    These observations are not shared with other agents, and contain the
    following measures:

    personal_storage
        The amount of energy currently available in the :py:class:`.Agent`'s
        personal battery.
        This amount is represented as a ratio between 0 (empty) and 1 (full),
        w.r.t. the Agent's battery capacity. See :py:attr:`.Agent.storage_ratio`
        for details.

    comfort
        This represents to which degree the agent satisfied its need by
        consuming energy. Intuitively, the more an agent's consumption is
        close to its need, the closer the comfort will be to 1. Conversely,
        if an agent does not consume, its comfort will tend towards 0.
        Comfort is computed through the Agent's comfort function; we describe
        several examples in the :py:mod:`.comfort` module, which rely on
        *generalized logistic curves* (similar to a sigmoid).

    payoff
        The agent's current amount of money. Money can be won by selling
        energy from the personal battery to the national grid, or lost by
        buying money from the national grid to the personal battery.
        The payoff observation is interpolated from the agent's real payoff
        and the payoff range to obtain a value between 0 (a loss) and 1 (a win),
        with 0.5 being the neutral value (neither win nor loss).
    """

    @classmethod
    def compute(cls, agent: Agent):
        """
        Return local observations for a single agent.

        This function extracts the relevant measures from an :py:class:`.Agent`.
        Most of the computing has already been done in the
        :py:meth:`.Agent.update` and :py:meth:`.Agent.handle_action` methods.

        :type agent: smartgrid.agent.Agent
        :param agent: The Agent for which we want to compute the local
            observations.

        :rtype: LocalObservation
        """
        # Individual data
        personal_storage = agent.storage_ratio
        comfort = agent.comfort
        payoff = agent.payoff_ratio

        return cls(
            personal_storage=personal_storage,
            comfort=comfort,
            payoff=payoff,
        )
