from abc import ABC, abstractmethod

from smartgrid.world import World
from smartgrid.agents import Agent


class Reward(ABC):
    """
    The Reward function is responsible for computing a reward for each agent.

    The reward is a signal telling the agent to which degree it performed
    correctly, with respect to the objective(s) specified by the reward
    function.

    Reward functions should judge the agent's behaviour, based on its
    actions and/or the action's consequences on the world (state).

    The actuel reward function is defined in :py:meth:`.calculate`; a simple
    function could be used instead, but using classes allows for easier
    extensions, and using attributes for complex computations.

    A reward function is identified by its :py:attr:`.name` (by default,
    the class name); this name is particularly used when multiple reward
    functions are used (multi-objective reinforcement learning).
    """

    name: str
    """Uniquely identifying, human-readable name for this reward function."""

    def __init__(self, name: str = None):
        if name is None:
            name = type(self).__name__
        self.name = name

    @abstractmethod
    def calculate(self, world: World, agent: Agent) -> float:
        """
        Compute the reward for a specific Agent at the current time step.

        :param world: The World, used to get the current state and determine
            consequences of the agent's action.

        :param agent: The Agent that is rewarded, used to access particular
            information about the agent (personal state) and its action.

        :return: A reward, i.e., a single value describing how well the agent
            performed. The higher the reward, the better its action was.
            Typically, a value in [0,1] but any range can be used.
        """
        pass

    def __str__(self):
        return 'Reward<{}>'.format(self.name)

    __repr__ = __str__
