from abc import ABC, abstractmethod


class Reward(ABC):
    """
    The Reward is responsible for computing a unique reward for an agent. The list of reward used is passed in parameter
    of :py:class:`.World`.

    The computing calls (:py:meth:`.calculate`) and return a single floating value.
    Multiple reward will be computed by :py:class:`.RewardCollection` and aggregate with :py:class:`.AggregateFunction`.
    Creation of a new unique Reward passed by the extension of this class.
    """
    name: str

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def calculate(self, world: 'World', agent: 'Agent') -> float:
        """
        Compute the reward for an Agent.

        :param world: reference on the world for global information.
        :param agent: reference on the agent for local information.
        :return: The value of the reward.
        """
        pass

    def __str__(self):
        return self.name
