"""
The base Model, an abstract class to provide a common (standard) API.
"""
import warnings
from abc import ABC, abstractmethod
from typing import Union, Dict

from smartgrid.environment import SmartGrid, AgentID, ObsDict, ActionDict


class Model(ABC):
    """
    A Model is a class that handles the decision-making.

    It must produce decisions (actions) for all agents in the environment.
    Using a single Model for all agents simplifies the use, e.g., it suffices
    to use ``model = SomeModel()`` and ``actions = model.forward(obs)``, instead
    of looping over all agents. However, it is not strictly necessary to use
    a ``Model`` in the interaction loop: for more complex cases, different
    models can be used together, or functions can be used directly to produce
    actions, etc.

    The goal of this class is to provide a standard API that other learning
    algorithms can follow, or at least take inspiration from, and to simplify
    the use of learning algorithms.
    """

    def __init__(self, env: SmartGrid, hyper_parameters: dict):
        """
        Create a Model, i.e., an entrypoint for the learning algorithm.

        :param env: The environment that the learning algorithm will interact
            with. This is useful for, e.g., accessing the agents' observations
            and actions spaces, knowing the number of agents, etc.
            Note that a :py:class:`~gymnasium.Wrapper` can also be used, such as
            a :py:class:`~smartgrid.wrappers.reward_aggregator.RewardAggregator`.

        :param hyper_parameters: An optional dictionary of hyper-parameters that
            control the creation of the learning agents. For example, the
            learning rate to use, etc. The hyper-parameters themselves are
            specific to the implemented Model.
        """
        self.env = env
        self.hyper_parameters = hyper_parameters

    @abstractmethod
    def forward(self, observations_per_agent: ObsDict) -> ActionDict:
        """
        Decide which actions should be taken, based on observations.

        This method represents the *decision* step.

        :param observations_per_agent: The observations per agent. See the
            :py:meth:`SmartGrid._get_obs() <smartgrid.environment.SmartGrid._get_obs>`
            method for details on its structure. These observations describe
            the current state of the simulator, and are the data used to
            take actions.

        :return: A dict mapping each agent to its action, where an action is a
            list of *action parameters*. See the
            :py:attr:`SmartGrid.action_space <smartgrid.environment.SmartGrid.action_space>`
            for details on the structure of *action parameters*.
        """
        pass

    @abstractmethod
    def backward(self,
                 observations_per_agent: ObsDict,
                 reward_per_agent: Union[Dict[AgentID, Dict[str, float]], Dict[AgentID, float]]):
        """
        Learn (improve) the agents' policies, based on observations and rewards.

        This method represents the *learning* step.

        :param observations_per_agent: The observations per agent, similar
            to those in the :py:meth:`.forward` method. They describe the
            *new* situation that happened after the agents' actions were
            executed in the world.

        :param reward_per_agent: The rewards per agent. They describe the degree
            to which agents' actions were satisfying (interesting), with respect
            to the moral values encoded in the reward functions. If multiple
            reward functions are used, this is a dict of dicts; otherwise, it
            is a dict of floats. See the
            :py:meth:`Smartgrid._get_reward() <smartgrid.environment.SmartGrid._get_reward>`
            for details
        """
        pass

    def get_optimal_actions(self, observations_per_agent: ObsDict) -> ActionDict:
        """
        Return the actions that are considered *optimal*, for each agent.

        In other terms, this method ensures *exploitation*, whereas the
        :py:meth:`~.forward` method encourages *exploitation-exploration*.

        It can be useful after the training phase, for testing purposes.

        :param observations_per_agent: A dictionary mapping agents' name to
            their observations. Exactly as in :py:meth:`~.forward`.

        :return: A dict mapping each agent to its action. Actions have the
            same structure as in :py:meth:`~.forward`, but they should be
            produced with only *exploitation* as a goal, i.e., selecting the
            action that should yield the best reward.

        .. warning:: By default, to ensure that all models will have this method,
            it simply returns the same actions as :py:meth:`~.forward`.
            Models that make a distinction between exploration and exploitation
            should override it.
        """
        warnings.warn(
            f"Model {type(self).__name__} does not override `get_optimal_actions()`."
            f" By default, we return the same actions as `forward()`: the"
            f" actions may not be optimal."
        )
        return self.forward(observations_per_agent)
