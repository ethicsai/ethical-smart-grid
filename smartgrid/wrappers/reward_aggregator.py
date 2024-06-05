"""
RewardAggregators wrap the multi-objective env into a single-objective by
aggregating rewards (e.g., using an average, min, weighted sum, ...).
"""

import warnings
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple

import numpy as np
from pettingzoo.utils.env import ActionDict, ObsDict

from smartgrid.environment import SmartGrid, RewardsDict, InfoDict, AgentID


class RewardAggregator(ABC, SmartGrid):
    """
    Wraps the multi-objective env into a single-objective by aggregating rewards.

    The :py:class:`smartgrid.environment.SmartGrid` environment supports
    multiple reward functions; its :py:meth:`.SmartGrid.step` method returns
    a dict of dictionaries, one dict for each agent, containing the rewards
    indexed by their reward function's name.
    However, most Reinforcement Learning algorithms expect a scalar reward,
    or in this case, a dict of scalar rewards, one for each agent.

    Classes that extend the ``RewardAggregator`` bridge this gap, by
    aggregating (scalarizing) the multiple rewards into a single one.

    .. note: PettingZoo only supports wrappers for AEC environments. AEC can
        be converted back-and-forth to Parallel environments, but that would
        hinder the performances. This class is a simpler wrapper for Parallel
        environments, although it does not follow PettingZoo's
        :py:class:`~pettingzoo.utils.wrappers.base.BaseWrapper` conventions.
    """

    def __init__(self, env: SmartGrid):
        self._env = env

    @abstractmethod
    def reward(self, rewards: RewardsDict) -> Dict[AgentID, float]:
        """
        Transform multi-objective rewards into single-objective rewards.

        :param rewards: A dict mapping each learning agent to its rewards.
            The rewards are represented as a dict themselves (dict of dicts),
            containing one or several rewards, indexed by their reward
            function's name, e.g., ``{ 'fct1': 0.8, 'fct2': 0.4 }``.

        :return: A dict mapping each agent to its scalar reward. The rewards
            are scalarized from the agents' dict of rewards.
        """
        pass

    def step(self, actions: ActionDict) -> Tuple[
        ObsDict, Dict[AgentID, float], Dict[AgentID, bool], Dict[AgentID, bool], InfoDict
    ]:
        obs, rewards, terminated, truncated, infos = self._env.step(actions)
        rewards = self.reward(rewards)
        return obs, rewards, terminated, truncated, infos

    def __getattribute__(self, name: str) -> Any:
        # Allow to use this wrapper exactly as the wrapped environment.
        # `getattribute` is similar to `getattr` but is called for *any*
        # attribute, even those that can be found in the class (e.g., through
        # inheritance). `getattr` is only called when the attribute is not found.
        if name.startswith('_') or name in ['reward', 'step', 'unwrapped']:
            # Private attribute or an attribute defined in this Wrapper class.
            # We want to directly access it (from this instance), not from the
            # wrapped env.
            return object.__getattribute__(self, name)
        else:
            # Another attribute: try to access it from the wrapped env.
            return object.__getattribute__(self._env, name)

    @property
    def unwrapped(self) -> SmartGrid:
        return self._env

    def __str__(self):
        """Return a name that looks like: ``Wrapper<WrappedEnv>``."""
        return f'{type(self).__name__}<{type(self.unwrapped).__name__}>'

    __repr__ = __str__


class SingleRewardAggregator(RewardAggregator):
    """
    Returns the single reward for simplicity.

    This wrapper can be used when a single reward function is used in the
    environment; although it still returns a dict, the dict consists of a
    single value, and thus the "aggregation" is in fact trivial.

    .. warning:
        This wrapper will raise a warning if multiple reward functions are used.
        In this case, the first reward of the dict will be returned.
    """

    def __init__(self, env: SmartGrid):
        super().__init__(env)
        nb_rewards = len(env.reward_calculator.rewards)
        if nb_rewards > 1:
            warnings.warn(f'Expected 1 reward function, found {nb_rewards}')

    def reward(self, rewards: RewardsDict) -> Dict[AgentID, float]:
        return {
            agent_name: list(agent_rewards.values())[0]
            for agent_name, agent_rewards in rewards.items()
        }


class WeightedSumRewardAggregator(RewardAggregator):
    """
    Scalarizes multiple rewards through a weighted sum.

    By default, coefficients are all equal to ``1/n`` where ``n`` is the number
    of rewards, i.e., this is equivalent to an average.
    """

    def __init__(self, env: SmartGrid, coefficients: dict = None):
        """
        Construct an instance of the Weighted Sum aggregator.

        :param env: The instance of the Smart Grid environment.

        :param coefficients: A dictionary describing the coefficients to use
            for each reward function. The keys must correspond to the name
            of the reward functions in the env
            (see its :py:attr:`.SmartGrid.reward_calculator`), and the values
            must be the weights (floats).
            Usually, the sum of weights is set to ``1.0`` to obtain a weighted
            average, but this is not mandatory.
            By default, weights are set to ``1 / n`` to obtain a simple average.

        .. warning:
            This class will emit a warning if the ``coefficients`` do not
            correspond to the reward functions' names. In this case, the
            coefficient during the computation is assumed to be ``0.0``, i.e.,
            the reward function is ignored.
        """
        super().__init__(env)
        if coefficients is None:
            nb_rewards = len(env.reward_calculator.rewards)
            coefficients = {
                reward.name: 1.0 / nb_rewards
                for reward in env.reward_calculator.rewards
            }
        else:
            # We use sets instead of lists, because we do not care about the order.
            expected_keys = {
                reward.name for reward in env.reward_calculator.rewards
            }
            found_keys = set(coefficients.keys())
            if expected_keys != found_keys:
                warnings.warn(f'Expected {expected_keys}, found {found_keys}')
        self._coefficients = coefficients

    def reward(self, rewards: RewardsDict) -> Dict[AgentID, float]:
        scalarized_rewards = {}
        for agent_name, agent_rewards in rewards.items():
            scalar = 0.0
            for reward_name, reward_value in agent_rewards.items():
                # We set a default in case the coefficient was not set.
                coeff = self._coefficients.get(reward_name, 0.0)
                scalar += reward_value * coeff
            scalarized_rewards[agent_name] = scalar
        return scalarized_rewards


class MinRewardAggregator(RewardAggregator):
    """
    Returns the minimum of the rewards to scalarize.

    This corresponds to some sort of "Aristotelian" ethics, in the sense that
    we put the focus on the reward function with the worst consequences.
    """

    def __init__(self, env: SmartGrid):
        super().__init__(env)

    def reward(self, rewards: RewardsDict) -> Dict[AgentID, float]:
        return {
            agent_name: min(agent_rewards.values())
            for agent_name, agent_rewards in rewards.items()
        }


class ProductRewardAggregator(RewardAggregator):
    """
    Scalarizes rewards by multiplying them together.

    This forces low rewards to have an important impact, because, e.g.,
    ``0.1 * 0.9`` equals to ``0.09``. In other words, a low reward cannot be
    compensated by a high reward (as it would be in an average, for example).

    .. warning:
        This aggregation relies on assumptions that are **only** true when the
        reward range is set to ``[0,1]``!
        Otherwise, the multiplication would still work mathematically, but
        certainly not make sense in terms of a reward function. For example,
        if the reward range is ``[0,5]``, we could have ``5 * 5 = 25``.
        Or, if the reward range is ``[-1,1]``, we could have ``-1 * -1 = 1``,
        i.e., two negative rewards giving a positive scalar...
    """

    def __init__(self, env: SmartGrid):
        super().__init__(env)

    def reward(self, rewards: RewardsDict) -> Dict[AgentID, float]:
        return {
            agent_name: np.prod(list(agent_rewards.values()), axis=0)
            for agent_name, agent_rewards in rewards.items()
        }
