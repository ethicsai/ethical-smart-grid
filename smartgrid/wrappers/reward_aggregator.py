"""
RewardAggregators wrap the multi-objective env into a single-objective by
aggregating rewards (e.g., using an average, min, weighted sum, ...).
"""

import warnings
from abc import ABC, abstractmethod
from typing import List, Dict

import numpy as np
from gymnasium.core import RewardWrapper
from numpy import ndarray

from smartgrid.environment import SmartGrid


class RewardAggregator(ABC, RewardWrapper):
    """
    Wraps the multi-objective env into a single-objective by aggregating rewards.

    The :py:class:`smartgrid.environment.SmartGrid` environment supports
    multiple reward functions; its :py:meth:`.SmartGrid.step` method returns
    a list of dictionaries, one dict for each agent, containing the rewards
    indexed by their reward function's name.
    However, most Reinforcement Learning algorithms expect a scalar reward,
    or in this case, a list of scalar rewards, one for each agent.

    Classes that extend the ``RewardAggregator`` bridge this gap, by
    aggregating (scalarizing) the multiple rewards into a single one.
    """

    def __init__(self, env: SmartGrid):
        super().__init__(env)

    @abstractmethod
    def reward(self, rewards: List[Dict[str, float]]) -> List[float]:
        """
        Transform multi-objective rewards into single-objective rewards.

        :param rewards: A list of dicts, one dict for each learning agent. Each
            dict contains one or several rewards, indexed by their reward
            function's name, e.g., ``{ 'fct1': 0.8, 'fct2': 0.4 }``.

        :return: A list of scalar rewards, one for each agent. The rewards
            are scalarized from the dict.
        """
        pass

    def __str__(self):
        return type(self).__name__


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

    def reward(self, rewards: List[Dict[str, float]]) -> List[float]:
        return [
            list(agent_rewards.values())[0]
            for agent_rewards in rewards
        ]


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
        self.coefficients = coefficients

    def reward(self, rewards: List[Dict[str, float]]) -> List[float]:
        scalarized_rewards = []
        for agent_rewards in rewards:
            scalar = 0.0
            for reward_name, reward_value in agent_rewards.items():
                # We set a default in case the coefficient was not set.
                coeff = self.coefficients.get(reward_name, 0.0)
                scalar += reward_value * coeff
            scalarized_rewards.append(scalar)
        return scalarized_rewards


class MinRewardAggregator(RewardAggregator):
    """
    Returns the minimum of the rewards to scalarize.

    This corresponds to some sort of "Aristotelian" ethics, in the sense that
    we put the focus on the reward function with the worst consequences.
    """

    def __init__(self, env: SmartGrid):
        super().__init__(env)

    def reward(self, rewards: List[Dict[str, float]]) -> List[float]:
        return [
            min(agent_rewards.values())
            for agent_rewards in rewards
        ]


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

    def reward(self, reward: List[Dict[str, float]]) -> List[ndarray]:
        return [
            np.prod(list(agent_rewards.values()), axis=0)
            for agent_rewards in reward
        ]
