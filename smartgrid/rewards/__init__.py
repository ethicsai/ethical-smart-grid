"""
This package contains various reward functions, used to compute a
reward signal for each agent, based on their action and the resulting
state of the environment.

Reward functions indicate to which degree the agent's actions were appropriate,
or in this case, *ethical*. By *ethical*, we mean that they are aligned with
one or several moral values. These moral values are encoded in the reward
function itself, which guides the learning of agents.

Traditionally, reward functions in the Reinforcement Learning literature are
purely numerical, i.e., based on mathematical expressions. Such functions can
be found in the :py:mod:`smartgrid.rewards.numeric` package.

Other functions can also be based on symbolic reasoning. (Work in progress)
"""

from .reward import Reward
from .reward_collection import RewardCollection
