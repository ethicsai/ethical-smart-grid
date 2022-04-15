from abc import ABC
from typing import List, Dict

import numpy as np
from gym import RewardWrapper

from smartgrid.environment import SmartGrid


class AggregateFunction(ABC, RewardWrapper):
    """
    Wrapper of reward for merging multiple metrics of reward.
    Our context is a System Multi-Agent, so multiple metrics of an Agent make one reward
    """
    name: str

    def reward(self, reward: List[Dict[str, float]]) -> List[float]:
        pass

    def __init__(self, env: SmartGrid, name: str):
        super().__init__(env)
        self.name = name

    def __str__(self):
        return self.name


class BasicAggregateFunction(AggregateFunction):
    """
    The basic Aggregate Function is used for single rewarding.
    """

    def __init__(self, env: SmartGrid):
        super(BasicAggregateFunction, self).__init__(env, "BasicAggregateFunction")

    def reward(self, reward: List[Dict[str, float]]) -> List[float]:
        return [list(r.values())[0] for r in reward if len(r) == 1]


class MultiObjectiveProduct(AggregateFunction):
    """
    Multiply all rewards into a single.
    """

    def __init__(self, env: SmartGrid):
        super(MultiObjectiveProduct, self).__init__(env, "MultiObjectiveProduct")

    def reward(self, reward: List[Dict[str, float]]) -> List[float]:
        return [np.prod(list(r.values()),axis=0) for r in reward]
