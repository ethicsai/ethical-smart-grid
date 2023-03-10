from abc import ABC, abstractmethod

from smartgrid.environment import SmartGrid


class Model(ABC):
    def __init__(self, env: SmartGrid, hyper_parameters: dict):
        self.env = env
        self.hyper_parameters = hyper_parameters

    @abstractmethod
    def forward(self, observations_per_agent):
        pass

    @abstractmethod
    def backward(self, observations_per_agent, reward_per_agent):
        pass
