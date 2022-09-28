from abc import ABC, abstractmethod

from smartgrid.environment import SmartGrid


class Model(ABC):
    def __init__(self, agent_num: int, env: SmartGrid, hyper_parameters: dict, device: str):
        self.agent_num = agent_num
        self.env = env
        self.hyper_parameters = hyper_parameters
        self.device = device

    @abstractmethod
    def forward(self, observations_per_agent):
        pass

    @abstractmethod
    def backward(self, observations_per_agent, reward_per_agent):
        pass

    @abstractmethod
    def save(self, path):
        pass

    @abstractmethod
    def load(self, path):
        pass
