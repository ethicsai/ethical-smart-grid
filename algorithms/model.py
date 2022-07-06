from abc import ABC, abstractmethod

from smartgrid.environment import SmartGrid

from algorithms.memory import Memory


class Model(ABC):
    def __init__(self, agent_num: int, env: SmartGrid, hyper_parameters: dict, device: str):
        self.agent_num = agent_num
        self.env = env
        self.hyper_parameters = hyper_parameters
        self.device = device
        self.memory = Memory(agent_num)

    def reminder(self, step, agent_id, action, observation, global_observation, reward, done):
        self.memory.add_step(step, agent_id, action, observation, global_observation, reward, done)

    @abstractmethod
    def forward(self, observations_per_agent):
        pass

    @abstractmethod
    def backward(self, observations_per_agent, reward_per_agent):
        pass

    @abstractmethod
    def save(self, param):
        pass
