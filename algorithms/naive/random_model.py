from random import random

from algorithms.model import Model

rand_experiment = {  # change: gamma
    "name": "random",
}


class RandomModel(Model):

    def __init__(self, agent_num, env, hyper_parameters: dict, device):
        super().__init__(agent_num, env, hyper_parameters, device)
        self.agent_num = agent_num

        self.action_spaces = env.action_space

    def forward(self, observations_per_agent):
        actions = []
        for i in range(self.agent_num):
            action = [random() for _ in range(self.action_spaces[i].shape[0])]
            actions.append(action)

        return actions

    def backward(self, observations_per_agent, reward_per_agent):
        pass

    def save(self, path):
        pass

    def load(self, path):
        pass
