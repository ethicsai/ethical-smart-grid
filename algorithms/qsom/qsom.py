"""
This module defines a Q-SOM helper that is used as an entrypoint to simplify
the instantiation of Q-SOM Agents from a Gym Environment.

It handles creating the correct structures, giving the correct parameters, ...
"""
import numpy as np

from algorithms.model import Model
from algorithms.qsom.qsom_agent import QsomAgent
from algorithms.qsom.som import SOM
from algorithms.util.action_perturbator import EpsilonActionPerturbator
from algorithms.util.action_selector import BoltzmannActionSelector
from smartgrid.environment import SmartGrid


class QSOM(Model):

    # todo add Memory
    def __init__(self, agent_num: int, env: SmartGrid, hyper_parameters: dict, device: str):
        super().__init__(agent_num, env, hyper_parameters, device)
        self.n_agents = env.n_agent
        self.qsom_agents = []

        action_selector = BoltzmannActionSelector(self.hyper_parameters["initial_tau"],
                                                  self.hyper_parameters["tau_decay"],
                                                  self.hyper_parameters["tau_decay_coeff"])
        action_perturbator = EpsilonActionPerturbator(self.hyper_parameters["noise"])

        for num_agent in range(self.n_agents):
            obs_space = env.observation_space[num_agent]
            assert len(obs_space.shape) == 1, 'Observation space must be 1D'
            action_space = env.action_space[num_agent]
            assert len(action_space.shape) == 1, 'Action space must be 1D'

            state_som = SOM(12, 12,
                            obs_space.shape[0],
                            sigma=self.hyper_parameters["sigma_state"],
                            learning_rate=self.hyper_parameters["lr_state"])
            action_som = SOM(3, 3,
                             action_space.shape[0],
                             sigma=self.hyper_parameters["sigma_action"],
                             learning_rate=self.hyper_parameters["lr_action"])

            qsom_agent = QsomAgent(obs_space,
                                   action_space,
                                   state_som,
                                   action_som,
                                   action_selector,
                                   action_perturbator,
                                   q_learning_rate=self.hyper_parameters["q_learning_rate"],
                                   q_discount_factor=self.hyper_parameters["q_discount_factor"],
                                   update_all=self.hyper_parameters["update_all"],
                                   use_neighborhood=self.hyper_parameters["use_neighborhood"])

            self.qsom_agents.append(qsom_agent)

    def forward(self, observations_per_agent):
        """Choose an action for each agent, based on their observations."""
        observations_per_agent = [list(observations_per_agent['local'][i]) + list(observations_per_agent['global']) for
                                  i in range(self.agent_num)]
        assert len(observations_per_agent) == self.n_agents
        actions = [
            self.qsom_agents[i].forward(observations_per_agent[i])
            for i in range(self.n_agents)
        ]
        return actions

    def backward(self, new_observations_per_agent, reward_per_agent):
        """Make each agent learn, based on their rewards and observations."""
        new_observations_per_agent = [
            list(new_observations_per_agent['local'][i]) + list(new_observations_per_agent['global']) for i in
            range(self.agent_num)]
        assert len(reward_per_agent) == self.n_agents
        assert len(new_observations_per_agent) == self.n_agents
        for i, agent in enumerate(self.qsom_agents):
            agent.backward(new_observations_per_agent[i],
                           reward_per_agent[i])
        return []

    def save(self, path):
        args = {}
        for i in range(len(self.qsom_agents)):
            dict = self.qsom_agents[i].save(i)
            for key in dict.keys():
                args[key] = dict[key]

        np.savez(path, **args)

    def load(self, path):
        # load agent
        weights = np.load(path + '.npz')
        for i in range(self.n_agents):
            weight = {
                "state" : weights[f"state_%i" % (i)],
                "action" : weights[f"action_%i" % (i)],
                "qtable" : weights[f"qtable_%i" % (i)],
            }
            self.qsom_agents[i].load(weight)